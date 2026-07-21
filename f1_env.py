"""
Ambiente Gymnasium per il progetto di Reinforcement Learning su racing.

L'ambiente mantiene invariati dinamica, osservazione e reward del progetto.
L'unica estensione necessaria al training multi-circuito è il campo
``track_name`` inserito negli ``info``: il callback può così separare le
metriche prodotte dai diversi ambienti PPO.

Durante ``watch`` il rendering mostra la raceline esterna al posto della
centerline. La raceline non entra mai nell'osservazione, nella reward o nella
dinamica: è soltanto un riferimento grafico post-training.

Nota importante su width_scale:
    width_scale è usato solo per il rendering. Non modifica larghezze fisiche,
    collisioni, off-track, reward o osservazioni dell'agente.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from car_dynamics import CarDynamics, LocalStateInfo
from reward import compute_racing_reward


WATCH_RENDER_REFERENCE = "raceline"


class SimpleF1Env(gym.Env):
    """Ambiente Gymnasium con azioni continue ``[throttle, steering]``."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        track_path: str,
        render_mode: str | None = None,
        max_steps: int = 5000,
        width_scale: float = 6.0,
        random_start: bool = False,
        track_name: str | None = None,
        raceline_path: str | None = None,
    ):
        super().__init__()

        self.track_path = str(track_path)
        self.track_name = str(track_name or Path(track_path).stem)
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.dt = 0.1
        self.width_scale = float(width_scale)
        self.random_start = bool(random_start)
        self.current_max_speed = 95.0

        # La raceline è usata esclusivamente nel rendering di watch.
        # Non entra nell'osservazione, nella reward, nella dinamica o nel training.
        self.raceline_path = str(raceline_path) if raceline_path else None
        self.raceline: np.ndarray | None = None

        if self.raceline_path is not None:
            raceline_file = Path(self.raceline_path)
            if not raceline_file.is_file():
                raise FileNotFoundError(f"Raceline non trovata: {raceline_file}")

            # Il database può contenere commenti e colonne aggiuntive.
            # Per il rendering servono soltanto le prime due colonne numeriche x e y.
            raceline_df = pd.read_csv(
                raceline_file,
                comment="#",
                header=None,
                sep=None,
                engine="python",
            )
            if raceline_df.shape[1] < 2:
                raise ValueError(
                    f"La raceline '{raceline_file}' deve contenere almeno due colonne."
                )

            raceline_xy = raceline_df.iloc[:, :2].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(raceline_xy) < 2:
                raise ValueError(
                    f"La raceline '{raceline_file}' non contiene almeno due "
                    "punti numerici validi."
                )

            self.raceline = raceline_xy.to_numpy(dtype=np.float32)

        # ------------------------------------------------------------------
        # Caricamento pista.
        # Formato CSV atteso: x, y, w_right, w_left.
        # ------------------------------------------------------------------
        df = pd.read_csv(
            self.track_path,
            comment="#",
            header=None,
            names=["x", "y", "w_right", "w_left"],
        )

        if len(df) < 3:
            raise ValueError(
                f"La pista '{self.track_path}' contiene meno di tre punti validi."
            )

        self.centerline = df[["x", "y"]].to_numpy(dtype=np.float32)

        # Larghezze FISICHE usate dall'MDP.
        self.w_right = df["w_right"].to_numpy(dtype=np.float32)
        self.w_left = df["w_left"].to_numpy(dtype=np.float32)

        # Larghezze solo VISUALI. Non vengono usate per reward/off-track.
        self.w_right_render = self.w_right * self.width_scale
        self.w_left_render = self.w_left * self.width_scale

        # Precomputazioni geometriche della centerline.
        next_center = np.roll(self.centerline, -1, axis=0)
        segments = next_center - self.centerline
        self.segment_lengths = np.linalg.norm(segments, axis=1).astype(np.float32)
        self.segment_lengths = np.maximum(self.segment_lengths, 1e-6)
        self.track_length = float(np.sum(self.segment_lengths))

        self.tangents = segments / self.segment_lengths[:, None]
        self.headings = np.arctan2(
            self.tangents[:, 1], self.tangents[:, 0]
        ).astype(np.float32)

        self.normals_left = np.zeros_like(self.tangents)
        self.normals_left[:, 0] = -self.tangents[:, 1]
        self.normals_left[:, 1] = self.tangents[:, 0]

        self.s_start = np.zeros(len(self.centerline), dtype=np.float32)
        for i in range(1, len(self.centerline)):
            self.s_start[i] = self.s_start[i - 1] + self.segment_lengths[i - 1]

        next_headings = np.roll(self.headings, -1)
        heading_diff = (
            next_headings - self.headings + np.pi
        ) % (2.0 * np.pi) - np.pi
        self.curvature = (heading_diff / self.segment_lengths).astype(np.float32)

        # Azione: [throttle, steering], entrambi in [-1, 1].
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Osservazione invariata: 8 feature base + 5 punti futuri x/y
        # + 5 curvature future.
        self.observation_dim = (
            8
            + 2 * len(CarDynamics.LOOKAHEAD_DISTANCES)
            + len(CarDynamics.LOOKAHEAD_DISTANCES)
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.car: CarDynamics | None = None
        self.step_count = 0
        self.start_progress = 0.0
        self.prev_progress_abs = 0.0
        self.episode_progress = 0.0
        self.stall_steps = 0

        self.fig, self.ax = None, None
        self.car_dot = None
        self.car_heading_line = None
        self.info_text = None

    def _compute_state_info(self) -> LocalStateInfo:
        assert self.car is not None, (
            "reset() deve essere chiamato prima di _compute_state_info()."
        )
        return self.car.compute_local_state(
            self.centerline,
            self.normals_left,
            self.headings,
            self.w_left,
            self.w_right,
            self.s_start,
            self.curvature,
            self.track_length,
            self.segment_lengths,
        )

    def _progress_delta(self, new_progress_abs: float) -> float:
        """Calcola il progresso incrementale gestendo il passaggio sul traguardo."""
        delta = float(new_progress_abs - self.prev_progress_abs)

        if delta < -0.5 * self.track_length:
            delta += self.track_length
        elif delta > 0.5 * self.track_length:
            delta -= self.track_length

        return delta

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.stall_steps = 0
        self.episode_progress = 0.0

        if self.random_start:
            start_idx = int(self.np_random.integers(0, len(self.centerline)))
        else:
            start_idx = 0

        start_x = float(self.centerline[start_idx, 0])
        start_y = float(self.centerline[start_idx, 1])
        start_yaw = float(self.headings[start_idx])

        self.car = CarDynamics(
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            max_speed=self.current_max_speed,
            initial_speed=0.0,
        )

        state_info = self._compute_state_info()
        self.start_progress = float(state_info.current_progress)
        self.prev_progress_abs = float(state_info.current_progress)
        obs = state_info.observation.astype(np.float32)

        return obs, {
            "track_name": self.track_name,
            "track_path": self.track_path,
            "start_idx": start_idx,
        }

    def step(self, action):
        # Verifica che l'ambiente sia stato inizializzato.
        assert self.car is not None, "Devi chiamare reset() prima di step()."

        self.step_count += 1

        # Converte e limita l'azione [throttle, steering] nell'intervallo [-1, 1].
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Salva i comandi precedenti prima dell'aggiornamento della dinamica.
        previous_steer = float(self.car.prev_steer)
        previous_throttle = float(self.car.prev_throttle)

        throttle = float(action[0])
        steer = float(action[1])

        # Aggiorna velocità, posizione e orientamento della macchina.
        self.car.update_physics(throttle, steer)

        # Calcola la nuova osservazione e le informazioni locali.
        state_info = self._compute_state_info()
        obs = state_info.observation.astype(np.float32)

        # Aggiorna il progresso percorso durante l'episodio.
        delta_progress = self._progress_delta(state_info.current_progress)
        self.prev_progress_abs = float(state_info.current_progress)
        self.episode_progress += delta_progress
        self.episode_progress = float(max(self.episode_progress, 0.0))

        # Errore laterale normalizzato rispetto alla semilarghezza della pista.
        normalized_lateral_error = abs(float(state_info.lateral_error)) / max(
            state_info.track_half_width,
            1e-6,
        )

        # Calcola la reward dello step.
        reward_breakdown = compute_racing_reward(
            delta_progress=delta_progress,
            normalized_lateral_error=normalized_lateral_error,
            heading_error=state_info.heading_error,
            v_lateral=state_info.v_lateral,
            v_longitudinal=state_info.v_longitudinal,
            future_curvatures=state_info.future_curvatures,
            speed=self.car.v,
            max_speed=self.car.max_speed,
            steer=steer,
            throttle=throttle,
            previous_steer=previous_steer,
            previous_throttle=previous_throttle,
            car_length=self.car.car_length,
            step_count=self.step_count,
        )
        reward = float(reward_breakdown.total)

        # Conta per quanti step consecutivi la macchina rimane quasi ferma.
        if self.car.v < 0.5 and self.step_count > 20:
            self.stall_steps += 1
        else:
            self.stall_steps = 0

        # Condizioni di completamento e blocco della macchina.
        lap_completed = self.episode_progress >= 0.995 * self.track_length
        stalled = self.stall_steps >= 50

        # Gestisce le terminazioni dell'episodio e i relativi premi o penalità.
        terminated = False
        if state_info.is_off_track:
            reward += -500.0
            terminated = True
        elif lap_completed:
            reward += 1000.0
            terminated = True
        elif stalled:
            reward += -50.0
            terminated = True

        # Il limite massimo di step causa una troncatura dell'episodio.
        truncated = self.step_count >= self.max_steps
        lap_time = self.step_count * self.dt if lap_completed else None

        # Informazioni usate per logging, valutazione e callback.
        info = {
            "track_name": self.track_name,
            "track_path": self.track_path,
            "off_track": bool(state_info.is_off_track),
            "lap_completed": bool(lap_completed),
            "stalled": bool(stalled),
            "closest_idx": int(state_info.closest_idx),
            "progress": float(self.episode_progress),
            "progress_abs": float(state_info.current_progress),
            "delta_progress": float(delta_progress),
            "track_length": float(self.track_length),
            "progress_ratio": float(
                min(self.episode_progress / self.track_length, 1.0)
            ),
            "lap_time": lap_time,
            "speed_mps": float(self.car.v),
            "speed_kmh": float(self.car.v * 3.6),
            "v_longitudinal": float(state_info.v_longitudinal),
            "v_lateral": float(state_info.v_lateral),
            "normalized_lateral_error": float(normalized_lateral_error),
            "lateral_error": float(state_info.lateral_error),
            "heading_error": float(state_info.heading_error),
        }

        # Aggiunge al dizionario le singole componenti della reward.
        info.update(reward_breakdown.as_dict())

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """Rendering matplotlib per osservare una policy allenata."""
        if self.render_mode != "human" or self.car is None:
            return

        state_info = self._compute_state_info()

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 8))

            left_boundary_render = (
                self.centerline
                + self.normals_left * self.w_left_render[:, None]
            )
            right_boundary_render = (
                self.centerline
                - self.normals_left * self.w_right_render[:, None]
            )

            self.ax.plot(
                left_boundary_render[:, 0],
                left_boundary_render[:, 1],
                "r-",
                linewidth=1.5,
                label="Bordo sinistro",
            )
            self.ax.plot(
                right_boundary_render[:, 0],
                right_boundary_render[:, 1],
                "b-",
                linewidth=1.5,
                label="Bordo destro",
            )
            if self.raceline is None:
                raise RuntimeError(
                    "Il rendering di watch richiede una raceline valida, ma non è "
                    "stata caricata."
                )

            # La centerline NON viene disegnata: in watch usiamo la raceline.
            self.ax.plot(
                self.raceline[:, 0],
                self.raceline[:, 1],
                color="#FF7F0E",
                linestyle="-",
                linewidth=2.2,
                label="Raceline di riferimento",
            )

            (self.car_dot,) = self.ax.plot(
                [self.car.x],
                [self.car.y],
                "go",
                markersize=8,
                label="Posizione auto",
            )
            (self.car_heading_line,) = self.ax.plot(
                [],
                [],
                color="black",
                linestyle="-",
                linewidth=2.5,
                label="Orientamento auto",
            )
            self.info_text = self.ax.text(
                0.02,
                0.98,
                "",
                transform=self.ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", alpha=0.8),
            )

            self.ax.set_title(self.track_name)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.grid(True)
            self.ax.legend()

        self.car_dot.set_data([self.car.x], [self.car.y])
        x2 = self.car.x + 24.0 * np.cos(self.car.yaw)
        y2 = self.car.y + 24.0 * np.sin(self.car.yaw)
        self.car_heading_line.set_data([self.car.x, x2], [self.car.y, y2])

        self.info_text.set_text(
            f"Pista: {self.track_name}\n"
            f"Step: {self.step_count}\n"
            f"Velocità: {self.car.v * 3.6:.1f} km/h\n"
            f"Err laterale: {state_info.lateral_error:.2f} m\n"
            f"Progresso: "
            f"{100.0 * self.episode_progress / self.track_length:.1f}%"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None