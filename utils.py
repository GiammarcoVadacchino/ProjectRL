"""
utils.py
========

Funzioni di supporto condivise dal progetto.

Questo modulo contiene le operazioni che non implementano direttamente il
training PPO: normalizzazione dei nomi, parsing delle architetture, creazione
dell'ambiente, salvataggio di CSV e metadata, valutazione deterministica e
supporto al comando ``watch``.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from stable_baselines3 import PPO

from f1_env import SimpleF1Env


WATCH_USES_EXTERNAL_RACELINE = True

WATCH_RESULT_FIELDS = [
    "Nome Modello",
    "seed",
    "Pre fine tune",
    "track_name",
    "track_length_km",
    "episodes",
    "completion_rate",
    "off_track_rate",
    "stalled_rate",
    "best_lap_time",
    "best_time_per_km",
    "mean_progress_ratio",
    "mean_reward",
    "mean_speed_kmh",
    "mean_abs_steering_delta",
]


@dataclass
class EvalResult:
    """Risultati aggregati della visualizzazione/evaluation deterministica."""

    track_name: str
    track_length_km: float
    episodes: int
    completion_rate: float
    off_track_rate: float
    stalled_rate: float
    best_lap_time: float | None
    mean_lap_time: float | None
    mean_progress_ratio: float
    mean_reward: float
    mean_speed_kmh: float
    mean_abs_steering_delta: float

    def as_dict(self) -> dict[str, Any]:
        """Restituisce soltanto le metriche previste da watch_results.csv."""
        return {
            "track_name": self.track_name,
            "track_length_km": self.track_length_km,
            "episodes": self.episodes,
            "completion_rate": self.completion_rate,
            "off_track_rate": self.off_track_rate,
            "stalled_rate": self.stalled_rate,
            "best_lap_time": self.best_lap_time,
            "best_time_per_km": (
                self.best_lap_time / self.track_length_km
                if self.best_lap_time is not None and self.track_length_km > 0.0
                else None
            ),
            "mean_progress_ratio": self.mean_progress_ratio,
            "mean_reward": self.mean_reward,
            "mean_speed_kmh": self.mean_speed_kmh,
            "mean_abs_steering_delta": self.mean_abs_steering_delta,
        }


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "run"


def architecture_name(net_arch: Iterable[int]) -> str:
    return "_".join(str(int(x)) for x in net_arch)


def track_labels(track_paths: list[str]) -> list[str]:
    """Crea etichette leggibili e univoche per le piste."""
    stems = [safe_name(Path(path).stem) for path in track_paths]
    counts: dict[str, int] = {}
    labels: list[str] = []
    for stem in stems:
        counts[stem] = counts.get(stem, 0) + 1
        suffix = "" if counts[stem] == 1 else f"_{counts[stem]}"
        labels.append(f"{stem}{suffix}")
    return labels


def track_set_token(track_labels: list[str]) -> str:
    if len(track_labels) == 1:
        return track_labels[0]
    return "multitrack_" + "_".join(track_labels)


def make_env(
    *,
    track_path: str,
    track_name: str,
    max_steps: int,
    width_scale: float,
    random_start: bool,
    render_mode: str | None = None,
    raceline_path: str | None = None,
) -> SimpleF1Env:
    return SimpleF1Env(
        track_path=track_path,
        track_name=track_name,
        render_mode=render_mode,
        max_steps=max_steps,
        width_scale=width_scale,
        random_start=random_start,
        raceline_path=raceline_path,
    )


def save_history(
    history: dict[str, list],
    output_dir: Path,
    model_name: str,
) -> Path:
    """Salva tutte le metriche episodiche del training senza aggregarle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_name}_training_history.csv"
    keys = list(history.keys())

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for i in range(len(history.get("episodes", []))):
            writer.writerow({key: history[key][i] for key in keys})

    print(f"Metriche training salvate in: {csv_path}")
    return csv_path


def parse_architecture_specs(specs: list[str]) -> list[list[int]]:
    parsed: list[list[int]] = []
    for raw_spec in specs:
        parts = [
            part.strip()
            for part in str(raw_spec).replace("x", ",").split(",")
            if part.strip()
        ]
        if not parts:
            raise ValueError(f"Architettura non valida: {raw_spec}")
        try:
            architecture = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Architettura non valida: {raw_spec}") from exc
        if any(value <= 0 for value in architecture):
            raise ValueError(f"Architettura non valida: {raw_spec}")
        parsed.append(architecture)
    return parsed


def automatic_run_name(
    args,
    *,
    track_labels: list[str],
    random_start: bool,
    architecture: list[int] | None,
) -> str:
    if args.run_name:
        return safe_name(args.run_name)

    track_token = track_set_token(track_labels)
    start_token = "random_start" if random_start else "fixed_start"

    if args.model_path is None:
        if architecture is None:
            raise ValueError("Architettura mancante nel training da zero")
        arch_token = architecture_name(architecture)
        return safe_name(
            f"ppo_f1_{arch_token}_{track_token}_{start_token}_seed_{args.seed}"
        )

    source_stem = Path(args.model_path).stem
    lr_token = (
        safe_name(f"{float(args.learning_rate):.0e}")
        if args.learning_rate is not None
        else "saved_lr"
    )
    return safe_name(
        f"{source_stem}_continued_{track_token}_{start_token}_"
        f"lr_{lr_token}_seed_{args.seed}"
    )


def write_metadata(
    *,
    output_dir: Path,
    model_name: str,
    metadata: dict[str, Any],
) -> Path:
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Metadata training salvati in: {metadata_path}")
    return metadata_path


def evaluate_policy(
    env: SimpleF1Env,
    *,
    model: PPO,
    episodes: int,
    render: bool,
    sleep: float,
) -> EvalResult:
    """Valuta sempre deterministicamente, senza aggiornare i pesi."""
    episode_rewards: list[float] = []
    progress_ratios: list[float] = []
    lap_times: list[float] = []
    speeds: list[float] = []
    steering_deltas: list[float] = []
    completed = 0
    off_track = 0
    stalled = 0

    for episode in range(int(episodes)):
        obs, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0.0
        previous_steer = 0.0
        info: dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            speeds.append(float(info.get("speed_kmh", 0.0)))
            steering_deltas.append(abs(float(action[1]) - previous_steer))
            previous_steer = float(action[1])

            if render:
                env.render()
                if sleep > 0.0:
                    time.sleep(sleep)

        completed += int(bool(info.get("lap_completed", False)))
        off_track += int(bool(info.get("off_track", False)))
        stalled += int(bool(info.get("stalled", False)))
        episode_rewards.append(episode_reward)
        progress_ratios.append(float(info.get("progress_ratio", 0.0)))
        if info.get("lap_time") is not None:
            lap_times.append(float(info["lap_time"]))

        print(
            f"Ep {episode + 1:03d}/{episodes} | "
            f"track={env.track_name} | "
            f"completed={info.get('lap_completed', False)} | "
            f"off_track={info.get('off_track', False)} | "
            f"progress={100.0 * float(info.get('progress_ratio', 0.0)):.1f}% | "
            f"lap_time={info.get('lap_time')}"
        )

    return EvalResult(
        track_name=env.track_name,
        track_length_km=float(env.track_length / 1000.0),
        episodes=int(episodes),
        completion_rate=float(completed / max(int(episodes), 1)),
        off_track_rate=float(off_track / max(int(episodes), 1)),
        stalled_rate=float(stalled / max(int(episodes), 1)),
        best_lap_time=float(np.min(lap_times)) if lap_times else None,
        mean_lap_time=float(np.mean(lap_times)) if lap_times else None,
        mean_progress_ratio=(
            float(np.mean(progress_ratios)) if progress_ratios else 0.0
        ),
        mean_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        mean_speed_kmh=float(np.mean(speeds)) if speeds else 0.0,
        mean_abs_steering_delta=(
            float(np.mean(steering_deltas)) if steering_deltas else 0.0
        ),
    )


def save_watch_result_csv(result_data: dict[str, Any]) -> Path:
    """
    Aggiunge il risultato a ``watch_results.csv`` usando uno schema fisso.

    Se il file esiste con uno schema precedente, viene aggiornato direttamente
    nello stesso percorso conservando i campi compatibili. Non vengono creati
    file di backup o altri output permanenti.
    """
    csv_path = Path("watch_results.csv")

    missing_fields = [
        field for field in WATCH_RESULT_FIELDS if field not in result_data
    ]
    extra_fields = [
        field for field in result_data if field not in WATCH_RESULT_FIELDS
    ]
    if missing_fields or extra_fields:
        raise ValueError(
            "Schema risultato watch non valido. "
            f"Campi mancanti: {missing_fields}; campi inattesi: {extra_fields}"
        )

    ordered_result = {
        field: result_data[field] for field in WATCH_RESULT_FIELDS
    }

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_fields != WATCH_RESULT_FIELDS:
            normalized_rows = [
                {field: row.get(field, "") for field in WATCH_RESULT_FIELDS}
                for row in existing_rows
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=WATCH_RESULT_FIELDS,
                )
                writer.writeheader()
                writer.writerows(normalized_rows)
            print(
                "Schema di watch_results.csv aggiornato direttamente "
                "senza creare file aggiuntivi."
            )

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=WATCH_RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(ordered_result)

    return csv_path


def resolve_watch_seed(explicit_seed: int | None, model_path: Path) -> int:
    """Usa il seed esplicito oppure lo ricava dal percorso del modello."""
    if explicit_seed is not None:
        return int(explicit_seed)

    patterns = (
        re.compile(r"seed[_-]?(\d+)", re.IGNORECASE),
        re.compile(r"^(\d+)$"),
    )
    candidates = [model_path.stem, *(part.name for part in model_path.parents)]

    for candidate in candidates:
        for pattern in patterns:
            match = pattern.search(str(candidate))
            if match:
                return int(match.group(1))

    raise ValueError(
        "Seed non determinabile dal percorso del modello. "
        "Specificalo nel comando watch con --seed."
    )


def resolve_pre_fine_tune(
    explicit_value: bool | None,
    model_path: Path,
) -> bool:
    """Usa il valore esplicito o lo deduce dal nome/percorso del modello."""
    if explicit_value is not None:
        return bool(explicit_value)

    normalized_path = str(model_path).lower()
    post_fine_tune_tokens = (
        "_continued_",
        "finetune",
        "fine_tune",
        "fine-tune",
        "post_fine",
        "post-fine",
    )
    return not any(
        token in normalized_path for token in post_fine_tune_tokens
    )


def resolve_raceline_path(track_path: str) -> Path:
    """Trova il CSV della raceline usata esclusivamente dal rendering watch."""
    track_file = Path(track_path)
    filename = track_file.name

    candidates = [
        track_file.parent.parent / "racelines" / filename,
        track_file.parent / "racelines" / filename,
        Path("racetrack-database") / "racelines" / filename,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Raceline non trovata per il circuito '{track_file.stem}'. "
        f"Percorsi controllati:\n  - {searched}"
    )


def watch_agent(args) -> EvalResult:
    """Mostra un modello su una pista e salva il risultato deterministico."""
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    track_name = safe_name(Path(args.track).stem)
    raceline_path = resolve_raceline_path(str(args.track))
    print(f"Raceline caricata solo per il rendering watch: {raceline_path}")

    env = make_env(
        track_path=str(args.track),
        track_name=track_name,
        max_steps=int(args.max_steps),
        width_scale=float(args.width_scale),
        random_start=False,
        render_mode="human",
        raceline_path=str(raceline_path),
    )
    model = PPO.load(model_path, env=env)

    result = evaluate_policy(
        env,
        model=model,
        episodes=int(args.episodes),
        render=True,
        sleep=float(args.render_sleep),
    )
    env.close()

    result_data = {
        "Nome Modello": model_path.stem,
        "seed": resolve_watch_seed(
            getattr(args, "seed", None),
            model_path,
        ),
        "Pre fine tune": resolve_pre_fine_tune(
            getattr(args, "pre_fine_tune", None),
            model_path,
        ),
        **result.as_dict(),
    }
    csv_path = save_watch_result_csv(result_data)

    print("\nRisultati watch deterministica:")
    print(json.dumps(result_data, indent=2))
    print(f"Risultati watch salvati in: {csv_path.resolve()}")
    return result
