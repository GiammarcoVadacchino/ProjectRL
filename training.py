"""

Training PPO da zero o ripreso da checkpoint su uno o più circuiti.

Questo modulo contiene esclusivamente la logica legata all'addestramento:
callback delle metriche, creazione dell'ambiente vettoriale, gestione dei
checkpoint, inizializzazione o caricamento di PPO e ciclo di training.
Le funzioni generiche di supporto e la valutazione sono definite in
``utils.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from f1_env import SimpleF1Env
from utils import (
    automatic_run_name,
    make_env,
    parse_architecture_specs,
    save_history,
    track_labels,
    write_metadata,
)

@dataclass
class TrainingResult:
    """Output essenziali prodotti da un singolo run di training."""

    model_path: Path
    model_name: str
    wall_time: float
    history: dict[str, list]
    num_envs: int


class RacingMetricsCallback(BaseCallback):
    """Registra le metriche di ogni episodio, separando i diversi circuiti."""

    def __init__(self, *, seed: int):
        super().__init__()
        self.seed = int(seed)
        self.history = {
            "episodes": [],
            "timesteps": [],
            "seed": [],
            "env_index": [],
            "track_name": [],
            "rewards": [],
            "progress": [],
            "lengths": [],
            "success": [],
            "lap_times": [],
            "off_track": [],
            "stalled": [],
        }
        self.ep_rewards: np.ndarray | None = None
        self.ep_steps: np.ndarray | None = None
        self.episode_count = 0

    def _on_training_start(self) -> None:
        num_envs = int(self.training_env.num_envs)
        self.ep_rewards = np.zeros(num_envs, dtype=np.float64)
        self.ep_steps = np.zeros(num_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        assert self.ep_rewards is not None and self.ep_steps is not None

        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=bool)
        infos = self.locals["infos"]

        self.ep_rewards += rewards
        self.ep_steps += 1

        for env_index, done in enumerate(dones):
            if not done:
                continue

            self.episode_count += 1
            info = infos[env_index]
            lap_time = info.get("lap_time")

            self.history["episodes"].append(self.episode_count)
            self.history["timesteps"].append(int(self.num_timesteps))
            self.history["seed"].append(self.seed)
            self.history["env_index"].append(int(env_index))
            self.history["track_name"].append(
                str(info.get("track_name", f"env_{env_index}"))
            )
            self.history["rewards"].append(float(self.ep_rewards[env_index]))
            self.history["progress"].append(
                min(float(info.get("progress_ratio", 0.0)), 1.0)
            )
            self.history["lengths"].append(int(self.ep_steps[env_index]))
            self.history["success"].append(
                1.0 if info.get("lap_completed", False) else 0.0
            )
            self.history["lap_times"].append(
                np.nan if lap_time is None else float(lap_time)
            )
            self.history["off_track"].append(
                1.0 if info.get("off_track", False) else 0.0
            )
            self.history["stalled"].append(
                1.0 if info.get("stalled", False) else 0.0
            )

            self.ep_rewards[env_index] = 0.0
            self.ep_steps[env_index] = 0

        return True


def _make_training_vec_env(
    args,
    *,
    random_start: bool,
) -> tuple[DummyVecEnv, list[str]]:
    """Crea un ambiente vettoriale con una istanza per ogni pista richiesta."""
    track_paths = [str(path) for path in args.tracks]
    if not track_paths:
        raise ValueError("Il comando train richiede almeno una pista.")

    labels = track_labels(track_paths)
    env_fns = []

    for rank, (track_path, track_name) in enumerate(zip(track_paths, labels)):
        def _init(
            path: str = track_path,
            name: str = track_name,
            env_rank: int = rank,
        ) -> SimpleF1Env:
            env = make_env(
                track_path=path,
                track_name=name,
                max_steps=int(args.max_steps),
                width_scale=float(args.width_scale),
                random_start=bool(random_start),
                render_mode=None,
            )
            # Ogni ambiente riceve un seed distinto ma riproducibile.
            env.reset(seed=int(args.seed) + env_rank)
            return env

        env_fns.append(_init)

    return DummyVecEnv(env_fns), labels


def _build_callbacks(
    *,
    output_dir: Path,
    model_name: str,
    checkpoint_freq_timesteps: int,
    num_envs: int,
    seed: int,
) -> tuple[RacingMetricsCallback, list[BaseCallback]]:
    metrics_callback = RacingMetricsCallback(seed=seed)
    callbacks: list[BaseCallback] = [metrics_callback]

    if checkpoint_freq_timesteps > 0:
        # CheckpointCallback usa n_calls; ogni call corrisponde a num_envs
        # timesteps globali. La divisione conserva la frequenza richiesta.
        save_freq = max(int(checkpoint_freq_timesteps) // max(num_envs, 1), 1)
        checkpoint_dir = output_dir / "checkpoints" / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=model_name,
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
        print(
            f"Checkpoint ogni circa {checkpoint_freq_timesteps} timesteps in: "
            f"{checkpoint_dir}"
        )

    return metrics_callback, callbacks


def _set_constant_learning_rate(model: PPO, learning_rate: float) -> None:
    """Imposta una schedule costante e aggiorna subito l'optimizer caricato."""
    lr = float(learning_rate)
    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = lr


def _current_learning_rate(model: PPO) -> float:
    optimizer_groups = model.policy.optimizer.param_groups
    if optimizer_groups:
        return float(optimizer_groups[0]["lr"])

    learning_rate = model.learning_rate
    if callable(learning_rate):
        return float(learning_rate(1.0))
    return float(learning_rate)


def _resolve_random_start(args) -> bool:
    """Mantiene i precedenti default: True da zero, False da checkpoint."""
    if args.random_start is not None:
        return bool(args.random_start)
    return args.model_path is None


def _train_once(
    args,
    *,
    architecture: list[int] | None,
    run_index: int,
    run_count: int,
) -> TrainingResult:
    # Converte il percorso della cartella di output in un oggetto Path.
    output_dir = Path(args.output_dir)

    # Crea la cartella di output e tutte le eventuali cartelle intermedie.
    # Se la cartella esiste già, non viene generato alcun errore.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determina se gli episodi devono iniziare da una posizione casuale.
    # Per impostazione predefinita:
    # - nel training da zero viene usato il random start;
    # -  
    random_start = _resolve_random_start(args)

    # Crea l'ambiente vettoriale di training.
    # Viene costruita un'istanza indipendente dell'ambiente per ogni circuito.
    # La funzione restituisce anche i nomi leggibili dei circuiti.
    train_env, track_labels = _make_training_vec_env(
        args,
        random_start=random_start,
    )

    # Numero di ambienti eseguiti in parallelo.
    # Corrisponde normalmente al numero di circuiti forniti al comando.
    num_envs = int(train_env.num_envs)

    # Genera automaticamente un nome univoco e descrittivo per il run.
    # Il nome può includere architettura, circuiti, seed, learning rate
    # e modalità di partenza.
    model_name = automatic_run_name(
        args,
        track_labels=track_labels,
        random_start=random_start,
        architecture=architecture,
    )

    # Percorso completo in cui verrà salvato il modello PPO finale.
    model_path = output_dir / f"{model_name}.zip"

    # Il training è considerato ripreso se è stato fornito il percorso
    # di un modello PPO già esistente.
    resumed = args.model_path is not None

    # Memorizza il percorso del modello iniziale in caso di training ripreso.
    # Nel training da zero il valore rimane None.
    source_model_path = Path(args.model_path) if resumed else None

    # Stampa un'intestazione che separa visivamente questo run dagli altri.
    print("\n" + "#" * 76)
    print(f"RUN {run_index}/{run_count}: {model_name}")
    print("#" * 76)

    # Indica se PPO viene inizializzato da zero oppure caricato da checkpoint.
    print("TRAINING PPO RIPRESO DA CHECKPOINT" if resumed else "TRAINING PPO DA ZERO")

    # Mostra la corrispondenza tra ogni ambiente, il nome del circuito
    # e il relativo file contenente la pista.
    for i, (label, path) in enumerate(zip(track_labels, args.tracks), start=1):
        print(f"Env {i}: {label} -> {path}")

    # Stampa i principali parametri del run.
    print(f"Numero ambienti: {num_envs}")
    print(f"Timesteps richiesti: {int(args.timesteps)}")

    # Stable-Baselines3 conta i timesteps globalmente su tutti gli ambienti.
    # Questa quantità rappresenta quindi una stima dei timesteps assegnati
    # a ciascun circuito.
    print(
        "Timesteps attesi per circuito: "
        f"circa {int(args.timesteps) / num_envs:,.0f}"
    )

    print(f"Random start: {random_start}")
    print(f"Seed: {int(args.seed)}")
    print(f"Checkpoint freq: {int(args.checkpoint_freq)}")

    # Numero di timesteps già presenti nel modello prima dell'inizio del run.
    # Rimane zero quando il training parte da pesi inizializzati casualmente.
    start_num_timesteps = 0

    # Il blocco try/finally garantisce la chiusura dell'ambiente anche in caso
    # di errore o interruzione durante il training.
    try:
        # Gestione del caso in cui il training venga ripreso da un modello.
        if resumed:
            # In questo ramo source_model_path deve necessariamente esistere.
            assert source_model_path is not None

            print(f"Modello iniziale: {source_model_path}")

            # Argomenti utilizzati per caricare il modello PPO.
            # Il nuovo ambiente vettoriale sostituisce quello eventualmente
            # associato al modello originale.
            load_kwargs: dict[str, Any] = {
                "env": train_env,
                "seed": int(args.seed),
            }

            # Se viene specificato un nuovo learning rate, custom_objects
            # sostituisce il valore memorizzato nel modello salvato durante
            # il caricamento.
            if args.learning_rate is not None:
                load_kwargs["custom_objects"] = {
                    "learning_rate": float(args.learning_rate)
                }

            # Carica il modello PPO dal checkpoint e lo collega ai nuovi ambienti.
            model = PPO.load(source_model_path, **load_kwargs)

            # Reimposta il seed del modello per rendere il nuovo run
            # il più possibile riproducibile.
            model.set_random_seed(int(args.seed))

            # Salva il numero di timesteps accumulati dal modello prima
            # della continuazione del training.
            start_num_timesteps = int(model.num_timesteps)

            # Se richiesto, imposta una schedule costante per il nuovo
            # learning rate e aggiorna immediatamente l'optimizer.
            if args.learning_rate is not None:
                _set_constant_learning_rate(model, float(args.learning_rate))

            # Stampa le informazioni effettive del modello caricato.
            print(f"Timesteps già presenti nel modello: {start_num_timesteps}")
            print(f"Learning rate effettivo: {_current_learning_rate(model)}")

            # Durante la continuazione l'architettura non viene ricostruita
            # dagli argomenti, ma viene mantenuta quella salvata nel modello.
            print("Architettura: caricata dal modello")

        # Gestione del training PPO completamente da zero.
        else:
            # Nel training da zero è obbligatorio specificare l'architettura
            # delle reti neurali.
            if architecture is None:
                raise ValueError("Architettura mancante nel training da zero")

            # Utilizza 3e-4 come learning rate predefinito, a meno che
            # l'utente non ne abbia specificato uno differente.
            learning_rate = (
                3e-4 if args.learning_rate is None else float(args.learning_rate)
            )

            # Il rollout buffer contiene n_steps transizioni per ciascun ambiente.
            # La sua dimensione globale è quindi n_steps * num_envs.
            rollout_size = int(args.n_steps) * num_envs

            # La batch size non può essere maggiore del numero di campioni
            # disponibili nel rollout buffer.
            if rollout_size < int(args.batch_size):
                raise ValueError(
                    "Il rollout buffer n_steps * num_envs deve essere almeno "
                    "grande quanto --batch-size"
                )

            # Usa la stessa architettura di livelli nascosti sia per:
            # - pi: rete della policy, che produce le azioni;
            # - vf: rete della value function, che stima il valore degli stati.
            policy_kwargs = dict(
                net_arch=dict(pi=architecture, vf=architecture)
            )

            # Inizializza un nuovo modello PPO con policy MLP.
            model = PPO(
                "MlpPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                n_steps=int(args.n_steps),
                batch_size=int(args.batch_size),

                # Parametro lambda della Generalized Advantage Estimation.
                # Controlla il compromesso tra bias e varianza della stima
                # dell'advantage.
                gae_lambda=0.95,

                # Fattore di sconto delle ricompense future.
                gamma=0.995,

                # Coefficiente del bonus di entropia.
                # Valori positivi favoriscono una maggiore esplorazione.
                ent_coef=float(args.ent_coef),

                # Mostra nel terminale le statistiche prodotte da PPO.
                verbose=1,

                # Seed usato per l'inizializzazione e il training.
                seed=int(args.seed),
            )

            print(f"Architettura policy e value network: {architecture}")
            print(f"Learning rate: {learning_rate}")
            print("Pesi PPO: inizializzati da zero")

        # Costruisce:
        # - la callback che registra le metriche di ogni episodio;
        # - l'eventuale callback per il salvataggio periodico dei checkpoint.
        metrics_callback, callbacks = _build_callbacks(
            output_dir=output_dir,
            model_name=model_name,
            checkpoint_freq_timesteps=int(args.checkpoint_freq),
            num_envs=num_envs,
            seed=int(args.seed),
        )

        # Registra l'istante di inizio per misurare la durata reale
        # dell'addestramento.
        start_time = time.time()

        # Avvia l'ottimizzazione PPO.
        model.learn(
            # Numero di nuovi timesteps richiesti per questo run.
            total_timesteps=int(args.timesteps),

            # Lista delle callback per metriche e checkpoint.
            callback=callbacks,

            # Nel training da zero il contatore dei timesteps viene azzerato.
            # Nella continuazione da checkpoint viene invece mantenuto il
            # conteggio già presente nel modello.
            reset_num_timesteps=not resumed,
        )

        # Calcola il tempo totale impiegato dal training.
        wall_time = time.time() - start_time

        # Numero complessivo di timesteps presenti nel modello dopo il training.
        final_num_timesteps = int(model.num_timesteps)

        # Salva il modello PPO finale nel percorso costruito precedentemente.
        model.save(str(model_path))

    finally:
        # Chiude sempre gli ambienti, liberando le risorse utilizzate,
        # anche se il training termina con un'eccezione.
        train_env.close()

    # Salva in formato CSV le metriche episodiche raccolte dalla callback.
    history_path = save_history(
        metrics_callback.history,
        output_dir,
        model_name,
    )

    # Estrae, mantenendo l'ordine originale, i nomi dei circuiti per cui
    # almeno un episodio è terminato ed è stato registrato nella history.
    observed_tracks = list(
        dict.fromkeys(
            str(name) for name in metrics_callback.history["track_name"]
        )
    )

    # Individua gli ambienti che non hanno ancora prodotto un episodio completo
    # o terminato durante il numero di timesteps richiesto.
    missing_tracks = [name for name in track_labels if name not in observed_tracks]

    print(f"Circuiti presenti nella history: {observed_tracks}")

    # Se un circuito non compare nella history, viene mostrato un avviso.
    # Questo non significa necessariamente che non sia stato utilizzato:
    # potrebbe semplicemente non aver completato o terminato alcun episodio.
    if missing_tracks:
        print(
            "ATTENZIONE: nessun episodio terminato per questi circuiti, quindi "
            f"non compaiono ancora nel CSV episodico: {missing_tracks}. "
            "Aumenta --timesteps o riduci --max-steps per una prova breve."
        )

    # Costruisce il dizionario contenente tutte le informazioni necessarie
    # per documentare e riprodurre il run.
    metadata = {
        # Indica se il modello è stato inizializzato da zero o da checkpoint.
        "training_mode": "resumed" if resumed else "from_scratch",

        # Percorso del modello iniziale, oppure None nel training da zero.
        "source_model": str(source_model_path) if source_model_path else None,

        # Elenco dei circuiti con nome leggibile e percorso del file.
        "tracks": [
            {"name": name, "path": str(path)}
            for name, path in zip(track_labels, args.tracks)
        ],

        # Numero di ambienti usati contemporaneamente.
        "num_envs": num_envs,

        # Timesteps globali richiesti per il run.
        "requested_timesteps": int(args.timesteps),

        # Stima dei timesteps assegnati a ogni circuito.
        "approx_requested_timesteps_per_track": int(args.timesteps) / num_envs,

        # Contatore PPO prima e dopo il training.
        "start_num_timesteps": int(start_num_timesteps),
        "final_num_timesteps": int(final_num_timesteps),

        # Architettura richiesta nel training da zero.
        # Nel training ripreso può essere None perché viene caricata dal modello.
        "architecture": list(architecture) if architecture is not None else None,

        # Iperparametri effettivamente presenti nel modello.
        "learning_rate": float(_current_learning_rate(model)),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "ent_coef": float(model.ent_coef),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),

        # Configurazione dell'ambiente e del run.
        "random_start": bool(random_start),
        "checkpoint_frequency": int(args.checkpoint_freq),
        "seed": int(args.seed),

        # Tempo reale impiegato per il training.
        "wall_time_seconds": float(wall_time),

        # Percorso del CSV contenente le metriche episodiche.
        "training_history_csv": str(history_path),

        # Circuiti per cui è presente almeno un episodio nella history.
        "tracks_observed_in_episode_history": observed_tracks,
    }

    # Salva i metadati del run in un file separato.
    write_metadata(
        output_dir=output_dir,
        model_name=model_name,
        metadata=metadata,
    )

    # Mostra il riepilogo finale del run.
    print(f"\nTraining completato in {wall_time:.2f} s")
    print(f"Modello salvato in: {model_path}")

    # Restituisce le informazioni essenziali prodotte dal training.
    return TrainingResult(
        model_path=model_path,
        model_name=model_name,
        wall_time=float(wall_time),
        history=metrics_callback.history,
        num_envs=num_envs,
    )


def train_agent(args) -> list[TrainingResult]:
    """Allena da zero oppure continua un singolo modello caricato."""
    if args.model_path is not None:
        architectures: list[list[int] | None] = [None]
    else:
        architectures = parse_architecture_specs(list(args.archs))

    if args.run_name and len(architectures) > 1:
        raise ValueError(
            "--run-name può essere usato solo con una singola architettura, "
            "per evitare sovrascritture"
        )

    results: list[TrainingResult] = []
    print("\n" + "=" * 76)
    print("CONFIGURAZIONE TRAINING")
    print(f"Circuiti: {list(args.tracks)}")
    print(f"Timesteps per run: {int(args.timesteps)}")
    print(f"Run richiesti: {len(architectures)}")
    print("=" * 76)

    for index, architecture in enumerate(architectures, start=1):
        result = _train_once(
            args,
            architecture=architecture,
            run_index=index,
            run_count=len(architectures),
        )
        results.append(result)

    print("\n" + "=" * 76)
    print("TUTTI I TRAINING RICHIESTI SONO TERMINATI")
    for result in results:
        print(f"- {result.model_name}: {result.model_path}")
    print("=" * 76)
    return results
