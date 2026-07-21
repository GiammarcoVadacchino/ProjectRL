

from __future__ import annotations

import argparse
from pathlib import Path


def add_common_environment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Numero massimo di step per episodio",
    )
    parser.add_argument(
        "--width-scale",
        type=float,
        default=6.0,
        help=(
            "Scala solo visuale dei bordi nel rendering; non modifica "
            "fisica, reward o osservazioni"
        ),
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracks",
        type=str,
        nargs="+",
        required=True,
        metavar="TRACK",
        help=(
            "Uno o più file CSV usati contemporaneamente nel training. "
            "Un percorso crea un singolo ambiente; tre percorsi creano i tre "
            "ambienti interleaved usati negli esperimenti multi-track."
        ),
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        metavar="ARCH",
        help=(
            "Architetture da allenare da zero, con layer separati da virgole. "
            "Esempio: --archs 64,64,32,16 128,128,64,32. "
            "Obbligatorio senza --model-path e non ammesso quando si carica "
            "un modello esistente."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Modello PPO da caricare per continuare il training. Se omesso, "
            "il training parte da zero usando --archs."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=15_000_000,
        help=(
            "Timesteps globali per ciascuna architettura, oppure timesteps "
            "aggiuntivi quando viene usato --model-path"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate costante. Se omesso: 3e-4 nel training da zero; "
            "nel training ripreso viene mantenuto il learning rate del modello."
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Numero di step PPO per ambiente; usato solo nel training da zero",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size PPO; usata solo nel training da zero",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Coefficiente di entropia PPO; usato solo nel training da zero",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Frequenza checkpoint in timesteps globali; 0 per disattivare",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs_training",
        help="Directory in cui salvare modello, CSV, metadata e checkpoint",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nome opzionale del run; se omesso viene generato automaticamente",
    )
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Abilita o disabilita la partenza casuale. Se non specificato, "
            "il default è True da zero e False quando si usa --model-path."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO per autonomous racing: training e valutazione"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser(
        "train",
        help="Allena da zero o continua un modello PPO esistente",
    )
    add_training_args(parser_train)
    add_common_environment_args(parser_train)

    parser_watch = subparsers.add_parser(
        "watch",
        help="Guarda deterministicamente un modello allenato",
    )
    parser_watch.add_argument("--model-path", type=str, required=True)
    parser_watch.add_argument("--track", type=str, required=True)
    parser_watch.add_argument("--episodes", type=int, default=1)
    parser_watch.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed associato al modello. Se omesso, viene cercato nel nome "
            "o nel percorso del modello, ad esempio SEED44 o seed_44."
        ),
    )
    fine_tune_group = parser_watch.add_mutually_exclusive_group()
    fine_tune_group.add_argument(
        "--pre-fine-tune",
        dest="pre_fine_tune",
        action="store_true",
        help="Salva Pre fine tune=True nel CSV",
    )
    fine_tune_group.add_argument(
        "--post-fine-tune",
        dest="pre_fine_tune",
        action="store_false",
        help="Salva Pre fine tune=False nel CSV",
    )
    parser_watch.set_defaults(pre_fine_tune=None)
    parser_watch.add_argument(
        "--render-sleep",
        type=float,
        default=1.0 / 30.0,
    )
    add_common_environment_args(parser_watch)

    return parser


def _validate_file(path: str, flag_name: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{flag_name}: file non trovato: {path}")


def _validate_args(args) -> None:
    if hasattr(args, "timesteps") and int(args.timesteps) <= 0:
        raise ValueError("--timesteps deve essere positivo")
    if hasattr(args, "max_steps") and int(args.max_steps) <= 0:
        raise ValueError("--max-steps deve essere positivo")
    if hasattr(args, "learning_rate"):
        learning_rate = getattr(args, "learning_rate", None)
        if learning_rate is not None and float(learning_rate) <= 0.0:
            raise ValueError("--learning-rate deve essere positivo")
    if hasattr(args, "batch_size") and int(args.batch_size) <= 0:
        raise ValueError("--batch-size deve essere positivo")
    if hasattr(args, "n_steps") and int(args.n_steps) <= 0:
        raise ValueError("--n-steps deve essere positivo")
    if hasattr(args, "checkpoint_freq") and int(args.checkpoint_freq) < 0:
        raise ValueError("--checkpoint-freq non può essere negativo")
    if hasattr(args, "episodes") and int(args.episodes) <= 0:
        raise ValueError("--episodes deve essere positivo")
    if hasattr(args, "render_sleep") and float(args.render_sleep) < 0.0:
        raise ValueError("--render-sleep non può essere negativo")
    if hasattr(args, "seed"):
        seed = getattr(args, "seed", None)
        if seed is not None and int(seed) < 0:
            raise ValueError("--seed non può essere negativo")

    if args.command == "train":
        if not args.tracks:
            raise ValueError("--tracks richiede almeno un percorso")
        for path in args.tracks:
            _validate_file(path, "--tracks")

        if args.model_path is None:
            if not args.archs:
                raise ValueError(
                    "Nel training da zero devi specificare almeno una "
                    "architettura con --archs"
                )
        else:
            _validate_file(args.model_path, "--model-path")
            if args.archs:
                raise ValueError(
                    "Quando usi --model-path l'architettura viene caricata dal "
                    "modello: rimuovi --archs"
                )

    if args.command == "watch":
        _validate_file(args.track, "--track")
        _validate_file(args.model_path, "--model-path")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)

    # Import ritardato: i comandi --help restano utilizzabili anche prima di
    # attivare l'ambiente virtuale con Stable-Baselines3.
    from training import train_agent
    from utils import watch_agent

    if args.command == "train":
        train_agent(args)
    elif args.command == "watch":
        watch_agent(args)
    else:
        raise ValueError(f"Comando non riconosciuto: {args.command}")


if __name__ == "__main__":
    main()
