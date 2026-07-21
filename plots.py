#!/usr/bin/env python3
"""
Crea tre grafici aggregando tre CSV, uno per ciascun training seed:

1. Mean Lap Progress over Training
2. Completion Rate over Training
3. Mean Episode Reward over Training

Per ogni circuito:
- la linea continua rappresenta la media tra i tre seed;
- la banda trasparente rappresenta media ± deviazione standard.

Struttura attesa dei CSV:
    timesteps, track_name, rewards, progress, success
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timesteps",
    "track_name",
    "rewards",
    "progress",
    "success",
}

# Gli stessi colori vengono mantenuti in tutti e tre i grafici.
TRACK_COLORS = {
    "Monza": "tab:blue",
    "Melbourne": "tab:orange",
    "Silverstone": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera i grafici di progress, completion rate e reward "
            "a partire da tre CSV ottenuti con seed diversi."
        )
    )

    parser.add_argument("seed_0", type=Path, help="CSV del primo seed")
    parser.add_argument("seed_1", type=Path, help="CSV del secondo seed")
    parser.add_argument("seed_2", type=Path, help="CSV del terzo seed")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Cartella in cui salvare i grafici. Default: plots",
    )

    parser.add_argument(
        "--prefix",
        default="ppo_multiseed",
        help="Prefisso dei file generati. Default: ppo_multiseed",
    )

    parser.add_argument(
        "--bin-size",
        type=int,
        default=200_000,
        help=(
            "Ampiezza degli intervalli di global training steps "
            "usati per aggregare gli episodi. Default: 200000"
        ),
    )

    parser.add_argument(
        "--smooth-bins",
        type=int,
        default=3,
        help=(
            "Finestra della media mobile applicata dopo il binning. "
            "Usa 1 per disattivarla. Default: 3"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Risoluzione dei PNG. Default: 300",
    )

    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """Carica un CSV e verifica che abbia le colonne necessarie."""

    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Il file {path} non contiene le colonne richieste: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    for column in ["timesteps", "rewards", "progress", "success"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["track_name"] = df["track_name"].astype(str)

    df = df.dropna(subset=["timesteps", "track_name"])
    df = df.sort_values("timesteps").reset_index(drop=True)

    return df


def common_tracks(histories: list[pd.DataFrame]) -> list[str]:
    """Restituisce i circuiti presenti in tutti e tre i CSV."""

    tracks = set(histories[0]["track_name"].unique())

    for df in histories[1:]:
        tracks &= set(df["track_name"].unique())

    if not tracks:
        raise ValueError("I tre CSV non hanno circuiti in comune.")

    preferred_order = ["Monza", "Melbourne", "Silverstone"]

    ordered = [
        track
        for track in preferred_order
        if track in tracks
    ]

    ordered.extend(sorted(tracks.difference(ordered)))

    return ordered


def metric_to_plot_units(
    values: pd.Series,
    metric_column: str,
) -> pd.Series:
    """
    Converte progress e success in percentuale se sono salvati in [0, 1].
    La reward viene lasciata nella sua unità originale.
    """

    values = pd.to_numeric(values, errors="coerce").astype(float)

    if metric_column in {"progress", "success"}:
        finite = values[np.isfinite(values)]

        if not finite.empty and finite.abs().max() <= 1.5:
            values = values * 100.0

    return values


def build_seed_curve(
    df: pd.DataFrame,
    track: str,
    metric_column: str,
    bin_edges: np.ndarray,
    smooth_bins: int,
) -> np.ndarray:
    """
    Costruisce la curva di un singolo seed per un circuito.

    Gli episodi vengono raggruppati in intervalli di global training steps.
    Per ogni intervallo viene calcolata la media della metrica.
    """

    current = df.loc[
        df["track_name"] == track,
        ["timesteps", metric_column],
    ].copy()

    current[metric_column] = metric_to_plot_units(
        current[metric_column],
        metric_column,
    )

    current = current.dropna(subset=[metric_column])

    current["step_bin"] = pd.cut(
        current["timesteps"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=False,
    )

    binned = current.groupby(
        "step_bin",
        observed=True,
    )[metric_column].mean()

    full_index = pd.RangeIndex(
        len(bin_edges) - 1,
        name="step_bin",
    )

    curve = binned.reindex(full_index).astype(float)

    # Interpola solamente eventuali buchi interni.
    curve = curve.interpolate(
        method="linear",
        limit_area="inside",
    )

    if smooth_bins > 1:
        curve = curve.rolling(
            window=smooth_bins,
            center=True,
            min_periods=1,
        ).mean()

    return curve.to_numpy(dtype=float)


def aggregate_metric(
    histories: list[pd.DataFrame],
    tracks: list[str],
    metric_column: str,
    bin_size: int,
    smooth_bins: int,
) -> tuple[
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """
    Calcola, per ogni circuito, media e deviazione standard tra i tre seed.
    """

    common_max_step = int(
        min(df["timesteps"].max() for df in histories)
    )

    if common_max_step < bin_size:
        raise ValueError(
            "Il tratto di training comune ai tre seed "
            "è più corto di un singolo bin."
        )

    n_bins = int(np.ceil(common_max_step / bin_size))

    bin_edges = np.arange(
        0,
        (n_bins + 1) * bin_size + 1,
        bin_size,
        dtype=float,
    )

    step_centers = (
        bin_edges[:-1] + bin_edges[1:]
    ) / 2.0

    aggregated = {}

    for track in tracks:
        seed_curves = np.vstack([
            build_seed_curve(
                df=df,
                track=track,
                metric_column=metric_column,
                bin_edges=bin_edges,
                smooth_bins=smooth_bins,
            )
            for df in histories
        ])

        # Un punto viene considerato soltanto quando esiste
        # per tutti e tre i seed.
        valid = np.all(np.isfinite(seed_curves), axis=0)

        mean = np.full(seed_curves.shape[1], np.nan)
        std = np.full(seed_curves.shape[1], np.nan)

        mean[valid] = np.mean(
            seed_curves[:, valid],
            axis=0,
        )

        # ddof=1: deviazione standard campionaria.
        std[valid] = np.std(
            seed_curves[:, valid],
            axis=0,
            ddof=1,
        )

        lower = mean - std
        upper = mean + std

        if metric_column in {"progress", "success"}:
            mean = np.clip(mean, 0.0, 100.0)
            lower = np.clip(lower, 0.0, 100.0)
            upper = np.clip(upper, 0.0, 100.0)

        aggregated[track] = (
            mean,
            lower,
            upper,
        )

    return step_centers, aggregated


def format_millions(value: float, _position: int) -> str:
    """Formatta i global training steps come 0M, 2M, 4M, ..."""

    if abs(value) < 1e-9:
        return "0M"

    return f"{value / 1_000_000:g}M"


def create_plot(
    steps: np.ndarray,
    aggregated: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    percentage_axis: bool,
) -> None:
    """Disegna un grafico con una linea e una banda per circuito."""

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    fallback_colors = list(
        plt.get_cmap("tab10").colors
    )

    for index, (track, values) in enumerate(
        aggregated.items()
    ):
        mean, lower, upper = values

        color = TRACK_COLORS.get(
            track,
            fallback_colors[index % len(fallback_colors)],
        )

        valid_band = (
            np.isfinite(lower)
            & np.isfinite(upper)
        )

        ax.fill_between(
            steps,
            lower,
            upper,
            where=valid_band,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        ax.plot(
            steps,
            mean,
            color=color,
            linewidth=2.2,
            label=f"{track} mean ± std",
        )

    ax.set_title(
        title,
        fontsize=16,
        pad=12,
    )

    ax.set_xlabel(
        "Global Training Steps",
        fontsize=12,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )

    if percentage_axis:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(
            MultipleLocator(20)
        )

    ax.xaxis.set_major_formatter(
        FuncFormatter(format_millions)
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.45,
    )

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=10,
    )

    ax.margins(x=0)
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.bin_size <= 0:
        raise ValueError(
            "--bin-size deve essere positivo."
        )

    if args.smooth_bins <= 0:
        raise ValueError(
            "--smooth-bins deve essere almeno 1."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    histories = [
        load_csv(args.seed_0),
        load_csv(args.seed_1),
        load_csv(args.seed_2),
    ]

    tracks = common_tracks(histories)

    plot_configurations = [
        {
            "metric": "progress",
            "title": "Mean Lap Progress over Training",
            "ylabel": "Mean Lap Progress (%)",
            "filename": (
                f"{args.prefix}_mean_lap_progress.png"
            ),
            "percentage_axis": True,
        },
        {
            "metric": "success",
            "title": "Completion Rate over Training",
            "ylabel": "Completion Rate (%)",
            "filename": (
                f"{args.prefix}_completion_rate.png"
            ),
            "percentage_axis": True,
        },
        {
            "metric": "rewards",
            "title": "Mean Episode Reward over Training",
            "ylabel": "Mean Episode Reward",
            "filename": (
                f"{args.prefix}_mean_reward.png"
            ),
            "percentage_axis": False,
        },
    ]

    created_files = []

    for config in plot_configurations:
        steps, aggregated = aggregate_metric(
            histories=histories,
            tracks=tracks,
            metric_column=config["metric"],
            bin_size=args.bin_size,
            smooth_bins=args.smooth_bins,
        )

        output_path = (
            args.output_dir
            / config["filename"]
        )

        create_plot(
            steps=steps,
            aggregated=aggregated,
            title=config["title"],
            ylabel=config["ylabel"],
            output_path=output_path,
            dpi=args.dpi,
            percentage_axis=config["percentage_axis"],
        )

        created_files.append(output_path)

    print("Grafici creati:")

    for path in created_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()