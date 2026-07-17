from __future__ import annotations

from ..flowbdt import FlowBDT
from ..build_trainds import Paths
from ..datasets import Dataset
from ..codecs import Codec
from .losses import total_mse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from numpy.typing import NDArray
from matplotlib.figure import Figure


def count_nodes(model: FlowBDT) -> int:
    """Total number of nodes across every (step, channel) BDT in the grid."""
    return sum(
        len(model.bdt_grid[step, channel].get_booster().trees_to_dataframe())
        for step    in range(model.n_steps)
        for channel in range(model.n_channels)
    )


def loss_vs_nodes(
    x: Paths,
    y: NDArray,
    data: Dataset,
    codec: Codec,
    max_depths: list[int],
    n_estimators: list[int],
    base_config: dict,
    n_sample_events: int = 500_000,
    channels: list[str] | None = None,
    n_threads: int = 1,
    export: str | None = None
) -> dict[int, tuple[list[int], list[float]]]:
                                        # One (nodes, loss) curve per max_depth;
    results = {}                        # each n_estimators value is a full retrain
    for max_depth in tqdm(max_depths, desc = "max_depths"):  
        nodes, losses = [], []

        for n in tqdm(n_estimators, desc = "n_estimators"):
            config = dict(base_config, max_depth = max_depth, n_estimators = n)

            model = FlowBDT(config)
            model.fit(x, y, n_threads = n_threads)

            samples = codec.decode(model.sample(n_sample_events))

            nodes.append(count_nodes(model))
            losses.append(total_mse(data, samples, channels = channels))

        results[max_depth] = (nodes, losses)

    if export: np.save(export, results)                                       # type: ignore

    return results


def plot_loss_vs_nodes(results: dict[int, tuple[list[int], list[float]]]) -> Figure:
    fig, ax = plt.subplots(figsize = (6, 5))

    for max_depth, (nodes, losses) in sorted(results.items()):
        ax.plot(nodes, losses, marker = "o", label = f"max_depth = {max_depth}")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("# tree nodes")
    ax.set_ylabel("Total MSE loss")
    ax.legend()

    fig.tight_layout()

    return fig
