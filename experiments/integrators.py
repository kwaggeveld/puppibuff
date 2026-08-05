from puppibuff.analyses.losses import channel_mse
from puppibuff.analyses.plotting import plot_distributions_flattened
from puppibuff.codecs import Codec
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.datasets import Dataset
from puppibuff.flowbdt import FlowBDT
from puppibuff.solvers import ab2_solve, euler_solve, heun_solve, midpoint_solve
from puppibuff.utils import initial_noise

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from numpy.typing import NDArray

#-----------------------------------------------------------------------------

# Which ODE solver the sampler should use, judged on FPGA cost rather than on
# order of accuracy. `hls.write` unrolls the sampling loop and instantiates a
# step's BDTs once per field evaluation, so the whole design's LUT/FF scale with
# the evaluation count: euler is exactly half a midpoint sampler.

N_STEPS      = 9
N_ESTIMATORS = 50
MAX_DEPTH    = 4

N_SAMPLES  = 500_000
N_REPEATS  = 3                          # Independent noise draws, to compute std

CHANNELS = [ "pt", "eta", "phi" ]

                                        # Field evaluations per interval
SOLVERS = {
    "euler":    (euler_solve,    1),
    "ab2":      (ab2_solve,      1),
    "midpoint": (midpoint_solve, 2),
    "heun":     (heun_solve,     2),
}


def make_config(n_steps: int) -> FlatPuppiJetConfig:
    config = FlatPuppiJetConfig(n_steps = n_steps, s1phi = False)
    config.tree_config["n_estimators"] = N_ESTIMATORS
    config.tree_config["max_depth"]    = MAX_DEPTH

    return config


def evaluate(
        data: Dataset,
        codec: Codec,
        model: FlowBDT,
        solver: Callable,
        x0: NDArray,
    ) -> tuple[dict[str, NDArray], dict[str, float]]:
    """Integrate the field with `solver` from `x0`, decode, and score every
    channel against the full dataset.
    """
    samples = codec.decode(solver(model.predict, x0, model.n_steps))

    return samples, { channel: channel_mse(data[channel], samples[channel])
                      for channel in CHANNELS }


def report(runs: list[tuple], losses: dict[str, list[dict]], n_groups: int) -> None:
    columns = CHANNELS + [ "total" ]

    print(f"\nHistogram MSE against the full dataset, mean +- std over "
          f"{N_REPEATS} noise draws.")
    print(f"evals = field evaluations per sample = `flowhls_field` instances; "
          f"BDTs = evals * {n_groups} groups.\n")

    print(f"{'solver':<12}{'evals':>6}{'BDTs':>6}"
          + "".join(f"{column:>21}" for column in columns))

    for label, _, _, _, evals in runs:
        per_channel = { channel: [ loss[channel] for loss in losses[label] ]
                        for channel in CHANNELS }
                                        # Mean over channels == analyses.total_mse
        per_channel["total"] = [np.mean(list(loss.values()))
                                for loss in losses[label]]

        cells = "".join(f"{np.mean(per_channel[column]):11.3e}"
                        f" +-{np.std(per_channel[column], ddof = 1):7.1e}"
                        for column in columns)

        print(f"{label:<12}{evals:>6}{evals * n_groups:>6}{cells}")


def main():
                                        # Create output directory
    outdir = Path(__file__).resolve().parent / "output" / Path(__file__).stem
    outdir.mkdir(exist_ok = True)

    config = make_config(N_STEPS)

    data, codec, model, x, y = config.setup()
    model.fit(x, y)


    runs = [ (name, model, codec, solver, per_interval * (N_STEPS - 1))
             for name, (solver, per_interval) in SOLVERS.items() ]

    losses = { label: [] for label, *_ in runs }
    for repeat in range(N_REPEATS):
                                        # Shared by every run of this repeat, so
                                        # the solvers differ only in how they
                                        # integrate. Both models encode to the
                                        # same channels, so both can start here
        x0 = initial_noise(N_SAMPLES, model.n_channels)

        for label, run_model, run_codec, solver, _ in runs:
            samples, channel_losses = evaluate(data, run_codec, run_model,
                                               solver, x0)
            losses[label].append(channel_losses)

            if repeat > 0: continue     # One plot per method

            figure = plot_distributions_flattened(
                data, samples, channels = CHANNELS, n_events = config.n_events,
            )
            figure.savefig(f"{outdir}/{label}.pdf")
            plt.close(figure)

    report(runs, losses, model.bdt_grid.shape[1])


if __name__ == "__main__":
    main()
