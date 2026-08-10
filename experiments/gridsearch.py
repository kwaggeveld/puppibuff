from __future__ import annotations

from puppibuff.analyses.losses import channel_wasserstein, joint_mse
from puppibuff.analyses.plotting import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.datasets import FlatPuppiJet
from puppibuff.flowbdt import FlowBDT

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

#-----------------------------------------------------------------------------

# Hyperparameter gridsearch over FlatPuppiJet. Every combination in the
# Cartesian product of the axes below is trained, sampled and scored against the
# data, one figure per combination. Runs sequentially.

MAX_DEPTH    = [ 2, 3, 4, 6 ]           # XGBoost tree depth
N_ESTIMATORS = [ 10, 20, 50, 100 ]      # Trees per BDT
S1PHI        = [ False, True ]          # Encode phi as (sin, cos) vs. one normalised channel
N_STEPS      = [ 5, 10, 15 ]            # Flow-matching discretisation
N_EVENTS     = [ 500_000, 2_000_000 ]   # Training-set size; None => whole dataset

N_SAMPLES    = 1_000_000                # Events drawn from each trained model

                                        # Grid axes, ordered as product() combines
                                        # them into each grid point below
AXES: tuple[str, ...] = ( "max_depth", "n_estimators", "s1phi", "n_steps", "n_events" )

NEST_BY = [ "s1phi", "n_steps" ]        # Nest output figures one subdir per axis
                                        # listed here, in order; the remaining axes
                                        # go into the filename. [] => flat

#-----------------------------------------------------------------------------

def make_config(max_depth: int, n_estimators: int, s1phi: bool,
                n_steps: int, n_events: int | None) -> FlatPuppiJetConfig:
    """Return config specific to one grid point."""
    config = FlatPuppiJetConfig(n_steps = n_steps, n_events = n_events, s1phi = s1phi)
    config.tree_config["max_depth"]    = max_depth
    config.tree_config["n_estimators"] = n_estimators
    return config


def model_size(model: FlowBDT, n_estimators: int, max_depth: int) -> int:
    """Compute upper-bound of the total leaf count as proxy for model size:
        (# BDTs) x (# trees / BDT) x (# leaves / tree)
    """
    return model.bdt_grid.size * n_estimators * 2 ** max_depth


def suptitle(grid_pt: tuple, size: int, per_channel: dict[str, float],
             mse: float) -> str:
    """Two-line figure caption: the grid point, then its scores."""
    max_depth, n_estimators, s1phi, n_steps, n_events = grid_pt

    params   = (f"{max_depth = } | {n_estimators = } | {s1phi = }  "
                f"{n_steps = } | {n_events = :,} | {size = :.2e}")
    channels = "  ".join( f"{ name } { value :.4g}" for name, value in per_channel.items() )
    scores   = f"joint MSE = { mse :.4g}    |    per-channel W:  { channels }"

    return params + "\n" + scores


def axis_token(axis: str, value) -> str:
    """Short token for one grid-axis value (subdir or filename part)."""
    if axis == "s1phi":
        return f"s1{ 'T' if value else 'F' }"
    if axis == "n_events":
        if value is None:
            return "neAll"
        for div, suffix in ( (1_000_000, "M"), (1_000, "k") ):
            if value % div == 0:
                return f"ne{ value // div }{ suffix }"
        return f"ne{ value }"                # Not a round k/M -> full count
    return { "max_depth": "d", "n_estimators": "nt", "n_steps": "ns" }[axis] + str(value)


def output_path(grid_pt: tuple) -> Path:
    """Relative figure path for one grid point: a subdir per `NEST_BY` axis, the
    remaining axes joined into the filename (e.g. s1T/ns15/d6_nt100_ne500k.pdf).
    """
    values  = dict(zip(AXES, grid_pt))
    subdirs = [ axis_token(axis, values[axis]) for axis in NEST_BY ]
    name    = "_".join(axis_token(axis, values[axis])
                       for axis in AXES if axis not in NEST_BY) + ".pdf"
    return Path(*subdirs, name)


def report(results: list[dict], outdir: Path) -> None:
    """Print every combination ranked by joint MSE (best first) and save the
    same table to `summary.txt` beside the figures.
    """
    channels = list(results[0]["per_channel"])

    header = (f"{ 'depth' :>5} { 'trees' :>5} { 's1phi' :>5} { 'steps' :>5} { 'events' :>8} "
              f"{ 'size' :>10} " + " ".join(f"{ 'W_' + name :>9}" for name in channels)
              + f" { 'MSE_3d' :>9}")

    lines = [ header ]
    for result in sorted(results, key = lambda result: result["mse"]):
        max_depth, n_estimators, s1phi, n_steps, n_events = result["combo"]
        per_channel = " ".join(f"{ result['per_channel'][name] :>9.4g}" for name in channels)

        lines.append(
            f"{ max_depth :>5} { n_estimators :>5} { str(s1phi) :>5} { n_steps :>5} "
            f"{ str(n_events) :>8} { result['size'] :>10.2e} { per_channel } "
            f"{ result['mse'] :>9.4g}"
        )

    table = "\n".join(lines)
    print("\nRanked by joint MSE (best first):\n")
    print(table)
    (outdir / "summary.txt").write_text(table + "\n")


def main():  # NB: tqdm.write used instead of print() to preserve progress bar
                                        # Create output directory
    outdir = Path(__file__).resolve().parent / "output" / Path(__file__).stem
    outdir.mkdir(parents = True, exist_ok = True)

    grid  = list(product(MAX_DEPTH, N_ESTIMATORS, S1PHI, N_STEPS, N_EVENTS))
    results = []

    data = FlatPuppiJet()               # Shared across runs

    for index, grid_pt in enumerate(tqdm(grid, desc = "Gridsearch"), start = 1):
        max_depth, n_estimators, s1phi, n_steps, n_events = grid_pt
        tqdm.write(f"\n[{ index }/{ len(grid) }] {max_depth = } | "
                   f"{n_estimators = } | {s1phi = } | "
                   f"{n_steps = } | {n_events = :,}")

        config = make_config(*grid_pt)
        _, codec, model, x, y = config.setup(data)      # Reuse the shared dataset
        model.fit(x, y)

        samples = codec.decode(model.sample(N_SAMPLES))

                                        # Per-channel Wasserstein for the plot,
                                        # joint MSE ranks models
        per_channel = { channel: channel_wasserstein(data[channel], samples[channel])
                        for channel in data.channels() }
        mse  = joint_mse(data, samples)
        size = model_size(model, n_estimators, max_depth)

                                        # Only sampled vs target distributions
        figure = plot_histograms(data, samples, n_events = None)
        figure.suptitle(suptitle(grid_pt, size, per_channel, mse), fontsize = 11)
        figure.subplots_adjust(top = 0.86)      # Room for the two-line suptitle

        path = outdir / output_path(grid_pt)
        path.parent.mkdir(parents = True, exist_ok = True)
        figure.savefig(path)
        plt.close(figure)

        results.append({ "combo": grid_pt, "size": size,
                         "per_channel": per_channel, "mse": mse })
        tqdm.write(f"    joint MSE = { mse :.4g}   size = { size :.2e}")

    report(results, outdir)


if __name__ == "__main__":
    main()
