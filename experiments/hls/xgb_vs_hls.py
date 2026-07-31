from puppibuff.analyses.plotting import plot_distributions_jet
from puppibuff.build_trainds import Paths
from puppibuff.configs import MultiplicityL1PuppiConfig
from puppibuff.fpga import FlowHLS

from puppibuff import FlowBDT, setup_from_config

import sys
from pathlib import Path
import numpy as np

N_SAMPLES  = 500_000

def compare_grid(model: FlowBDT, hls: FlowHLS, x: Paths) -> None:
    """Per-BDT XGBoost vs HLS residuals, on the inputs the BDTs trained on."""
    rows = []
    for step in range(model.n_steps):
        xt = x[step][:N_SAMPLES]

        for group, (bdt, cmodel) in enumerate(zip(model.bdt_grid[step], hls.bdt_grid[step])):
            residual = cmodel.decision_function(xt) - bdt.predict(xt)

            rows.append((step, group, residual.mean(),
                         np.abs(residual).mean(), np.abs(residual).max()))

    stats = np.array([row[2:] for row in rows])
    print(f"\n{len(rows)} BDTs: |bias| <= {np.abs(stats[:, 0]).max():.5f}, "
          f"mad <= {stats[:, 1].max():.5f}, max <= {stats[:, 2].max():.5f}")

    print(f"\n{'step':>4} {'group':>5} {'bias':>10} {'mad':>10} {'max':>10}"
          f"   (5 worst by |bias|)")
    for step, group, bias, mad, worst in sorted(rows, key = lambda row: -abs(row[2]))[:5]:
        print(f"{step:4d} {group:5d} {bias:+10.5f} {mad:10.5f} {worst:10.5f}")


def main():                             # Pass directory with compiled BDTs as 
    if len(sys.argv) < 2:               # argument 1
        sys.exit(f"Usage: {sys.argv[0]} [workdir]")

    workdir = sys.argv[1]

    config = MultiplicityL1PuppiConfig(s1phi = False,
                                       n_steps = 4,
                                       n_events = 100_000)
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2

    data, codec, model, x, y = setup_from_config(config)

    model.fit(x, y)

    hls = FlowHLS.load(workdir)

    compare_grid(model, hls, x)

    print(f"\nResources: {hls.resource_estimates()}")

    reuse = True
                                        # `build_trainds` noise is unseeded, so
                                        # a reused grid has no XGBoost partner:
                                        # `model` here was never fitted
    samplers = [("hls", hls)] if reuse else [("xgboost", model), ("hls", hls)]

    outdir = Path(__file__).resolve().parent / Path(__file__).stem
    outdir.mkdir(exist_ok = True)

                                        # Same target, both velocity fields
    for label, sampler in samplers:
        figure = plot_distributions_jet(
            data, codec.decode(sampler.sample(N_SAMPLES)),
            channels = ["pt", "eta", "phi"], n_events = config.n_events,
        )
        figure.suptitle(label)
        workdir_name = Path(workdir).name
        figure.savefig(outdir / f"{workdir_name}_{label}.pdf", format = "pdf")


if __name__ == "__main__":
    main()
