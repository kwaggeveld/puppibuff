from puppibuff.datasets import FlatPuppiJet as Dataset
from puppibuff.codecs import FixedMCodec
from puppibuff.analyses import loss_vs_nodes, plot_loss_vs_nodes

from puppibuff import build_trainds

from os import environ



N_STEPS = 15
N_EVENTS = 500_000
N_SAMPLE_EVENTS = 500_000

MAX_DEPTHS   = [ 2, 3, 4, 6, 8 ]
N_ESTIMATORS = [ 5, 10, 20, 50, 100, 200 ]

base_config = {
    "objective": "reg:squarederror",
    "learning_rate": .1,
    "n_jobs": 16,
    "subsample": 1.,
    "reg_alpha": .2,
    "reg_lambda": .1,
    "seed": 666,
    "tree_method": "hist",
    "device": "cpu",
}

def main():
    data = Dataset(environ['PUPPIJET_LOCATION'])

    codec = FixedMCodec()
    codec.fit(data)

    x1 = codec.encode(data)[:N_EVENTS]

    x, y = build_trainds(x1, N_STEPS)

    results = loss_vs_nodes(
        x, y, data, codec,
        max_depths = MAX_DEPTHS,
        n_estimators = N_ESTIMATORS,
        base_config = base_config,
        n_sample_events = N_SAMPLE_EVENTS,
        export = "loss_vs_nodes"
    )

    plot_loss_vs_nodes(results).savefig(f"loss_vs_nodes.pdf", format = "pdf")
    # plot_loss_vs_nodes(results).show()


if __name__ == "__main__":
    main()
