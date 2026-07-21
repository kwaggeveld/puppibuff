from puppibuff.analyses import loss_vs_nodes, plot_loss_vs_nodes
from puppibuff.configs import FlatPuppiJetConfig

from puppibuff import setup_from_config

#-----------------------------------------------------------------------------

MAX_DEPTHS   = [ 2, 3, 4, 6, 8 ]
N_ESTIMATORS = [ 5, 10, 20, 50, 100, 200 ]

def main():
    config = FlatPuppiJetConfig()

    data, codec, _, x, y = setup_from_config(config)

    results = loss_vs_nodes(
        x, y, data, codec,
        max_depths = MAX_DEPTHS,
        n_estimators = N_ESTIMATORS,
        base_config = config.tree_config,
        n_sample_events = config.n_sample_events,
        export = "loss_vs_nodes"
    )

    plot_loss_vs_nodes(results).savefig(f"loss_vs_nodes.pdf", format = "pdf")
    # plot_loss_vs_nodes(results).show()


if __name__ == "__main__":
    main()
