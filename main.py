from puppibuff.datasets import FlatPuppiJet as Dataset
from puppibuff.codecs import FixedMCodec
from puppibuff.validation import plot_distributions

from puppibuff import build_trainds, FlowBDT

N_STEPS = 15                            # Will come from config later
N_EVENTS = 200_000

                                        # Config copied from BUFF .ipynb
tree_config = {
    "n_estimators": 50,                 # Number of gradient boosted trees
    "max_depth": 4,                     # Max tree depth (for "base learners"?)
    "objective": "reg:squarederror",
    "learning_rate": .1,
    "n_jobs": 16,                       # Number of parallel threads used
    "subsample": 1.,                    # Fraction of events sampled before growing 
    "reg_alpha": .2,                    # L1 regularisation term on weights
    "reg_lambda": .1,                   # L2 regularisation term on weights
    "seed": 666,
    "tree_method": "hist",
    "device": "cpu",
}

def main():
    data = Dataset("/Users/koenwaggeveld/MinBias/PuppiJet")

    codec = FixedMCodec()
    codec.fit(data)
    # codec.to_json("out.json")

    x1 = codec.encode(data)[:N_EVENTS]

    x, y = build_trainds(x1, N_STEPS)

    model = FlowBDT(tree_config)
    model.fit(x, y)

    samples = codec.decode(model.sample(N_EVENTS))
    plot_distributions(data, samples).show()


if __name__ == "__main__":
    main()
