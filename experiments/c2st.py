from puppibuff.analyses import classifier_two_sample_test, joint_mse
from puppibuff.utils import from_zip

import sys

import numpy as np

#-----------------------------------------------------------------------------

# Classifier two-sample test:
# Trains a BDT to distinguish real data vs sampled data, and computes AOC score 
# to quantify how different the two distributions are

N_SAMPLES = 500_000

CHANNELS = [ "pt", "eta", "phi" ]
                                        # The resolution of training data. Without
                                        # it the classifier scores .93 on float
                                        # precision alone, regardless of the model.
GRID = { "pt": .25 }

def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <model>")

    config, codec, model = from_zip(sys.argv[1])

    data = config.dataset()
                                        # Try to use events that have not been 
                                        # used for training
    holdout = data[config.n_events:] if config.n_events is not None else data

    samples = codec.decode(model.sample(N_SAMPLES))

    auc   = classifier_two_sample_test(holdout, samples, channels = CHANNELS,
                                       quantisation = GRID)

    print(f"\nC2ST over { CHANNELS }, { N_SAMPLES } samples\n"
          f"  AUC       { auc :.4f}    (0.5 => indistinguishable)\n"
          f"  excess    { auc - .5 :+.4f}\n"
          f"  joint_mse { joint_mse(holdout, samples, channels = CHANNELS) :.3e}")


if __name__ == "__main__":
    main()
