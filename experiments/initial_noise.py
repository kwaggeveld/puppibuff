from puppibuff.analyses import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

# Supplying a different prior distribution to the fitting and sampling steps
#   Idea: `phi` is uniform on [-pi, pi] (encoded [-\sqrt3, \sqrt3]) (*)
#   and supplying this as prior for `phi` may make the flow field easier to learn.


# (*) X ~ U[a, b] => Var(X) = (b - a)^2 / 12. For `phi`, Var(`phi`) = pi^2 / 3.
# Thus dividing by `std` = pi / \sqrt3 gives normalised U[-\sqrt3, \sqrt3]

def noise(rng: np.random.Generator, n_events: int) -> NDArray:
    x0 = np.empty((n_events, 3), dtype = np.float32)
    x0[:, :2] = rng.standard_normal((n_events, 2))
    x0[:, 2] = rng.uniform(-np.sqrt(3), np.sqrt(3), size = n_events)
    return x0

def main():
    config = FlatPuppiJetConfig()
    config.n_events = 500_000

    rng = np.random.default_rng()
    x0 = noise(rng, config.n_events)

    data, codec, model, x, y = config.setup(x0 = x0)

    model.fit(x, y)

    x0 = noise(rng, config.n_events)
    samples = codec.decode(model.sample(500_000, x0 = x0))

    figure = plot_histograms(
        data, samples, n_events = config.n_events,
    )

    figure.show()
    input()


if __name__ == "__main__":
    main()
