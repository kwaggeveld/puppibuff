from puppibuff.analyses import plot_histograms, plot_contours
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff import FlowBDT, build_trainds

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

    data = config.dataset()
    codec = config.codec()
    codec.fit(data)

    x1 = codec.encode(data[:config.n_events])

    rng = np.random.default_rng()
    x0 = noise(rng, config.n_events)
    x0[:, 2] = x1[:, 2]

    x, y = build_trainds(x1, config.n_steps, x0)

    model = FlowBDT()
    model.fit(x, y)

    x0 = noise(rng, config.n_events)
    samples = codec.decode(model.sample(500_000, x0 = x0))

    figure = plot_contours(
        data, samples, n_events = config.n_events,
    )

    figure.show()
    input()


if __name__ == "__main__":
    main()
