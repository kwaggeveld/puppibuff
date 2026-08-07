from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.utils import to_zip

import sys


def main():                             # Pass the export archive as argument 1
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} [outfile]")

    outfile = sys.argv[1]

    config = FlatPuppiJetConfig(n_steps = 4)
    config.tree_config['n_estimators'] = 20
    config.tree_config['max_depth'] = 2

    _, codec, model, x, y = config.setup()

    model.fit(x, y)

    to_zip(outfile, config, codec, model)   # Read back with utils.from_zip

    print(f"Wrote config, codec and model to {outfile}")


if __name__ == "__main__":
    main()
