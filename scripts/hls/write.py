from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.hls import FlowHLS

from puppibuff import setup_from_config

import sys

PER_BDT = True

def main():                             # Pass directory for the HLS project
    if len(sys.argv) >= 2:              # as argument 1
        output_dir = sys.argv[1]
    else:
        output_dir = "flowhls"

    config = FlatPuppiJetConfig(s1phi = False,
                                n_steps = 4,
                                n_events = 500_000)
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2

    _, _, model, x, y = setup_from_config(config)

    model.fit(x, y)
    hls = FlowHLS.convert(model, output_dir = output_dir)
    hls.write(per_bdt = PER_BDT)

if __name__ == "__main__":
    main()
