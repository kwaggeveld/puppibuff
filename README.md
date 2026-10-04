# puppibuff

[![CERN](assets/badge_cern.svg)](https://home.cern/)
[![Next Generation Triggers](assets/badge_ngt.svg)](https://nextgentriggers.web.cern.ch/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-016C7A)](https://xgboost.readthedocs.io/)
[![conifer](https://img.shields.io/badge/conifer-%E2%89%A51.9-orange)](https://github.com/thesps/conifer)

**BDT-based flow matching for on-the-fly event generation on FPGA firmware**

A package for designing, training of and sampling from BDT-based flow-matching generative models for ultra-fast simulation of PUPPI particle kinematics. 
Includes a module for writing a trained model to FPGA firmware for end-to-end on-chip sampling for ultra-ultra-fast generation.

## Why this exists

At the CMS experiment, the Level 1 Scouting system is a real-time hardware system that saves and analyses events at the full collision rate of the LHC, using particles reconstructed by the Level 1 Trigger.
A significantly expanded L1 Scouting system is being designed for the CMS Phase 2 Upgrade, with new hardware and improved reconstruction from the L1 Trigger. Stress-testing the capability of the upgraded system needs significant amounts of simulated data.

[BUFF](https://arxiv.org/pdf/2404.18219v1) (Jiang et al.) proposes applying boosted decision trees to the task of generative modelling via flow matching, offering high-fidelity simulation combined with lightweight production of new examples.
`puppibuff` combines that architecture with [conifer](https://github.com/thesps/conifer), already used at CMS for fast BDT inference on L1 Trigger FPGAs, so that the full generator runs on the L1T's FPGAs. 
The result is an on-chip generator that generates PUPPI jet kinematics at 360 MHz, which can be used to stress-test the upgraded Scouting system.

This repository was developed during the CERN Summer Student Programme 2026.

## Repository structure

```
Dataset }
  &      } = Config -> .setup()  ->  FlowBDT.fit()  ->  .sample()  ->  Codec.decode()
Codec   }
```

- **`datasets/`** loads and manipulates dataset `.npy` files to store several structures.
- **`codecs/`** maps between physical and encoded space with different encoding strategies.
- **`build_trainds.py`** builds per-step interpolation paths between Gaussian noise and data.
- **`flowbdt.py`** trains one `XGBRegressor` per (time step, channel) and samples by integrating the learned velocity field.
- **`hls/`** converts a trained grid to HLS code using conifer and merges the whole sampling loop, BDTs and ODE integration, into a single FPGA design.
- **`analyses/`** compares generated vs. real distributions using histograms, KDE contours and metrics.

## Installation

```bash
pip install -e .
```

Jet-clustering script needs an extra:

```bash
pip install -e ".[scripts]"
```

Each `Dataset` reads its input directory from an environment variable (`PUPPIJET_LOCATION`, `CLUSTERED_L1PUPPI_LOCATION`). Set these to your `.npy` data location.

## Quick start

Get an overview of CLI functionality:
```bash
puppibuff --help
```

Train a model, then look at its generated output:

```bash
export PUPPIJET_LOCATION=~/MinBias/PuppiJet

puppibuff train models/my_run
puppibuff plot histograms models/my_run --show
```

`train` writes one archive holding the config, codec and trained model together, which `puppibuff.from_zip` loads. `plot` methods display samples from such a model:

```bash
puppibuff plot histograms models/my_run -n 1e6 -s 0
```

See `puppibuff plot --help` for details. 

### Moving to FPGA

`puppibuff hls` translates a trained model to HLS firmware and checks it against its Python implementation:

```bash
puppibuff hls write  models/my_run -o outdir/            # Write HLS sources
puppibuff hls build  outdir/                             # Synthesise sources
puppibuff hls sample outdir/ models/my_run -n 1e5        # Sample firmware and Python
puppibuff hls plot   outdir/outdir_samples.npz models/my_run
```

See `puppibuff hls --help` for details. 

### In Python

The CLI trains on each `Config`'s defaults. Configure them in Python:

```python
from puppibuff import to_zip
from puppibuff.analyses import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig

config = FlatPuppiJetConfig(n_steps = 15, n_events = 500_000, seed = 0)
config.tree_config["max_depth"] = 6

data, codec, model, x, y = config.setup()   # Load, encode, build training paths
model.fit(x, y)

samples = codec.decode(model.sample(1_000_000))
figure  = plot_histograms(data, samples, n_events = config.n_events)

to_zip("models/my_run", config, codec, model)
```

To train on your own data, subclass `Dataset`, load your set by overriding `_load`, and let `s_CHANNELS` name them. Pair that with a `Codec` in your own `Config`.

## Related works

Waggeveld, K. C. (2026). Ultra-Ultra-Fast Flow Matching for On-the-Fly Event Generation. CERN. https://doi.org/10.17181/n9hgk-sd971

## Contact

Koen Waggeveld at [k.c.waggeveld@student.rug.nl](mailto:k.c.waggeveld@student.rug.nl)
