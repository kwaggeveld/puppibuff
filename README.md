# puppibuff

[![CERN](assets/badge_cern.svg)](https://home.cern/)
[![Next Generation Triggers](assets/badge_ngt.svg)](https://nextgentriggers.web.cern.ch/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-016C7A)](https://xgboost.readthedocs.io/)
[![conifer](https://img.shields.io/badge/conifer-%E2%89%A51.9-orange)](https://github.com/thesps/conifer)

**Ultra-Ultra-Fast Flow Matching for on-the-fly event generation.**

A package for designing, training of and sampling from BDT-based flow-matching generative models for ultra-fast simulation of PUPPI particle kinematics. 
Includes a module for writing a trained model to FPGA firmware for end-to-end on-chip sampling for ultra-ultra-fast generation.

## Why this exists

At the CMS experiment, the Level 1 Scouting system is a real-time hardware system that saves and analyses events at the full collision rate of the LHC, using particles reconstructed by the Level 1 Trigger.
A significantly expanded L1 Scouting system is being designed for the CMS Phase 2 Upgrade, with new hardware and improved reconstruction from the L1 Trigger. Stress-testing the capability of the upgraded system needs significant amounts of simulated data.

[BUFF](https://arxiv.org/pdf/2404.18219v1) (Jiang et al.) proposes applying boosted decision trees to the task of generative modelling via flow matching, offering high-fidelity simulation combined with lightweight production of new examples.
`puppibuff` combines that architecture with [conifer](https://github.com/thesps/conifer), already used at CMS for fast BDT inference on L1 Trigger FPGAs, so that the full generator runs on the L1T's FPGAs. 
The result is an on-chip generator that generates PUPPI jet kinematics at 360 MHz, which can be used to stress-test the upgraded Scouting system.

This repository was developed during the CERN Summer Student Programme 2026. Corresponding report: to be added.

## Repository structure

```
Dataset }
  &      } = Config -> .setup()  ->  FlowBDT.fit()  ->  .sample()  ->  Codec.decode  ->  analyses
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

Jet-clustering script need an extra:

```bash
pip install -e ".[scripts]"
```

Each `Dataset` reads its input directory from an environment variable (`PUPPIJET_LOCATION`, `CLUSTERED_L1PUPPI_LOCATION`). Set these to your `.npy` data location.

## Quick start

Coming soon...

## Contact

Koen Waggeveld at [k.c.waggeveld@student.rug.nl](mailto:k.c.waggeveld@student.rug.nl)
