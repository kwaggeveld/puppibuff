from puppibuff.configs import FlatPuppiJetConfig

import sys
from pathlib import Path

MODEL_FILE = "flowbdt"
CODEC_FILE = "codec"


def main():                             # Pass the export directory as argument 1
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} [outdir]")

    outdir = Path(sys.argv[1])
    outdir.mkdir(parents = True, exist_ok = True)

    config = FlatPuppiJetConfig()

    _, codec, model, x, y = config.setup()

    model.fit(x, y)

    model.to_disk(str(outdir / MODEL_FILE))
    codec.to_json(str(outdir / CODEC_FILE))

                                        # Read back with FlowBDT.from_disk and
                                        # config.codec_cls.from_json
    print(f"Wrote {outdir / MODEL_FILE} and {outdir / CODEC_FILE}")


if __name__ == "__main__":
    main()
