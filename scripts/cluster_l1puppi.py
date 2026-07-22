import fastjet as fj
import vector
import awkward as ak

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from matplotlib import pyplot as plt

vector.register_awkward()

JETDEF = fj.JetDefinition(fj.antikt_algorithm, 0.4)

#-----------------------------------------------------------------------------

def _to_jagged(events: NDArray) -> ak.Array:
                                        # Save jagged array structure
    counts = np.fromiter((len(event) for event in events), dtype = np.int64, count = len(events))
    flat = np.concatenate(events).astype(np.float64, copy = False)
    return ak.unflatten(flat, counts)   # Reconstruct jagged structure from counts


def cluster_file(file: Path) -> fj.ClusterSequence:
    data = np.load(file, allow_pickle = True).item()

    pt  = _to_jagged(data['L1Puppi_pt'])
    eta = _to_jagged(data['L1Puppi_eta'])
    phi = _to_jagged(data['L1Puppi_phi'])

    events = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": ak.zeros_like(pt)   # may change, for now assuming mass = 0
        },
        with_name = "Momentum4D"
    )

    return fj.ClusterSequence(events, JETDEF)


def files_iterator(in_dir: Path, desc: str):
    files = sorted(in_dir.glob("*.npy"))
    return tqdm(files, desc = desc)


def plot_multiplicities(in_dir: Path) -> None:
    multiplicities_list = []
    for file in files_iterator(in_dir, "Clustering dataset"):
        cs = cluster_file(file)
        per_jet = ak.num(cs.constituents(), axis = 2)   # per-event list of per-jet counts

        multiplicities_list.append(ak.to_numpy(ak.flatten(per_jet, axis = 1)))

    multiplicities = np.concatenate(multiplicities_list)
    values, counts = np.unique(multiplicities, return_counts = True)

    plt.figure(figsize = (9, 6))

    plt.bar(values, counts, width = 0.85)

    plt.xticks(values)
    plt.xlabel("Multiplicity")
    plt.ylabel("Count")
    plt.semilogy()

    plt.savefig("L1Puppi_multiplicity_distribution.pdf")
    plt.show()


def zero_pad(field: ak.Array, m: int) -> NDArray:
    padded = ak.pad_none(field, m, axis = 1, clip = True)   # clip -> Truncate
    return ak.to_numpy(ak.fill_none(padded, 0.0)).astype(np.float32)


def cluster_flatten_zeropad(file: Path, multiplicity: int) -> dict[str, NDArray]:
    """Cluster one file's events into jets, then zero-pad or truncate each jet's
    constituents to a fixed length `m`.

    Returns dense (n_jets, m) arrays for keys "pt"/"eta"/"phi", plus a "real" 
    mask (1 = particle, 0 = padding). Particle are pt-sorted (descending).
    """
    cs = cluster_file(file)
    jets = ak.flatten(cs.constituents(), axis = 1)   # n_jets * var (particles)
    jets = jets[ak.argsort(jets.pt, axis = 1, ascending = False)]

    channels = { 
        channel: zero_pad(jets[channel], multiplicity) 
        for channel in ("pt", "eta", "phi") 
    }
    channels["real"] = zero_pad(ak.ones_like(jets.pt), multiplicity)

    return channels


def write_metadata(out_dir: Path, multiplicity: int) -> None:
    (out_dir / "metadata.txt").write_text(
        f"Clustered with anti-kt (R = {JETDEF.R()}) and zero-padded/truncated to "
        f"multiplicity = {multiplicity} particles per jet.\n"
        f"Channels: pt, eta, phi, real (1 = particle, 0 = padding).\n"
    )


def build_clustered_dataset(in_dir: Path, out_dir: Path, multiplicity: int) -> None:
    out_dir.mkdir(parents = True, exist_ok = True)

    for file in files_iterator(in_dir, "Clustering & zero-padding dataset"):
        channels = cluster_flatten_zeropad(file, multiplicity)
        np.save(out_dir / file.name, channels, allow_pickle = True)           # type: ignore

    write_metadata(out_dir, multiplicity)


#-----------------------------------------------------------------------------

def main():
    multiplicity = int(input("Desired multiplicity?\n> "))
    in_dir  = Path(input("L1Puppi location?\n> "))
    out_dir = input("Output location? (blank for L1Puppi/../ClusteredL1Puppi/)\n> ")

    if out_dir == "":
        out_dir = in_dir.parent / ("Clustered" + str(in_dir.name))

    out_dir = Path(out_dir)
    # plot_multiplicities(in_dir)
    build_clustered_dataset(in_dir, out_dir, multiplicity = multiplicity)


if __name__ == "__main__":
    main()
