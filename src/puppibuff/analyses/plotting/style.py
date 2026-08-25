from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

#-----------------------------------------------------------------------------

DOC_WIDTH = 0.9 * 6.3                         # Typical article `\the\textwidth` / 72.28
LEGEND_LOC = "outside upper right"

#--- Style defaults ---
                                        
STYLE = {                               # cajohare/HowToMakeAPlot's `sty.mplstyle`,
    "xtick.direction":     "in",        # minus font size specifications (in `tex`
    "ytick.direction":     "in",        # style)
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":    5.0,
    "ytick.major.size":    5.0,
    "xtick.minor.size":    2.5,
    "ytick.minor.size":    2.5,
    "legend.frameon": False,
    "axes.grid": False,
    # "axes.prop_cycle": cycler(color = [ "#0C5DA5", "#00B945", "#FF9500", "#FF2C00",
    #                                     "#845B97", "#474747", "#9E9E9E" ]),
    "axes.prop_cycle": cycler(color = [ "tab:blue", "tab:green", "tab:orange", "tab:red",
                                        "tab:purple", "tab:brown", "tab:gray" ]),
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
}

FALLBACK = {                            # `TEX_STYLE` without LaTeX
    "text.usetex":           False,
    "font.family":           "serif",
    "mathtext.fontset":      "cm",
    "font.size":             11.0,
    "axes.titlesize":        "medium",
    "figure.labelsize":      "medium",
    "figure.titlesize":      "medium",
    "legend.fontsize":       "small",
    "legend.title_fontsize": "small",
    "xtick.labelsize":       "small",
    "ytick.labelsize":       "small",
}


def use_style(tex: bool = True) -> None:
    """Apply the report style to matplotlib's global rcParams.

    With `tex = False`, use of LaTeX rendering is avoided (for faster plots)
    """
    if tex:
        try:
            plt.style.use("tex")        # My style sheet
        except OSError:                 # Style sheet not installed on this machine
            mpl.rcParams.update(FALLBACK)
    else:
        mpl.rcParams.update(FALLBACK)

    mpl.rcParams.update(STYLE)
