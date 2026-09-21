from .common import DOC_WIDTH
from .histograms import plot_histograms
from .distributions import plot_distributions
from .contours import plot_contours

import logging

#-----------------------------------------------------------------------------
            # Supresses error:
            #   'created' timestamp seems very low; regarding as unix timestamp
            #   'modified' timestamp seems very low; regarding as unix timestamp
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

__all__ = [
    "DOC_WIDTH",
    "plot_histograms",
    "plot_distributions",
    "plot_contours",
]
