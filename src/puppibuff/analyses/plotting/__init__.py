from .style import DOC_WIDTH, use_style
from .histograms import plot_histograms
from .distributions import plot_distributions
from .contours import plot_contours
                             
use_style()

__all__ = [
    "DOC_WIDTH",
    "use_style",
    "plot_histograms",
    "plot_distributions",
    "plot_contours",
]
