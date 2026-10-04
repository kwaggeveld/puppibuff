# Public interface

from .losses import (channel_mse as channel_mse, total_mse as total_mse,
                     channel_wasserstein as channel_wasserstein,
                     joint_mse as joint_mse,
                     sliced_wasserstein as sliced_wasserstein,
                     classifier_two_sample_test as classifier_two_sample_test)
from .node_analysis import (count_nodes as count_nodes,
                            loss_vs_nodes as loss_vs_nodes,
                            plot_loss_vs_nodes as plot_loss_vs_nodes)
from .plotting import (DOC_WIDTH as DOC_WIDTH,
                       plot_histograms as plot_histograms,
                       plot_distributions as plot_distributions,
                       plot_contours as plot_contours)
