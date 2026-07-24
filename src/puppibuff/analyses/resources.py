from ..flowbdt import FlowBDT

from conifer.converters import convert_from_xgboost
from conifer.utils.performance import performance_estimates
from conifer.backends.xilinxhls import auto_config as xilinxhls_config

#-----------------------------------------------------------------------------

def resource_estimates(model: FlowBDT) -> dict[str, int]:
    """Use conifer to estimate resource (lut, ff) requirements
    for entire BDT ensamble of `model`
    """
    ret = {
        "lut": 0,
        "ff": 0
    }

    config = xilinxhls_config()
    for tree in model.bdt_grid.flat:    # Iterate over all BDT's in the model
        conifer_model = convert_from_xgboost(tree, config)

        estimates = performance_estimates(conifer_model)

        for key in ret.keys():
            ret[key] += estimates[key]

    return ret
