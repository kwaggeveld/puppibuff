from logging import getLogger, ERROR

for logger in ("conifer.converters",                     # ydf, xgboost >= 2.0
               "conifer.backends.xilinxhls.runtime",     # pynq, xrt
               "conifer.backends.fpu.runtime",           # pynq, xrt
               "conifer.utils.performance.prediction"):  # experimental banner
    getLogger(logger).setLevel(ERROR)

getLogger("conifer.utils.misc").addFilter(lambda r: "Could not find hls_stream" not in r.getMessage())

from .flowhls import FlowHLS

__all__ = [
    "FlowHLS",
]
