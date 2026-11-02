"""SidewalkBench-PG: pedestrian-reactive sidewalk navigation environments.

Subpackages:
    pg: environment, scene, primitives, and utilities.
    models: pre-trained navigation model inference wrappers.
"""
import onnxruntime as _ort

# Fail loudly rather than silently falling back to CPU inference. The CUDA and
# cuDNN libraries are made loadable by importing `torch` (a core dependency)
# before `onnxruntime`; the model wrappers already import them in that order.
if "CUDAExecutionProvider" not in _ort.get_available_providers():
    raise RuntimeError(
        "sidewalkbench requires onnxruntime with CUDAExecutionProvider; "
        f"available providers: {_ort.get_available_providers()}"
    )
