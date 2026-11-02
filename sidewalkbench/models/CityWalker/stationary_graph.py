"""Fix the public ONNX atan2 export at zero coordinates without changing weights."""

import hashlib
from pathlib import Path

import numpy as np


PUBLIC_SHA256 = "e97279414bcb71d258ccf7b7610bee5c1c32bf02d722de55a5b63d430eb370b5"


def patch_stationary_angles(model):
    """Return stable atan2 angles for stationary and straight reset histories.

    Upstream uses torch.atan2(y, x). The ONNX export starts with atan(y / x),
    which produces NaN at (0, 0) and wrong angles on signed-zero axes. Avoid
    zero division, explicitly select vertical-axis angles, and treat signed
    zeros as canonical zeros. Ordinary non-axis points retain the export's
    quadrant correction. The origin angle is defined as zero.
    """
    import onnx
    from onnx import helper, numpy_helper

    prefix = "/inner/cord_embedding/"
    matches = [(i, node) for i, node in enumerate(model.graph.node)
               if node.name == prefix + "Div" and node.op_type == "Div"]
    if len(matches) != 1:
        raise ValueError("Expected the public CityWalker polar-embedding division.")
    index, division = matches[0]
    expected = [prefix + "Gather_1_output_0", prefix + "Gather_output_0"]
    if list(division.input) != expected:
        raise ValueError("Unexpected CityWalker atan2 export inputs.")
    y, x = division.input
    by_name = {node.name: node for node in model.graph.node}
    angle = by_name.get(prefix + "Where_1")
    positive_y = by_name.get(prefix + "Greater")
    if angle is None or angle.op_type != "Where" or positive_y is None or positive_y.op_type != "Greater":
        raise ValueError("Unexpected CityWalker atan2 quadrant correction.")
    tag = "sidewalkbench_citywalker_"
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array(0, np.float32), tag + "zero"),
        numpy_helper.from_array(np.array(1, np.float32), tag + "one"),
        numpy_helper.from_array(np.array(np.pi / 2, np.float32), tag + "half_pi"),
        numpy_helper.from_array(np.array(-np.pi / 2, np.float32), tag + "neg_half_pi"),
    ])
    guards = [
        helper.make_node("Equal", [x, tag + "zero"], [tag + "x_zero"]),
        helper.make_node("Where", [tag + "x_zero", tag + "one", x], [tag + "denominator"]),
    ]
    division.input[1] = tag + "denominator"
    for offset, node in enumerate(guards):
        model.graph.node.insert(index + offset, node)
    # atan2(+0, negative x) is +pi; normalize the sign of an exact zero y.
    positive_y.op_type = "GreaterOrEqual"
    output = angle.output[0]
    angle.output[0] = tag + "nonaxis_angle"
    axes = [
        helper.make_node("Greater", [y, tag + "zero"], [tag + "y_positive"]),
        helper.make_node("Less", [y, tag + "zero"], [tag + "y_negative"]),
        helper.make_node("Where", [tag + "y_negative", tag + "neg_half_pi", tag + "zero"], [tag + "negative_axis"]),
        helper.make_node("Where", [tag + "y_positive", tag + "half_pi", tag + "negative_axis"], [tag + "axis_angle"]),
        helper.make_node("Where", [tag + "x_zero", tag + "axis_angle", tag + "nonaxis_angle"], [output]),
    ]
    index = next(i for i, node in enumerate(model.graph.node) if node.name == prefix + "Where_1") + 1
    for offset, node in enumerate(axes):
        model.graph.node.insert(index + offset, node)
    onnx.checker.check_model(model)
    return model


def safe_stationary_graph(onnx_path):
    """Return a corrected in-memory graph; leave the pinned download intact."""
    import onnx

    data = Path(onnx_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != PUBLIC_SHA256:
        raise ValueError("CityWalker checkpoint differs from the pinned public export.")
    model = onnx.load_model_from_string(data)
    del data
    return patch_stationary_angles(model).SerializeToString()
