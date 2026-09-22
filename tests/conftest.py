"""Test fixtures: a tiny synthetic ONNX classifier so the serving path runs without any download."""

import io

import numpy as np
import pytest
from PIL import Image

N_CLASSES = 10


@pytest.fixture(scope="session")
def synthetic_model(tmp_path_factory) -> tuple[str, str]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    weight = numpy_helper.from_array(rng.standard_normal((3, N_CLASSES)).astype(np.float32), "W")
    bias = numpy_helper.from_array(np.zeros(N_CLASSES, dtype=np.float32), "b")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 224, 224])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", N_CLASSES])
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "b"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "tiny-classifier", [inp], [out], initializer=[weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    folder = tmp_path_factory.mktemp("model")
    model_path = folder / "tiny.onnx"
    onnx.save(model, str(model_path))
    labels_path = folder / "labels.txt"
    labels_path.write_text("\n".join(f"class-{i}" for i in range(N_CLASSES)), "utf-8")
    return str(model_path), str(labels_path)


@pytest.fixture(scope="session")
def jpeg_bytes() -> bytes:
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
