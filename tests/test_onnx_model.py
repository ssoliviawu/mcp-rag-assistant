import onnxruntime as ort


MODEL_PATH = (
    "models/bge-small-en-v1.5/onnx/model.onnx"
)


def test_onnx_model():

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    print("\nInputs:")

    for input_meta in session.get_inputs():
        print(
            "name:",
            input_meta.name,
            "shape:",
            input_meta.shape,
            "type:",
            input_meta.type,
        )

    print("\nOutputs:")

    for output_meta in session.get_outputs():
        print(
            "name:",
            output_meta.name,
            "shape:",
            output_meta.shape,
            "type:",
            output_meta.type,
        )