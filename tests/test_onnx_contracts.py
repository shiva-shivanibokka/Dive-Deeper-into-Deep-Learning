"""Contract tests for the exported browser models.

These guard the export<->inference boundary: every committed ONNX model must keep
the input/output names, dtypes and shapes the web app feeds it, and the JSON
metadata must stay consistent with the models. If a re-export drifts (renamed I/O,
changed latent dim, wrong maxlen, mismatched label count), CI fails instead of the
demo silently breaking in the browser.

Deps: onnxruntime, numpy, pytest (no torch needed).
"""
import json
import os

import numpy as np
import onnxruntime as ort
import pytest

MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "public", "models")


def sess(name):
    return ort.InferenceSession(os.path.join(MODELS, name), providers=["CPUExecutionProvider"])


def dummy(dtype, shape):
    if dtype == "tensor(int64)":
        return np.zeros(shape, dtype=np.int64)
    return np.random.randn(*shape).astype(np.float32)


# name -> (input_name, dtype, concrete input shape, output_name, expected output shape)
SPECS = {
    "mnist_cnn.onnx":     ("image", "tensor(float)", (1, 1, 28, 28), "logits", (1, 10)),
    "autoencoder.onnx":   ("image", "tensor(float)", (1, 1, 28, 28), "recon",  (1, 1, 28, 28)),
    "vae_decoder.onnx":   ("z",     "tensor(float)", (1, 8),         "image",  (1, 1, 28, 28)),
    "gan_generator.onnx": ("z",     "tensor(float)", (16, 32),       "image",  (16, 1, 28, 28)),
    "vit.onnx":           ("image", "tensor(float)", (1, 1, 28, 28), "logits", (1, 10)),
    "lstm_text.onnx":     ("tokens","tensor(int64)", (1, 80),        "logits", (1, 6)),
    "tabular_mlp.onnx":   ("features","tensor(float)",(1, 6),        "logit",  (1, 1)),
}


@pytest.mark.parametrize("name,inp,dtype,shape,out,outshape", [(k, *v) for k, v in SPECS.items()])
def test_single_input_model(name, inp, dtype, shape, out, outshape):
    s = sess(name)
    assert [i.name for i in s.get_inputs()] == [inp]
    assert s.get_inputs()[0].type == dtype
    assert [o.name for o in s.get_outputs()] == [out]
    result = s.run(None, {inp: dummy(dtype, shape)})[0]
    assert result.shape == outshape
    assert np.isfinite(result).all()


def test_diffusion_unet_two_inputs():
    s = sess("diffusion_unet.onnx")
    names = {i.name: i.type for i in s.get_inputs()}
    assert names == {"x": "tensor(float)", "t": "tensor(int64)"}
    assert [o.name for o in s.get_outputs()] == ["noise"]
    out = s.run(None, {"x": dummy("tensor(float)", (1, 1, 28, 28)),
                       "t": np.array([5], dtype=np.int64)})[0]
    assert out.shape == (1, 1, 28, 28)
    assert np.isfinite(out).all()


def test_lstm_vocab_matches_model():
    v = json.load(open(os.path.join(MODELS, "lstm_vocab.json")))
    # 0 = <pad>, 1 = <unk> are reserved implicitly; real words start at index 2.
    assert min(v["word2idx"].values()) >= 2
    s = sess("lstm_text.onnx")
    assert s.get_inputs()[0].shape[1] == v["maxlen"]          # seq length agrees
    assert s.get_outputs()[0].shape[1] == len(v["labels"])    # one logit per topic label


def test_tabular_meta_consistent():
    m = json.load(open(os.path.join(MODELS, "tabular_meta.json")))
    assert len(m["mean"]) == len(m["std"]) == len(m["numeric"])
    assert set(m["numeric"]).issubset(set(m["features"]))
    assert sess("tabular_mlp.onnx").get_inputs()[0].shape[1] == len(m["features"])


def test_diffusion_schedule_consistent():
    d = json.load(open(os.path.join(MODELS, "diffusion_schedule.json")))
    n = d["steps"]
    assert len(d["betas"]) == len(d["alphas"]) == len(d["abars"]) == n
    assert all(0 < b < 1 for b in d["betas"])


def test_latent_dims_match_frontend_constants():
    # The VAE/GAN latent dims are hardcoded in the frontend (VaeTab Z=8, GanTab ZDIM=32);
    # a re-export with a different dim would silently break the tensors the browser sends.
    assert sess("vae_decoder.onnx").get_inputs()[0].shape[1] == 8
    assert sess("gan_generator.onnx").get_inputs()[0].shape[1] == 32
