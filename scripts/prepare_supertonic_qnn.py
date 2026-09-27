#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import snapshot_download
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
REPO_ID = "Supertone/supertonic-3"
TEXT_LEN = 128
LATENT_LEN = 128

SAMPLES = [
    ("pl", "Witamy w naszej aplikacji VoiceStudio."),
    ("pl", "To jest lokalny test syntezy mowy na telefonie."),
    ("pl", "Dzisiaj sprawdzamy jakość i szybkość generowania głosu."),
    ("en", "Hello from VoiceStudio running locally on Android."),
    ("en", "This sentence is used to calibrate the mobile speech model."),
    ("de", "Dies ist ein kurzer Kalibrierungssatz für die Sprachsynthese."),
    ("fr", "Ceci est une phrase de calibration pour la synthèse vocale."),
    ("es", "Esta es una frase de calibración para la síntesis de voz."),
]

VOICES = ["M1", "F3"]


def flatten_style(obj: dict) -> np.ndarray:
    dims = tuple(int(x) for x in obj["dims"])
    data = np.asarray(obj["data"], dtype=np.float32)
    return data.reshape(dims)


def preprocess(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFKD", text.strip())
    t = t.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")
    t = " ".join(t.split())
    if not t.endswith((".", "!", "?", ";", ":")):
        t += "."
    return f"<{lang}>{t}</{lang}>"


def tokenize(text: str, indexer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cps = [ord(ch) for ch in text]
    if len(cps) > TEXT_LEN:
        cps = cps[:TEXT_LEN]
    ids = np.zeros((1, TEXT_LEN), dtype=np.int64)
    mask = np.zeros((1, 1, TEXT_LEN), dtype=np.float32)
    for i, cp in enumerate(cps):
        if cp >= len(indexer):
            raise ValueError(f"Unsupported U+{cp:04X}")
        ids[0, i] = int(indexer[cp])
        mask[0, 0, i] = 1.0
    return ids, mask


def specialize_model(src: Path, dst: Path) -> None:
    model = onnx.load(src, load_external_data=True)
    for value in list(model.graph.input) + list(model.graph.output):
        shape = value.type.tensor_type.shape
        for dim in shape.dim:
            name = dim.dim_param
            if name == "batch_size":
                dim.ClearField("dim_param")
                dim.dim_value = 1
            elif name == "text_length":
                dim.ClearField("dim_param")
                dim.dim_value = TEXT_LEN
            elif name == "latent_length":
                dim.ClearField("dim_param")
                dim.dim_value = LATENT_LEN
    onnx.save_model(model, dst)


class ListReader(CalibrationDataReader):
    def __init__(self, rows: list[dict[str, np.ndarray]]):
        self.rows = rows
        self._iter = iter(self.rows)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self.rows)


def collect_calibration(root: Path):
    onnx_dir = root / "onnx"
    indexer = np.asarray(json.loads((onnx_dir / "unicode_indexer.json").read_text()), dtype=np.int64)

    styles = {}
    for voice in VOICES:
        obj = json.loads((root / "voice_styles" / f"{voice}.json").read_text())
        styles[voice] = {
            "ttl": flatten_style(obj["style_ttl"]),
            "dp": flatten_style(obj["style_dp"]),
        }

    sessions = {
        "dp": ort.InferenceSession(str(onnx_dir / "duration_predictor.onnx"), providers=["CPUExecutionProvider"]),
        "te": ort.InferenceSession(str(onnx_dir / "text_encoder.onnx"), providers=["CPUExecutionProvider"]),
        "ve": ort.InferenceSession(str(onnx_dir / "vector_estimator.onnx"), providers=["CPUExecutionProvider"]),
        "voc": ort.InferenceSession(str(onnx_dir / "vocoder.onnx"), providers=["CPUExecutionProvider"]),
    }

    rows = {"dp": [], "te": [], "ve": [], "voc": []}
    rng = np.random.default_rng(0x5354)

    for lang, raw in SAMPLES:
        text = preprocess(raw, lang)
        ids, text_mask = tokenize(text, indexer)

        for voice in VOICES:
            style_ttl = styles[voice]["ttl"].astype(np.float32)
            style_dp = styles[voice]["dp"].astype(np.float32)

            dp_feed = {
                "text_ids": ids,
                "style_dp": style_dp,
                "text_mask": text_mask,
            }
            rows["dp"].append(dp_feed)

            te_feed = {
                "text_ids": ids,
                "style_ttl": style_ttl,
                "text_mask": text_mask,
            }
            rows["te"].append(te_feed)
            text_emb = sessions["te"].run(None, te_feed)[0].astype(np.float32)

            latent = rng.normal(0.0, 1.0, size=(1, 144, LATENT_LEN)).astype(np.float32)
            latent_mask = np.ones((1, 1, LATENT_LEN), dtype=np.float32)

            for step in (0, 1, 3, 5):
                ve_feed = {
                    "noisy_latent": latent,
                    "text_emb": text_emb,
                    "style_ttl": style_ttl,
                    "latent_mask": latent_mask,
                    "text_mask": text_mask,
                    "current_step": np.asarray([step], dtype=np.float32),
                    "total_step": np.asarray([6], dtype=np.float32),
                }
                rows["ve"].append(ve_feed)
                latent = sessions["ve"].run(None, ve_feed)[0].astype(np.float32)

            rows["voc"].append({"latent": latent})

    return rows


def quantize_one(src: Path, dst: Path, reader: CalibrationDataReader) -> None:
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "DedicatedQDQPair": True,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="supertonic3-qnn")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    work = out / "work"
    src_root = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            allow_patterns=[
                "onnx/*.onnx",
                "onnx/*.json",
                "voice_styles/*.json",
                "LICENSE",
            ],
        )
    )

    if out.exists():
        shutil.rmtree(out)
    (work / "onnx-static").mkdir(parents=True)
    (out / "onnx").mkdir(parents=True)
    (out / "voice_styles").mkdir(parents=True)

    for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"):
        specialize_model(
            src_root / "onnx" / f"{name}.onnx",
            work / "onnx-static" / f"{name}.onnx",
        )

    rows = collect_calibration(src_root)

    mapping = {
        "duration_predictor": "dp",
        "text_encoder": "te",
        "vector_estimator": "ve",
        "vocoder": "voc",
    }

    for name, key in mapping.items():
        print(f"Quantizing {name} with {len(rows[key])} calibration rows", flush=True)
        quantize_one(
            work / "onnx-static" / f"{name}.onnx",
            out / "onnx" / f"{name}.onnx",
            ListReader(rows[key]),
        )

    shutil.copy2(src_root / "onnx" / "tts.json", out / "onnx" / "tts.json")
    shutil.copy2(src_root / "onnx" / "unicode_indexer.json", out / "onnx" / "unicode_indexer.json")
    for voice in ("M1", "M3", "M4", "M5", "F3", "F4", "F5"):
        shutil.copy2(src_root / "voice_styles" / f"{voice}.json", out / "voice_styles" / f"{voice}.json")
    shutil.copy2(src_root / "LICENSE", out / "LICENSE")

    manifest = {
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "text_length": TEXT_LEN,
        "latent_length": LATENT_LEN,
        "calibration_samples": len(SAMPLES) * len(VOICES),
        "voices": VOICES,
    }
    (out / "qnn-model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    shutil.rmtree(work, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    main()
