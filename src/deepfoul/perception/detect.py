"""Person detection with rtmlib's YOLOX, keeping the confidence scores ByteTrack needs.

rtmlib's `YOLOX.__call__` returns boxes only and drops anything under 0.3. The bundled YOLOX
ONNX models have NMS built in (output `dets` is [1, N, 5] = x1, y1, x2, y2, score), so we run
rtmlib's own preprocess + inference and read `dets` ourselves.
"""

from __future__ import annotations

import numpy as np


def parse_dets(dets: np.ndarray, ratio: float, score_min: float,
               frame_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """[1, N, 5] model output -> (boxes Nx4 xyxy in frame pixels, scores N), score >= score_min."""
    dets = np.asarray(dets, dtype=np.float32)
    if dets.ndim != 3 or dets.shape[-1] != 5:
        raise ValueError(f"expected YOLOX output with built-in NMS, shape [1, N, 5]; got {dets.shape}")
    d = dets[0]
    d = d[d[:, 4] >= score_min]
    boxes = d[:, :4] / ratio
    h, w = frame_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[ok], d[ok, 4].copy()


class PersonDetector:
    """The profile's rtmlib YOLOX (same model file as Wholebody, so the cached onnx is reused)."""

    def __init__(self, pose_mode: str, device: str, score_min: float):
        from rtmlib import YOLOX, Wholebody

        spec = Wholebody.MODE[pose_mode]
        self.model = YOLOX(spec["det"], model_input_size=spec["det_input_size"],
                           backend="onnxruntime", device=device)
        self.score_min = score_min

    @property
    def providers(self) -> list[str]:
        return self.model.session.get_providers()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img, ratio = self.model.preprocess(image)
        outputs = self.model.inference(img)
        return parse_dets(outputs[0], ratio, self.score_min, image.shape[:2])
