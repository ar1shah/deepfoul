"""RTMW whole-body pose (133 COCO-WholeBody keypoints) on the kept person boxes.

Uses rtmlib's RTMPose with the same model file `Wholebody` uses for the profile's `pose_mode`.
rtmlib runs one box per ONNX call; when the model's batch dimension is dynamic we stack every
box of a frame into one call instead (same preprocess/postprocess, checked equal in tests by
construction: each crop is processed independently).

Pose runs once per kept box in the tracking pass. 1.3's identity reads it, and 1.4's extract
writes the same arrays to data/keypoints, so RTMW never runs twice on a box.
"""

from __future__ import annotations

import numpy as np

N_KPT = 133


class PoseEstimator:
    def __init__(self, pose_mode: str, device: str, batch: bool = True):
        from rtmlib import RTMPose, Wholebody

        spec = Wholebody.MODE[pose_mode]
        self.model = RTMPose(spec["pose"], model_input_size=spec["pose_input_size"],
                             backend="onnxruntime", device=device)
        shape = self.model.session.get_inputs()[0].shape
        self.batched = batch and not isinstance(shape[0], int)

    @property
    def providers(self) -> list[str]:
        return self.model.session.get_providers()

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """boxes Nx4 xyxy -> (keypoints N x 133 x 2 in frame pixels, scores N x 133)."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, N_KPT, 2), np.float32), np.zeros((0, N_KPT), np.float32)
        m = self.model
        if not self.batched:
            k, s = m(image, bboxes=list(boxes))
            return k.astype(np.float32), s.astype(np.float32)
        crops, cs = [], []
        for b in boxes:
            img, center, scale = m.preprocess(image, b)
            crops.append(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32))
            cs.append((center, scale))
        sess = m.session
        outs = sess.run([o.name for o in sess.get_outputs()],
                        {sess.get_inputs()[0].name: np.stack(crops)})
        kpts, scores = [], []
        for i, (center, scale) in enumerate(cs):
            k, s = m.postprocess([o[i:i + 1] for o in outs], center, scale)
            kpts.append(k[0])
            scores.append(s[0])
        return np.asarray(kpts, np.float32), np.asarray(scores, np.float32)
