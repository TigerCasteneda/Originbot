# reading Video
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

h1 = 100
h2 = 500
DEFAULT_VIDEO_NAME = "赛道2.avi"
ONNX_MODEL_PATH = "./best_line_follower_model_xy.onnx"


def preprocess(image: np.ndarray) -> np.ndarray:
    # Match train-time preprocessing: BGR->RGB, resize, [0,1], normalize, NCHW.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def pred(image: np.ndarray, session: onnxruntime.InferenceSession, input_name: str, out_name: str):
    data = preprocess(image)
    pred_onx = session.run([out_name], {input_name: data})
    xx = int((pred_onx[0][0][0] + 1) * 960)
    yy = int((pred_onx[0][0][1] + 1) * 0.5 * (h2 - h1))
    return xx, yy


def main():
    parser = argparse.ArgumentParser(description="Run ONNX inference on a video.")
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO_NAME, help="Path to input video file")
    parser.add_argument("--onnx", default=ONNX_MODEL_PATH, help="Path to ONNX model file")
    args = parser.parse_args()

    session = onnxruntime.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    video_path = str(Path(args.video))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crop_image = frame[h1:h2, :, :].copy()
        x1, y1 = pred(crop_image, session, input_name, out_name)
        frame = cv2.circle(frame, (x1, y1 + h1), 20, (0, 0, 255), -1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(250) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
