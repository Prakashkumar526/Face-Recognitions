"""Step 3 - live recognition + attendance.   python recognize.py [--backend dlib|cnn]

Pipeline (report flowchart): webcam -> Haar face detection -> recognition ->
anti-spoofing (movement check) -> Excel attendance. Press q to quit.
"""
import argparse
import json
import pickle

import cv2
import numpy as np

import config
from antispoof import MovementVerifier
from attendance import AttendanceLog
from datacollect import padded_crop

GREEN, ORANGE, RED = (0, 200, 0), (0, 165, 255), (0, 0, 255)


class DlibIdentifier:
    """128-d face_recognition embeddings, nearest match under MATCH_TOLERANCE."""

    def __init__(self):
        if not config.ENCODINGS_FILE.exists():
            raise SystemExit("No model found - run encode_faces.py first.")
        with open(config.ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        self.enc, self.labels = np.asarray(data["encodings"]), data["labels"]

    def identify(self, frame, box):
        import face_recognition
        s = config.PROCESS_SCALE
        small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=s, fy=s), cv2.COLOR_BGR2RGB)
        x, y, w, h = (int(v * s) for v in box)
        enc = face_recognition.face_encodings(small, [(y, x + w, y + h, x)])
        if not enc:
            return None
        d = face_recognition.face_distance(self.enc, enc[0])
        i = int(np.argmin(d))
        return self.labels[i] if d[i] <= config.MATCH_TOLERANCE else None


class CnnIdentifier:
    """Keras CNN from train_cnn.py; accepts predictions above CNN_MIN_CONFIDENCE."""

    def __init__(self):
        if not config.CNN_MODEL_FILE.exists():
            raise SystemExit("No CNN model found - run train_cnn.py first.")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(config.CNN_MODEL_FILE)
        self.classes = json.loads(config.CNN_CLASSES_FILE.read_text())

    def identify(self, frame, box):
        crop = cv2.cvtColor(padded_crop(frame, box), cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, config.CNN_IMG_SIZE)[None].astype("float32")
        p = self.model.predict(crop, verbose=0)[0]
        return self.classes[int(p.argmax())] if p.max() >= config.CNN_MIN_CONFIDENCE else None


def run(source=config.CAMERA_INDEX, backend="dlib", gui=True, max_frames=None):
    ident = CnnIdentifier() if backend == "cnn" else DlibIdentifier()
    cascade = cv2.CascadeClassifier(config.HAAR_FILE)
    verifier, log = MovementVerifier(), AttendanceLog()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source {source!r}")

    s, marked, n = config.PROCESS_SCALE, [], 0
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            small = cv2.resize(frame, (0, 0), fx=s, fy=s)
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

            for fx, fy, fw, fh in faces:
                box = (int(fx / s), int(fy / s), int(fw / s), int(fh / s))
                x, y, w, h = box
                roll = ident.identify(frame, box)
                if roll is None:
                    color, text = RED, "Unknown"
                elif verifier.update(roll, box):             # live face confirmed
                    if log.mark(roll):
                        marked.append(roll)
                        print(f"[+] Attendance marked: {roll}")
                    color, text = GREEN, f"{roll} PRESENT" if log.already_marked(roll) else roll
                else:
                    color, text = ORANGE, f"{roll} - move slightly to verify"
                if gui:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if gui:
                cv2.imshow("Attendance (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if gui:
            cv2.destroyAllWindows()
    print(f"Done. Newly marked this session: {sorted(set(marked)) or 'none'}")
    return marked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["dlib", "cnn"], default="dlib")
    ap.add_argument("--source", default=str(config.CAMERA_INDEX), help="camera index or video file path")
    ap.add_argument("--no-gui", action="store_true", help="run without a window (servers / tests)")
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    run(src, a.backend, gui=not a.no_gui)


if __name__ == "__main__":
    main()
