"""Step 1 - build the dataset.   python datacollect.py --roll BTECH10526 [--count 30]

Captures face crops from the webcam (Haar cascade) into dataset/<ROLL>/ .
Turn your head slightly between shots and vary the lighting for a robust model.
"""
import argparse
import re

import cv2

import config


def clean_roll(roll):
    roll = re.sub(r"[^A-Za-z0-9._-]", "", roll.strip()).upper()
    if not roll:
        raise ValueError("Roll number is empty after removing invalid characters.")
    return roll


def padded_crop(frame, box, pad_ratio=config.FACE_PADDING):
    x, y, w, h = box
    pad = int(pad_ratio * max(w, h))
    H, W = frame.shape[:2]
    return frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]


def collect(roll, count=config.IMAGES_PER_PERSON, camera=config.CAMERA_INDEX):
    roll = clean_roll(roll)
    out_dir = config.DATASET_DIR / roll
    out_dir.mkdir(parents=True, exist_ok=True)
    start = len(list(out_dir.glob("*.jpg")))

    cascade = cv2.CascadeClassifier(config.HAAR_FILE)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")

    saved = frame_no = 0
    try:
        while saved < count:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            view = frame.copy()
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if frame_no % config.CAPTURE_EVERY == 0:
                    saved += 1
                    cv2.imwrite(str(out_dir / f"{roll}_{start + saved:03d}.jpg"), padded_crop(frame, faces[0]))
            else:
                cv2.putText(view, "Exactly one face please", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(view, f"{roll}: {saved}/{count}  (q = quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Dataset collection", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} images to {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roll", help="roll number (asked interactively if omitted)")
    ap.add_argument("--count", type=int, default=config.IMAGES_PER_PERSON)
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    a = ap.parse_args()
    collect(a.roll or input("Enter Roll Number: "), a.count, a.camera)


if __name__ == "__main__":
    main()
