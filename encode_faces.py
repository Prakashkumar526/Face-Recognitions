"""Step 2 - train / encode.   python encode_faces.py

Converts every dataset image into a 128-d embedding (dlib ResNet via the
face_recognition library), saves models/encodings.pkl, and evaluates the
matcher on a held-out 20 % split (classification report + confusion matrix).
"""
import pickle

import numpy as np

import config


def load_dataset(dataset_dir=config.DATASET_DIR):
    import face_recognition
    encodings, labels = [], []
    for person in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        n_ok = 0
        for img_path in sorted(person.glob("*.[jp][pn]g")):
            img = face_recognition.load_image_file(str(img_path))
            locs = face_recognition.face_locations(img)
            if not locs:                       # images are already face crops -> use the whole image
                h, w = img.shape[:2]
                locs = [(0, w, h, 0)]
            loc = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))   # largest face
            enc = face_recognition.face_encodings(img, [loc])
            if enc:
                encodings.append(enc[0])
                labels.append(person.name)
                n_ok += 1
        print(f"  {person.name}: {n_ok} usable images")
    return np.array(encodings), labels


def predict(train_enc, train_lab, enc, tolerance=config.MATCH_TOLERANCE):
    d = np.linalg.norm(train_enc - enc, axis=1)
    i = int(np.argmin(d))
    return train_lab[i] if d[i] <= tolerance else "Unknown"


def evaluate(encodings, labels):
    from collections import Counter
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        print("Skipping evaluation (need >= 2 people with >= 2 images each).")
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(encodings, labels, test_size=0.2,
                                              stratify=labels, random_state=42)
    pred = [predict(X_tr, y_tr, e) for e in X_te]
    names = sorted(counts)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, labels=names, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=names)
    print(f"\nHeld-out accuracy: {acc:.2%}\n{report}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "evaluation.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}\n{cm}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 1.2 * len(names) + 2))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    return acc


def main():
    if not config.DATASET_DIR.exists() or not any(config.DATASET_DIR.iterdir()):
        raise SystemExit("Dataset is empty - run datacollect.py first.")
    print("Encoding faces ...")
    encodings, labels = load_dataset()
    if len(encodings) == 0:
        raise SystemExit("No usable face images found.")
    evaluate(encodings, labels)
    config.MODEL_DIR.mkdir(exist_ok=True)
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "labels": labels}, f)
    print(f"\nSaved {len(labels)} encodings for {len(set(labels))} people -> {config.ENCODINGS_FILE}")


if __name__ == "__main__":
    main()
