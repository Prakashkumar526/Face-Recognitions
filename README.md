# Facial Recognition Attendance System

Working implementation of the BIT Mesra minor-project report: **Haar-cascade face detection → face_recognition (dlib ResNet, 128-d embeddings) → movement-based anti-spoofing → Excel attendance.**

## Setup (Python 3.10+)
```bash
pip install -r requirements.txt      # dlib needs CMake; or use: pip install dlib-bin
```

## Usage
```bash
python main.py              # menu, or run the steps directly:

python datacollect.py --roll BTECH10526     # 1. capture ~30 face images per person (q = quit)
python encode_faces.py                      # 2. build models/encodings.pkl + evaluation report
python recognize.py                         # 3. live attendance (q = quit)
python view_attendance.py [--date DD-MM-YYYY | --all] [--csv out.csv]   # 4. view / export
```
Output: `output/attendance.xlsx` with columns **Roll Number | Time | Date** (one row per person per day).
Close the file in Excel while recognition is running, otherwise Windows blocks the save (the app retries).

### Optional CNN (report §4.3 / §5.1)
```bash
pip install tensorflow
python train_cnn.py                     # saves model + output/accuracy_vs_epochs.png
python recognize.py --backend cnn
```

## How it maps to the report
| Report section | File |
|---|---|
| 4.2 Dataset creation (roll-number labelled folders) | `datacollect.py` |
| 4.3 Training, evaluation (classification report, confusion matrix) | `encode_faces.py`, `train_cnn.py` |
| 3.1 / 4.4 Detection, recognition, attendance marking | `recognize.py`, `attendance.py` |
| 3.1 / 5.1 Anti-spoofing (movement threshold = 60) | `antispoof.py` |
| Parameters (tolerance 0.6, threshold 60, 128×128, 10 epochs) | `config.py` |

## Tests
```bash
python -m unittest discover -s tests -v
```

## Notes
- Anti-spoofing is movement-only, as in the report: a still photo/screen is rejected, but a photo moved by hand can pass. Add blink or depth checks for stronger protection.
- The 60 px threshold is tuned for a 640×480 webcam; change `SPOOF_THRESHOLD` / `SPOOF_WINDOW_SEC` in `config.py`.
- Face data is personal data: get consent and keep `dataset/` and `models/` private.
