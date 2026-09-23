# Face Recognition Web Dashboard

This is a web interface around the supplied YuNet + InsightFace/ArcFace face-recognition pipeline.

## Project structure

Place this folder beside your existing `Data/` and `models/` folders:

```text
YOUR_PROJECT/
├── Data/
│   └── Embeddings/
│       ├── embeddings_clean.npy
│       └── metadata_clean.csv
├── models/
│   └── face_detection_yunet_2026may.onnx
└── face_recognition_complete/
    ├── app.py
    ├── person_details.json
    ├── requirements.txt
    ├── README.md
    ├── templates/
    │   └── index.html
    └── static/
        ├── app.js
        └── style.css
```

The generated `person_details.json` contains the 14 persons found in your metadata file.

## 1. Install dependencies

Recommended: create a virtual environment.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r face_recognition_complete\requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r face_recognition_complete/requirements.txt
```

## 2. Edit person details

Open:

```text
face_recognition_complete/person_details.json
```

Each person has:

- `name`
- `relation`
- `status`
- `description`
- `image`

Example:

```json
"person_name": {
  "name": "Person Name",
  "relation": "Friend",
  "status": "Good",
  "description": "Short description.",
  "image": "Data/Classmates_Aligned/person_name/photo.jpg"
}
```

Use paths relative to the project root for `image`.

## 3. Run

From the directory containing `Data/`, `models/`, and `face_recognition_complete/`:

```bash
python face_recognition_complete/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 4. Camera

Default:

```python
CAMERA_INDEX = 0
```

If the wrong camera opens, change it to `1`, `2`, etc.

## 5. Recognition threshold

Default:

```python
THRESHOLD = 0.40
```

This is the same baseline threshold as the supplied pipeline. Increasing it generally makes recognition stricter; decreasing it generally makes it more permissive.

## 6. Architecture

```text
Web Browser
     |
     | HTTP / MJPEG
     v
Flask Dashboard
     |
     v
Camera Worker
     |
     +--> YuNet face detection
     |
     +--> ArcFace embedding
     |
     +--> cosine similarity
     |
     +--> identity decision
     |
     +--> temporal tracking
     |
     +--> person_details.json
     |
     +--> pyttsx3 voice notification
     v
Live frame + JSON status
```

## Notes

- The dashboard does not replace your stored embeddings.
- It reads the existing `embeddings_clean.npy` and `metadata_clean.csv`.
- It reads additional profile information from `person_details.json`.
- Unknown faces are displayed separately.
- Multiple faces can be displayed simultaneously.
- The browser dashboard does not require `cv2.imshow()`.
