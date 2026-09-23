import os
import json
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyttsx3
from flask import Flask, Response, jsonify, render_template, send_file

from insightface.app import FaceAnalysis


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# If you save this dashboard folder somewhere else, change
# PROJECT_ROOT below to the folder containing Data/ and models/.
YUNET_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2026may.onnx"
EMBEDDINGS_FILE = PROJECT_ROOT / "Data" / "Embeddings" / "embeddings_clean.npy"
METADATA_FILE = PROJECT_ROOT / "Data" / "Embeddings" / "metadata_clean.csv"
PERSON_DETAILS_FILE = BASE_DIR / "person_details.json"

CAMERA_INDEX = 0
THRESHOLD = 0.40
REPEAT_TIME = 30.0
MAX_TRACK_DISTANCE = 100
TRACK_LOST_TIMEOUT = 1.0

app_web = Flask(__name__)

# ============================================================
# SHARED RUNTIME STATE
# ============================================================

frame_lock = threading.Lock()
state_lock = threading.Lock()

latest_jpeg = None
latest_state = {
    "running": False,
    "camera": CAMERA_INDEX,
    "fps": 0.0,
    "faces_detected": 0,
    "recognised": 0,
    "unknown": 0,
    "threshold": THRESHOLD,
    "people": [],
    "error": None,
    "uptime": 0.0,
}

camera_thread = None
stop_event = threading.Event()

tracks = {}
next_track_id = 0
start_time = time.time()

# ============================================================
# TEXT TO SPEECH
# ============================================================

speech_queue = queue.Queue()
speech_thread = None


def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
    except Exception as exc:
        print("TTS initialization error:", exc)
        engine = None

    while True:
        name = speech_queue.get()
        try:
            if name is None:
                break
            if engine is not None:
                print(f"[VOICE] {name} detected")
                engine.say(f"{name} detected")
                engine.runAndWait()
        except Exception as exc:
            print("Speech error:", exc)
        finally:
            speech_queue.task_done()


def start_speech():
    global speech_thread
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()


def speak_name(name):
    if name != "UNKNOWN":
        # Avoid unbounded TTS backlog.
        if speech_queue.qsize() < 2:
            speech_queue.put(name)


# ============================================================
# PERSON DETAILS
# ============================================================

def load_person_details():
    if not PERSON_DETAILS_FILE.exists():
        return {}
    try:
        with PERSON_DETAILS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print("Could not load person_details.json:", exc)
        return {}


person_details = load_person_details()


def normalize_person_key(name):
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def get_person_profile(name):
    if name == "UNKNOWN":
        return {
            "name": "Unknown Person",
            "relation": "Not recognised",
            "status": "Unknown",
            "description": "This face was detected but did not meet the recognition threshold.",
            "image": None,
        }

    key = normalize_person_key(name)

    # Support both normalized keys and exact CSV-style names.
    profile = person_details.get(key) or person_details.get(name)

    if profile is None:
        for candidate_key, candidate_profile in person_details.items():
            if normalize_person_key(candidate_key) == key:
                profile = candidate_profile
                break

    profile = profile or {}

    return {
        "name": profile.get("name", name),
        "relation": profile.get("relation", "Not provided"),
        "status": profile.get("status", "Not provided"),
        "description": profile.get(
            "description",
            "No description has been added for this person yet.",
        ),
        "image": profile.get("image"),
    }


# ============================================================
# LOAD FACE DATABASE
# ============================================================

def load_database():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False).astype(np.float32)
    metadata = pd.read_csv(METADATA_FILE)

    if "person" not in metadata.columns:
        raise ValueError("metadata_clean.csv must contain a 'person' column.")

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs "
            f"{len(metadata)} metadata rows"
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    labels = metadata["person"].astype(str).to_numpy()

    print("=" * 60)
    print("FACE DATABASE")
    print("=" * 60)
    print("Embeddings shape:", embeddings.shape)
    print("Metadata shape  :", metadata.shape)
    print("Unique persons  :", metadata["person"].nunique())
    print("=" * 60)

    return embeddings, labels, metadata


# ============================================================
# TRACKING
# ============================================================

def create_track(center, name, similarity, bbox):
    global next_track_id

    track_id = next_track_id
    next_track_id += 1

    now = time.time()
    tracks[track_id] = {
        "center": center,
        "name": name,
        "similarity": similarity,
        "first_seen": now,
        "last_seen": now,
        "last_spoken": 0.0,
        "bbox": bbox,
    }
    return track_id


def find_matching_track(center, used_tracks):
    best_track_id = None
    best_distance = MAX_TRACK_DISTANCE
    cx, cy = center

    for track_id, track in tracks.items():
        if track_id in used_tracks:
            continue

        tx, ty = track["center"]
        distance = float(np.hypot(cx - tx, cy - ty))

        if distance < best_distance:
            best_distance = distance
            best_track_id = track_id

    return best_track_id


# ============================================================
# IMAGE HELPERS
# ============================================================

def resolve_person_image(image_value):
    if not image_value:
        return None

    candidate = Path(str(image_value))

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    try:
        candidate = candidate.resolve()
        root = PROJECT_ROOT.resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None

    return candidate if candidate.is_file() else None


def encode_frame(frame):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    return encoded.tobytes() if ok else None


# ============================================================
# RECOGNITION WORKER
# ============================================================

def camera_worker():
    global latest_jpeg, latest_state, tracks

    try:
        database_embeddings, database_labels, database_metadata = load_database()

        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}")

        print("Loading ArcFace...")
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        recognition_model = face_app.models["recognition"]

        print("Loading YuNet...")
        yunet = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            "",
            (320, 320),
            0.6,
            0.3,
            5000,
        )

        print("Opening camera...")
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not camera.isOpened():
            # Fallback for Linux/macOS or OpenCV builds without CAP_DSHOW.
            camera.release()
            camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            raise RuntimeError(
                f"Camera {CAMERA_INDEX} could not be opened. "
                "Try CAMERA_INDEX = 1 if another camera is available."
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret, test_frame = camera.read()
        if not ret or test_frame is None:
            camera.release()
            raise RuntimeError("Camera opened but could not read a frame.")

        print("Camera started:", test_frame.shape[1], "x", test_frame.shape[0])

        with state_lock:
            latest_state["running"] = True
            latest_state["error"] = None

        last_fps_time = time.time()
        fps_counter = 0
        displayed_fps = 0.0

        while not stop_event.is_set():
            ret, frame = camera.read()

            if not ret or frame is None:
                raise RuntimeError("Could not read a camera frame.")

            height, width = frame.shape[:2]
            yunet.setInputSize((width, height))

            _, detections = yunet.detect(frame)

            used_tracks = set()
            current_track_ids = set()
            current_people = []

            if detections is not None:
                for detection in detections:
                    x, y, w, h = detection[:4]

                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)

                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(width, x + w)
                    y2 = min(height, y + h)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    center = (
                        int((x1 + x2) / 2),
                        int((y1 + y2) / 2),
                    )

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    try:
                        embedding = recognition_model.get_feat(face_crop)
                        embedding = np.asarray(
                            embedding,
                            dtype=np.float32,
                        ).flatten()

                        norm = np.linalg.norm(embedding)
                        if norm == 0:
                            continue

                        embedding /= norm
                    except Exception as exc:
                        print("Embedding error:", exc)
                        continue

                    similarities = np.dot(database_embeddings, embedding)
                    best_index = int(np.argmax(similarities))
                    best_similarity = float(similarities[best_index])
                    predicted_name = str(database_labels[best_index])

                    name = (
                        predicted_name
                        if best_similarity >= THRESHOLD
                        else "UNKNOWN"
                    )

                    track_id = find_matching_track(center, used_tracks)

                    if track_id is None:
                        track_id = create_track(
                            center,
                            name,
                            best_similarity,
                            (x1, y1, x2, y2),
                        )
                        track = tracks[track_id]

                        if name != "UNKNOWN":
                            track["last_spoken"] = time.time()
                            speak_name(name)
                    else:
                        track = tracks[track_id]
                        track["center"] = center
                        track["bbox"] = (x1, y1, x2, y2)
                        track["last_seen"] = time.time()

                        if name != "UNKNOWN":
                            track["name"] = name
                            track["similarity"] = best_similarity

                            now = time.time()
                            if now - track["last_spoken"] >= REPEAT_TIME:
                                track["last_spoken"] = now
                                speak_name(track["name"])

                    used_tracks.add(track_id)
                    current_track_ids.add(track_id)

                    profile = get_person_profile(track["name"])

                    current_people.append({
                        "track_id": track_id,
                        "name": profile["name"],
                        "recognised_name": track["name"],
                        "similarity": round(track["similarity"], 4),
                        "similarity_percent": round(
                            max(0.0, min(1.0, track["similarity"])) * 100,
                            1,
                        ),
                        "relation": profile["relation"],
                        "status": profile["status"],
                        "description": profile["description"],
                        "image": profile["image"],
                        "bbox": [x1, y1, x2, y2],
                    })

                    # Camera overlay.
                    if track["name"] == "UNKNOWN":
                        label = f"UNKNOWN ({track['similarity']:.2f})"
                    else:
                        label = (
                            f"{profile['name']} "
                            f"({track['similarity']:.2f})"
                        )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - 42)),
                        (x2, y1),
                        (0, 0, 0),
                        -1,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, max(22, y1 - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, min(height - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Remove tracks that have disappeared.
            now = time.time()
            to_remove = [
                track_id
                for track_id, track in tracks.items()
                if now - track["last_seen"] > TRACK_LOST_TIMEOUT
            ]

            for track_id in to_remove:
                del tracks[track_id]

            # FPS.
            fps_counter += 1
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                displayed_fps = fps_counter / elapsed
                fps_counter = 0
                last_fps_time = now

            # Overlay status.
            recognised_count = sum(
                1 for p in current_people
                if p["recognised_name"] != "UNKNOWN"
            )
            unknown_count = len(current_people) - recognised_count

            cv2.putText(
                frame,
                f"FPS: {displayed_fps:.1f} | Faces: {len(current_people)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            encoded = encode_frame(frame)
            if encoded:
                with frame_lock:
                    latest_jpeg = encoded

            # Do not expose internal absolute image paths to browser.
            browser_people = []
            for person in current_people:
                item = dict(person)
                image_path = resolve_person_image(item.pop("image", None))
                item["has_image"] = image_path is not None
                browser_people.append(item)

            with state_lock:
                latest_state = {
                    "running": True,
                    "camera": CAMERA_INDEX,
                    "fps": round(displayed_fps, 1),
                    "faces_detected": len(current_people),
                    "recognised": recognised_count,
                    "unknown": unknown_count,
                    "threshold": THRESHOLD,
                    "people": browser_people,
                    "error": None,
                    "uptime": round(time.time() - start_time, 1),
                }

        camera.release()

    except Exception as exc:
        print("CAMERA WORKER ERROR:", exc)
        with state_lock:
            latest_state["running"] = False
            latest_state["error"] = str(exc)

    finally:
        with state_lock:
            if not latest_state.get("error"):
                latest_state["running"] = False


# ============================================================
# FLASK ROUTES
# ============================================================

@app_web.route("/")
def index():
    return render_template(
        "index.html",
        threshold=THRESHOLD,
    )


@app_web.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(latest_state)


@app_web.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = latest_jpeg

            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app_web.route("/person_image/<path:person_name>")
def person_image(person_name):
    profile = get_person_profile(person_name)
    image_path = resolve_person_image(profile.get("image"))

    if image_path is None:
        return ("", 404)

    return send_file(image_path)


@app_web.route("/api/project")
def project_info():
    return jsonify({
        "name": "Real-Time Face Recognition System",
        "detector": "YuNet",
        "recognizer": "InsightFace / ArcFace",
        "matching": "Cosine similarity",
        "threshold": THRESHOLD,
        "tracking": "Centroid-based temporal tracking",
        "voice": "pyttsx3",
        "database_records": int(len(database_metadata)),
        "unique_persons": int(database_metadata["person"].nunique()),
    })


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    start_speech()

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True,
    )
    camera_thread.start()

    print()
    print("=" * 60)
    print("FACE RECOGNITION WEB DASHBOARD")
    print("=" * 60)
    print("Open: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        app_web.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        stop_event.set()
        speech_queue.put(None)
