import cv2
import numpy as np
import pyttsx3
import time
import threading
import queue
import pandas as pd

from insightface.app import FaceAnalysis


# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)


# ============================================================
# SPEECH QUEUE
# ============================================================

speech_queue = queue.Queue()


def speech_worker():

    while True:

        name = speech_queue.get()

        if name is None:
            speech_queue.task_done()
            break

        try:

            print(f"[VOICE] {name} detected")

            engine.say(f"{name} detected")
            engine.runAndWait()

        except Exception as e:

            print("Speech error:", e)

        finally:

            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


def speak_name(name):

    if name == "UNKNOWN":
        return

    speech_queue.put(name)


# ============================================================
# PATHS
# ============================================================

YUNET_MODEL = "models/face_detection_yunet_2026may.onnx"

EMBEDDINGS_FILE = "./Data/Embeddings/embeddings_clean.npy"

METADATA_FILE = "./Data/Embeddings/metadata_clean.csv"

# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0


# ============================================================
# ARCFACE RECOGNITION THRESHOLD
# ============================================================

THRESHOLD = 0.40


# ============================================================
# SPEECH SETTINGS
# ============================================================

# Speak immediately when a person appears.
# Then repeat after this many seconds if continuously visible.

REPEAT_TIME = 30.0


# ============================================================
# PERSON PRESENCE SETTINGS
# ============================================================

# Maximum distance for matching the same face
# between consecutive frames.

MAX_TRACK_DISTANCE = 100


# If a face disappears for more than this time,
# consider that person to have left.

TRACK_LOST_TIMEOUT = 1.0


# ============================================================
# LOAD STORED EMBEDDINGS
# ============================================================

print()
print("==============================================")
print("FACE DATABASE")
print("==============================================")


data = np.load(
    EMBEDDINGS_FILE,
    allow_pickle=False
)

database_embeddings = data.astype(np.float32)

print("Embeddings shape:", database_embeddings.shape)



# ============================================================
# LOAD FACE DATABASE
# ============================================================

database_embeddings = np.load(
    EMBEDDINGS_FILE,
    allow_pickle=False
).astype(np.float32)


database_metadata = pd.read_csv(
    METADATA_FILE
)

print("\n==============================================")
print("FACE DATABASE")
print("==============================================")

print("Embeddings shape:", database_embeddings.shape)
print("Metadata shape  :", database_metadata.shape)


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

if len(database_embeddings) != len(database_metadata):
    raise ValueError(
        f"Mismatch: "
        f"{len(database_embeddings)} embeddings vs "
        f"{len(database_metadata)} metadata rows"
    )


# ------------------------------------------------------------
# Person labels come from metadata CSV
# ------------------------------------------------------------

database_labels = database_metadata["person"].astype(str).to_numpy()

print("Database labels:", len(database_labels))
print("Unique persons :", database_metadata["person"].nunique())

print("\nPersons:")
print(database_metadata["person"].value_counts())


print("Embeddings:", database_embeddings.shape)

print("Labels:", len(database_labels))


# ============================================================
# NORMALIZE DATABASE EMBEDDINGS
# ============================================================

database_norms = np.linalg.norm(
    database_embeddings,
    axis=1,
    keepdims=True
)

database_embeddings = (
    database_embeddings /
    np.maximum(database_norms, 1e-10)
)


# ============================================================
# LOAD ARCFACE
# ============================================================

print()
print("Loading ArcFace...")


app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)


app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


recognition_model = app.models["recognition"]


print("ArcFace recognition model loaded.")


# ============================================================
# LOAD YUNET
# ============================================================

print()
print("Loading YuNet...")


yunet = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


print("YuNet model loaded.")


# ============================================================
# CAMERA
# ============================================================

print()
print("Opening camera...")


camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    speech_queue.put(None)

    raise SystemExit(1)


# ============================================================
# CAMERA SETTINGS
# ============================================================

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# TEST FIRST FRAME
# ============================================================

ret, test_frame = camera.read()


if not ret or test_frame is None:

    print()
    print("ERROR: Camera opened but could not read a frame.")
    print()

    print("Possible causes:")
    print("1. Another application is using the camera.")
    print("2. Windows camera permission is disabled.")
    print("3. Wrong camera index.")
    print("4. Camera driver problem.")
    print()

    camera.release()

    speech_queue.put(None)

    raise SystemExit(1)


print()
print("==============================================")
print("CAMERA STARTED")
print("==============================================")

print(
    "Camera resolution:",
    test_frame.shape[1],
    "x",
    test_frame.shape[0]
)

print()
print("Speech behavior:")
print(" - New person -> speak immediately")
print(" - Same person continuously visible -> repeat every 30 seconds")
print(" - Person leaves and returns -> speak immediately again")
print()
print("Press Q to quit.")
print()


# ============================================================
# TRACKING DATA
# ============================================================

tracks = {}


# ============================================================
# NEXT TRACK ID
# ============================================================

next_track_id = 0


# ============================================================
# CREATE NEW TRACK
# ============================================================

def create_track(
    center,
    name,
    similarity,
    bbox
):

    global next_track_id

    track_id = next_track_id

    next_track_id += 1

    current_time = time.time()

    tracks[track_id] = {

        "center": center,

        "name": name,

        "similarity": similarity,

        # When this person first appeared
        "first_seen": current_time,

        # Last frame in which this person was detected
        "last_seen": current_time,

        # Last time we spoke this person's name
        "last_spoken": 0.0,

        "bbox": bbox

    }

    return track_id


# ============================================================
# FIND CLOSEST EXISTING TRACK
# ============================================================

def find_matching_track(
    center,
    used_tracks
):

    best_track_id = None

    best_distance = MAX_TRACK_DISTANCE

    cx, cy = center

    for track_id, track in tracks.items():

        # Do not use one track for two faces
        if track_id in used_tracks:
            continue

        tx, ty = track["center"]

        distance = np.sqrt(
            (cx - tx) ** 2 +
            (cy - ty) ** 2
        )

        if distance < best_distance:

            best_distance = distance

            best_track_id = track_id

    return best_track_id


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    # ========================================================
    # READ CAMERA FRAME
    # ========================================================

    ret, frame = camera.read()


    if not ret or frame is None:

        print("ERROR: Could not read frame.")

        break


    # ========================================================
    # FRAME SIZE
    # ========================================================

    height, width = frame.shape[:2]


    # ========================================================
    # TELL YUNET CURRENT FRAME SIZE
    # ========================================================

    yunet.setInputSize(
        (width, height)
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    _, detections = yunet.detect(frame)


    # ========================================================
    # TRACKS USED IN THIS FRAME
    # ========================================================

    used_tracks = set()


    # ========================================================
    # CURRENT TRACK IDS
    # ========================================================

    current_track_ids = set()


    # ========================================================
    # PROCESS DETECTED FACES
    # ========================================================

    if detections is not None:

        for detection in detections:

            # =================================================
            # BOUNDING BOX
            # =================================================

            x, y, w, h = detection[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)


            # =================================================
            # KEEP BOUNDING BOX INSIDE IMAGE
            # =================================================

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(width, x + w)
            y2 = min(height, y + h)


            if x2 <= x1 or y2 <= y1:

                continue


            # =================================================
            # FACE CENTER
            # =================================================

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            center = (
                center_x,
                center_y
            )


            # =================================================
            # FACE CROP
            # =================================================

            face_crop = frame[
                y1:y2,
                x1:x2
            ]


            if face_crop.size == 0:

                continue


            # =================================================
            # ARCFACE EMBEDDING
            # =================================================

            try:

                embedding = recognition_model.get_feat(
                    face_crop
                )


                embedding = np.asarray(
                    embedding,
                    dtype=np.float32
                ).flatten()


                # =============================================
                # NORMALIZE EMBEDDING
                # =============================================

                norm = np.linalg.norm(
                    embedding
                )


                if norm == 0:

                    continue


                embedding = (
                    embedding /
                    norm
                )


            except Exception as e:

                print(
                    "Embedding error:",
                    e
                )

                continue


            # =================================================
            # COSINE SIMILARITY
            # =================================================

            similarities = np.dot(
                database_embeddings,
                embedding
            )


            best_index = np.argmax(
                similarities
            )


            best_similarity = float(
                similarities[best_index]
            )


            predicted_name = database_labels[
                best_index
            ]


            # =================================================
            # RECOGNITION DECISION
            # =================================================

            if best_similarity >= THRESHOLD:

                name = str(
                    predicted_name
                )

            else:

                name = "UNKNOWN"


            # =================================================
            # FIND EXISTING TRACK
            # =================================================

            track_id = find_matching_track(
                center,
                used_tracks
            )


            # =================================================
            # NEW TRACK / NEW PERSON
            # =================================================

            if track_id is None:

                track_id = create_track(
                    center,
                    name,
                    best_similarity,
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )


                used_tracks.add(
                    track_id
                )


                current_track_ids.add(
                    track_id
                )


                track = tracks[
                    track_id
                ]


                # =============================================
                # NEW PERSON
                # =============================================

                if name != "UNKNOWN":

                    print()
                    print(
                        "================================"
                    )

                    print(
                        f"NEW PERSON DETECTED: {name}"
                    )

                    print(
                        f"Similarity: {best_similarity:.4f}"
                    )

                    print(
                        "Speaking name..."
                    )

                    print(
                        "================================"
                    )


                    # =========================================
                    # SPEAK IMMEDIATELY
                    # =========================================

                    track["last_spoken"] = time.time()

                    speak_name(name)


            # =================================================
            # EXISTING TRACK
            # =================================================

            else:

                used_tracks.add(
                    track_id
                )


                current_track_ids.add(
                    track_id
                )


                track = tracks[
                    track_id
                ]


                # =============================================
                # UPDATE POSITION
                # =============================================

                track["center"] = center

                track["bbox"] = (
                    x1,
                    y1,
                    x2,
                    y2
                )


                track["last_seen"] = time.time()


                # =============================================
                # UPDATE RECOGNITION
                # =============================================

                if name != "UNKNOWN":

                    track["name"] = name

                    track["similarity"] = (
                        best_similarity
                    )


                # =============================================
                # 30 SECOND REPEAT
                # =============================================

                current_time = time.time()


                if (
                    track["name"] != "UNKNOWN"
                    and
                    current_time -
                    track["last_spoken"]
                    >= REPEAT_TIME
                ):

                    print()
                    print(
                        "--------------------------------"
                    )

                    print(
                        f"30 SECOND REPEAT: "
                        f"{track['name']}"
                    )

                    print(
                        "Speaking name..."
                    )

                    print(
                        "--------------------------------"
                    )


                    # Update BEFORE putting into queue
                    # so another frame cannot queue
                    # the same person repeatedly.

                    track["last_spoken"] = current_time


                    speak_name(
                        track["name"]
                    )


            # =================================================
            # DISPLAY INFORMATION
            # =================================================

            track = tracks[
                track_id
            ]


            display_name = track[
                "name"
            ]


            display_similarity = track[
                "similarity"
            ]


            # =================================================
            # LABEL
            # =================================================

            if display_name == "UNKNOWN":

                label = (
                    f"UNKNOWN "
                    f"({display_similarity:.2f})"
                )

            else:

                label = (
                    f"{display_name} "
                    f"({display_similarity:.2f})"
                )


            # =================================================
            # DRAW RECTANGLE
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # DRAW NAME
            # =================================================

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        30
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # =================================================
            # DRAW TRACKING ID
            # =================================================

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (
                    x1,
                    min(
                        y2 + 25,
                        height - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


    # ========================================================
    # REMOVE LOST TRACKS
    # ========================================================

    current_time = time.time()

    tracks_to_remove = []


    for track_id, track in tracks.items():

        time_since_seen = (
            current_time -
            track["last_seen"]
        )


        if time_since_seen > TRACK_LOST_TIMEOUT:

            tracks_to_remove.append(
                track_id
            )


    # ========================================================
    # DELETE LOST TRACKS
    # ========================================================

    for track_id in tracks_to_remove:

        old_name = tracks[
            track_id
        ]["name"]


        print(
            f"Person left frame: "
            f"{old_name}"
        )


        del tracks[
            track_id
        ]


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "YuNet + ArcFace Real-Time Face Recognition",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


# ============================================================
# STOP SPEECH WORKER
# ============================================================

speech_queue.put(None)

speech_thread.join(
    timeout=2
)


print()
print("Camera stopped.")
print("Program finished.")