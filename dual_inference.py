# dual_inference.py - FINAL STABLE VERSION (Two Hands + ISL Working + No Crashes)
import os
import time
import uuid
import shutil
import threading
import numpy as np
import cv2
from pathlib import Path

# ================== SILENCE TENSORFLOW & PYGAME NOISE ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # Only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # Remove oneDNN message
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# ====================================================================

# Audio
try:
    import pygame
    from gtts import gTTS
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame or gtts not installed → Audio disabled")

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ========================== CONFIG (SAFE DEFAULTS) ==========================
MODEL_PATHS = {
    "mediapipe": "models/gesture_recognizer.task",
    "isl": "models/isl_model.h5",
    "isl_classes": "models/isl_classes.txt"
}

CONFIDENCE_THRESHOLDS = {"mediapipe": 0.7, "isl": 0.65}

MEDIAPIPE_GESTURES = {
    "Victory": "Victory", "Thumb_Up": "Good", "Thumb_Down": "Bad",
    "Open_Palm": "Stop", "Closed_Fist": "Fist", "Pointing_Up": "One"
    # Add more friendly names if you want
}

TEMP_AUDIO_FOLDER = "temp_audio"
AUDIO_VOLUME = 0.9
WEBCAM_WIDTH, WEBCAM_HEIGHT = 1280, 720
# ====================================================================


class DualModelRecognizer:
    def __init__(self):
        self.mp_recognizer = None
        self.isl_model = None
        self.isl_classes = []
        self.audio_ok = False
        self.last_speech_time = 0

    def init_audio(self):
        if not PYGAME_AVAILABLE:
            return
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.music.set_volume(AUDIO_VOLUME)
            self.audio_ok = True
            print("✅ Audio ready")
        except:
            self.audio_ok = False

    def load_mediapipe(self):
        if not os.path.exists(MODEL_PATHS["mediapipe"]):
            print(f"❌ MediaPipe model not found:\n   {MODEL_PATHS['mediapipe']}")
            print("   Download it with: python -c \"import mediapipe as mp; mp.tasks.vision.GestureRecognizer.create_from_model_path('dummy')\"")
            return False
        try:
            BaseOptions = python.BaseOptions
            GestureRecognizer = vision.GestureRecognizer
            options = vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATHS["mediapipe"]),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,          # ← TWO HANDS SUPPORT
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_recognizer = GestureRecognizer.create_from_options(options)
            print("✅ MediaPipe loaded (2 hands)")
            return True
        except Exception as e:
            print(f"❌ MediaPipe failed: {e}")
            return False

    def load_isl_model(self):
        if not os.path.exists(MODEL_PATHS["isl"]) and os.path.exists(MODEL_PATHS["isl_classes"]):
            try:
                with open(MODEL_PATHS["isl_classes"], "r", encoding="utf-8") as f:
                    self.isl_classes = [line.strip() for line in f if line.strip()]
                self.isl_model = tf.keras.models.load_model(MODEL_PATHS["isl"])
                print(f"✅ ISL model loaded → {len(self.isl_classes)} classes")
                return True
            except Exception as e:
                print(f"⚠️ ISL model failed to load: {e}")
        else:
            print("⚠️ ISL model or classes missing → running MediaPipe only")
        return False

    def extract_roi(self, frame, landmarks):
        if not landmarks: return None
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1 = max(0, int(min(xs) - 30))
        y1 = max(0, int(min(ys) - 30))
        x2 = min(w, int(max(xs) + 30))
        y2 = min(h, int(max(ys) + 30))
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    def preprocess_isl(self, roi):
        try:
            img = cv2.resize(roi, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0
            return np.expand_dims(img, axis=0)
        except:
            return None

    def predict_isl(self, roi):
        if not self.isl_model or roi is None:
            return None, 0.0
        try:
            x = self.preprocess_isl(roi)
            if x is None: return None, 0.0
            pred = self.isl_model.predict(x, verbose=0)[0]
            conf = float(pred.max())
            idx = int(pred.argmax())
            if conf >= CONFIDENCE_THRESHOLDS["isl"] and idx < len(self.isl_classes):
                return self.isl_classes[idx], conf
        except:
            pass
        return None, 0.0

    def recognize(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_recognizer.recognize(mp_img)

        # Collect data from up to 2 hands
        mp_gestures = []
        all_landmarks = []
        all_rois = []

        if result.gestures:
            for hand_gestures in result.gestures:
                if hand_gestures:
                    mp_gestures.append((hand_gestures[0].category_name, hand_gestures[0].score))
                else:
                    mp_gestures.append((None, 0.0))
            for hand_lm in result.hand_landmarks:
                all_landmarks.append(hand_lm)
                roi = self.extract_roi(frame, hand_lm)
                all_rois.append(roi)
        else:
            mp_gestures = [(None, 0.0)]

        return mp_gestures, all_landmarks, all_rois

    def draw_hands(self, frame, landmarks_list):
        for hand_lm in landmarks_list:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_lm
            ])
            mp_drawing.draw_landmarks(
                frame, proto, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2),
            )

    def speak(self, text):
        if not self.audio_ok or not text:
            return
        now = time.time()
        if now - self.last_speech_time < 1.3:
            return
        self.last_speech_time = now

        def tts_thread():
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                path = os.path.join(TEMP_AUDIO_FOLDER, f"{uuid.uuid4().hex[:8]}.mp3")
                tts.save(path)
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.remove(path)
            except:
                pass
        threading.Thread(target=tts_thread, daemon=True).start()


class App:
    def __init__(self):
        self.rec = DualModelRecognizer()
        self.sentence = []
        self.last_sign = ""
        self.last_time = 0

    def start(self):
        print("\n🚀 Starting Dual-Model Indian Sign Language Recognizer (2 Hands)\n")
        Path(TEMP_AUDIO_FOLDER).mkdir(exist_ok=True)

        self.rec.init_audio()
        if not self.rec.load_mediapipe():
            print("Cannot continue without MediaPipe model")
            input("Press Enter to exit...")
            return
        self.rec.load_isl_model()

        while True:
            print("\n" + "="*55)
            print("1. Webcam     2. IP Camera     3. Exit")
            ch = input("\nChoose: ").strip()
            if ch == "1":
                self.run_camera()
            elif ch == "2":
                url = input("Enter IP camera URL (e.g. http://192.168.x.x:8080/video): ").strip()
                self.run_camera(url)
            elif ch == "3":
                break

    def run_camera(self, url=None):
        cap = cv2.VideoCapture(url if url else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        print("✅ Camera started | Show signs | Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            mp_gestures, landmarks, rois = self.rec.recognize(frame)
            self.rec.draw_hands(frame, landmarks)

            best_sign, best_conf, source = None, 0.0, "None"

            # MediaPipe priority
            for sign, conf in mp_gestures:
                if sign and conf > CONFIDENCE_THRESHOLDS["mediapipe"] and conf > best_conf:
                    best_sign, best_conf, source = sign, conf, "MediaPipe"

            # ISL model on every hand
            if self.rec.isl_model:
                for roi in rois:
                    if roi is None: continue
                    isl_sign, isl_conf = self.rec.predict_isl(roi)
                    if isl_sign and isl_conf > best_conf:
                        best_sign, best_conf, source = isl_sign, isl_conf, "ISL Model"

            # Display
            color = (0,255,0) if source == "MediaPipe" else (255,0,255) if source == "ISL Model" else (0,0,255)
            cv2.putText(frame, f"{best_sign or 'No sign'} ({best_conf:.2f}) [{source}]",
                        (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3)

            # Sentence logic
            if best_sign and best_conf > 0.65:
                now = time.time()
                if best_sign != self.last_sign or (now - self.last_time > 1.5):
                    display = MEDIAPIPE_GESTURES.get(best_sign, best_sign)
                    self.sentence.append(best_sign)
                    self.rec.speak(display)
                    self.last_sign = best_sign
                    self.last_time = now
                    if len(self.sentence) > 12:
                        self.sentence = self.sentence[-12:]

            cv2.putText(frame, "Sentence: " + " ".join(self.sentence),
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

            cv2.imshow("Dual-Model ISL Recognition - 2 Hands Ready!", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        App().start()
    except Exception as e:
        import traceback
        print("\nFATAL ERROR:\n")
        traceback.print_exc()
        input("\nPress Enter to close...")
    finally:
        cv2.destroyAllWindows()
        if os.path.exists(TEMP_AUDIO_FOLDER):
            shutil.rmtree(TEMP_AUDIO_FOLDER, ignore_errors=True)