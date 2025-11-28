# dual_inference.py - COMPLETE Dual-Model ISL Recognition System
import cv2
import os
import uuid
import numpy as np
import shutil
import time
import threading
import sys
from pathlib import Path

# Audio imports
import pygame
from gtts import gTTS

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# TensorFlow imports for ISL model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    print("❌ TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

# Configuration
try:
    from config import (
        MEDIAPIPE_GESTURES, ISL_GESTURES, MODEL_PATHS, 
        CONFIDENCE_THRESHOLDS, TEMP_AUDIO_FOLDER, AUDIO_VOLUME,
        WEBCAM_WIDTH, WEBCAM_HEIGHT, COMMON_IP_CAMERAS
    )
except ImportError as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class DualModelGestureRecognizer:
    """Dual-model system combining MediaPipe and custom ISL model"""
    
    def __init__(self):
        self.mediapipe_recognizer = None
        self.isl_model = None
        self.isl_classes = []
        self.audio_system = None
        self.audio_available = False
        
    def initialize_audio(self):
        """Initialize audio system"""
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            pygame.mixer.music.set_volume(AUDIO_VOLUME)
            self.audio_available = True
            print("✅ Audio system initialized")
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            self.audio_available = False
    
    def load_mediapipe_model(self):
        """Load MediaPipe gesture recognizer"""
        if not os.path.exists(MODEL_PATHS["mediapipe"]):
            print(f"❌ MediaPipe model not found: {MODEL_PATHS['mediapipe']}")
            print("   Run: python download_models.py")
            return False
            
        try:
            BaseOptions = python.BaseOptions
            GestureRecognizer = vision.GestureRecognizer
            GestureRecognizerOptions = vision.GestureRecognizerOptions
            RunningMode = vision.RunningMode

            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATHS["mediapipe"]),
                running_mode=RunningMode.IMAGE,
                num_hands=1
            )
            self.mediapipe_recognizer = GestureRecognizer.create_from_options(options)
            print("✅ MediaPipe model loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load MediaPipe model: {e}")
            return False
    
    def load_isl_model(self):
        """Load custom ISL model"""
        if not os.path.exists(MODEL_PATHS["isl"]):
            print(f"⚠ ISL model not found: {MODEL_PATHS['isl']}")
            print("   This model will be skipped. Using MediaPipe only.")
            return False
            
        if not os.path.exists(MODEL_PATHS["isl_classes"]):
            print(f"❌ ISL classes file not found: {MODEL_PATHS['isl_classes']}")
            return False
            
        try:
            # Load class names
            with open(MODEL_PATHS["isl_classes"], 'r') as f:
                self.isl_classes = [line.strip() for line in f.readlines()]
            
            # Load model
            self.isl_model = load_model(MODEL_PATHS["isl"])
            print(f"✅ ISL model loaded - {len(self.isl_classes)} classes")
            return True
        except Exception as e:
            print(f"❌ Failed to load ISL model: {e}")
            return False
    
    def preprocess_for_isl(self, hand_roi):
        """Preprocess hand ROI for ISL model"""
        try:
            # Resize to 64x64 (common size for ISL models)
            resized = cv2.resize(hand_roi, (64, 64))
            # Convert to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Normalize
            normalized = rgb.astype('float32') / 255.0
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            return batched
        except Exception as e:
            print(f"ISL preprocessing error: {e}")
            return None
    
    def extract_hand_roi(self, frame, landmarks):
        """Extract hand region from landmarks"""
        try:
            h, w = frame.shape[:2]
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            
            # Add padding
            padding = 20
            x1 = max(0, min(x_coords) - padding)
            y1 = max(0, min(y_coords) - padding)
            x2 = min(w, max(x_coords) + padding)
            y2 = min(h, max(y_coords) + padding)
            
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
            return None, None
        except:
            return None, None
    
    def recognize_mediapipe(self, frame):
        """Recognize gesture using MediaPipe"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            result = self.mediapipe_recognizer.recognize(mp_img)
            
            if not result.gestures:
                return None, 0.0, None, None
                
            top_gesture = result.gestures[0][0]
            sign = top_gesture.category_name
            conf = top_gesture.score
            
            # Extract hand ROI for ISL model
            hand_roi, bbox = None, None
            if result.hand_landmarks:
                hand_roi, bbox = self.extract_hand_roi(frame, result.hand_landmarks[0])
            
            return sign, conf, result.hand_landmarks, hand_roi
            
        except Exception as e:
            print(f"MediaPipe recognition error: {e}")
            return None, 0.0, None, None
    
    def recognize_isl(self, hand_roi):
        """Recognize gesture using ISL model"""
        if self.isl_model is None or hand_roi is None:
            return None, 0.0
            
        try:
            processed = self.preprocess_for_isl(hand_roi)
            if processed is None:
                return None, 0.0
                
            predictions = self.isl_model.predict(processed, verbose=0)
            confidence = np.max(predictions[0])
            class_idx = np.argmax(predictions[0])
            
            if class_idx < len(self.isl_classes):
                sign = self.isl_classes[class_idx]
                return sign, confidence
                
            return None, 0.0
        except Exception as e:
            print(f"ISL recognition error: {e}")
            return None, 0.0
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        if not landmarks:
            return
            
        for hand_landmarks in landmarks:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                frame, proto, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
            )
    
    def speak(self, text):
        """Speak text using TTS"""
        if not self.audio_available:
            return
            
        def speak_thread():
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
                filepath = os.path.join(TEMP_AUDIO_FOLDER, filename)
                tts.save(filepath)
                
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                print(f"🔊 Spoken: {text}")
                
            except Exception as e:
                print(f"Audio error: {e}")
        
        threading.Thread(target=speak_thread, daemon=True).start()

class CameraSystem:
    """Camera management system"""
    
    @staticmethod
    def test_camera_url(url):
        """Test if camera URL is accessible"""
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
        except:
            pass
        return False
    
    @staticmethod
    def get_camera_choice():
        """Get camera selection from user"""
        print("\n" + "="*50)
        print("🤖 DUAL-MODEL ISL RECOGNITION SYSTEM")
        print("="*50)
        print("📷 CAMERA SELECTION:")
        print("1. Local Webcam")
        print("2. IP Camera (Phone)")
        print("3. Image File")
        print("4. Video File")
        print("5. Exit")
        
        while True:
            choice = input("\nChoose option (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("❌ Invalid choice. Please enter 1-5.")
    
    @staticmethod
    def setup_ip_camera():
        """Setup IP camera connection"""
        print("\n📱 IP CAMERA SETUP")
        print("Available options:")
        for i, (name, url) in enumerate(COMMON_IP_CAMERAS.items(), 1):
            print(f"  {i}. {name}: {url}")
        
        while True:
            choice = input(f"\nSelect option (1-{len(COMMON_IP_CAMERAS)}) or 'c' for custom: ").strip()
            
            if choice.lower() == 'c':
                custom_url = input("Enter custom IP camera URL: ").strip()
                if custom_url:
                    return custom_url
                else:
                    print("❌ Please enter a valid URL")
                    
            elif choice.isdigit() and 1 <= int(choice) <= len(COMMON_IP_CAMERAS):
                selected = list(COMMON_IP_CAMERAS.values())[int(choice)-1]
                if selected == "custom":
                    custom_url = input("Enter custom IP camera URL: ").strip()
                    return custom_url
                return selected
            else:
                print("❌ Invalid selection")

class DualModelApp:
    """Main application class"""
    
    def __init__(self):
        self.recognizer = DualModelGestureRecognizer()
        self.sentence = []
        self.last_sign = None
        self.fps_counter = 0
        self.start_time = time.time()
        
    def initialize(self):
        """Initialize the application"""
        print("🚀 Initializing Dual-Model ISL Recognition System...")
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        if os.path.exists(TEMP_AUDIO_FOLDER):
            shutil.rmtree(TEMP_AUDIO_FOLDER)
        os.makedirs(TEMP_AUDIO_FOLDER)
        
        # Initialize systems
        self.recognizer.initialize_audio()
        
        if not self.recognizer.load_mediapipe_model():
            return False
            
        self.recognizer.load_isl_model()
        
        print("✅ System initialized successfully!")
        print("📊 Models loaded:")
        print(f"   • MediaPipe: ✓")
        print(f"   • ISL Model: {'✓' if self.recognizer.isl_model else '✗ (Using MediaPipe only)'}")
        return True
    
    def process_frame(self, frame, fps=0):
        """Process a single frame with both models"""
        # Recognize with MediaPipe
        mp_sign, mp_conf, landmarks, hand_roi = self.recognizer.recognize_mediapipe(frame)
        
        final_sign = None
        final_conf = 0.0
        model_used = "None"
        
        # Draw landmarks
        self.recognizer.draw_landmarks(frame, landmarks)
        
        # Try MediaPipe first
        if mp_sign and mp_conf > CONFIDENCE_THRESHOLDS["mediapipe"]:
            final_sign = mp_sign
            final_conf = mp_conf
            model_used = "MediaPipe"
            
            # If MediaPipe detects a letter/number, try ISL model for confirmation
            if mp_sign in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" and hand_roi is not None:
                isl_sign, isl_conf = self.recognizer.recognize_isl(hand_roi)
                if isl_sign and isl_conf > CONFIDENCE_THRESHOLDS["isl"]:
                    final_sign = isl_sign
                    final_conf = isl_conf
                    model_used = "ISL Model"
        
        # If MediaPipe didn't find anything good, try ISL model
        elif hand_roi is not None and self.recognizer.isl_model:
            isl_sign, isl_conf = self.recognizer.recognize_isl(hand_roi)
            if isl_sign and isl_conf > CONFIDENCE_THRESHOLDS["isl"]:
                final_sign = isl_sign
                final_conf = isl_conf
                model_used = "ISL Model"
        
        # Process the final sign
        if final_sign and final_conf > 0.5:
            display_text = MEDIAPIPE_GESTURES.get(final_sign, final_sign)
            
            # Color coding based on model
            if model_used == "MediaPipe":
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 255)  # Magenta for ISL model
            
            # Display recognition info
            cv2.putText(frame, f"{display_text} ({final_conf:.2f})", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Model: {model_used}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add to sentence and speak if new sign
            if (final_sign != self.last_sign and 
                final_sign not in ['None', 'Unknown'] and
                final_conf > max(CONFIDENCE_THRESHOLDS.values())):
                
                self.sentence.append(final_sign)
                self.recognizer.speak(display_text)
                self.last_sign = final_sign
                
                # Limit sentence length
                if len(self.sentence) > 8:
                    self.sentence = self.sentence[-8:]
        else:
            cv2.putText(frame, "No gesture detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display current sentence
        sentence_text = " ".join(self.sentence) if self.sentence else "Show gestures here..."
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display FPS
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run_webcam(self, camera_url=None):
        """Run with webcam or IP camera"""
        if camera_url:
            print(f"📱 Connecting to IP camera: {camera_url}")
            cap = cv2.VideoCapture(camera_url)
            window_name = "Dual-Model ISL - IP Camera"
        else:
            print("📷 Starting local webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)
            window_name = "Dual-Model ISL - Webcam"
            
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        
        print("✅ Camera connected! Press 'Q' to quit")
        print("🎯 Show hand gestures to the camera...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Calculate FPS
            self.fps_counter += 1
            fps = 0
            if self.fps_counter % 30 == 0:
                end_time = time.time()
                try:
                    fps = 30 / (end_time - self.start_time)
                except ZeroDivisionError:
                    fps = 0
                self.start_time = time.time()
            
            # Process frame
            frame = self.process_frame(frame, fps)
            
            # Display instructions
            cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Show window
            cv2.imshow(window_name, frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def run_image_mode(self):
        """Process a single image"""
        img_path = input("Enter image path: ").strip().strip('"')
        if not os.path.exists(img_path):
            print("❌ Image not found!")
            return
        
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Failed to load image!")
            return
        
        img = self.process_frame(img)
        cv2.putText(img, "Press any key to exit", (10, img.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Dual-Model ISL - Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_video_mode(self):
        """Process a video file"""
        vid_path = input("Enter video path: ").strip().strip('"')
        if not os.path.exists(vid_path):
            print("❌ Video not found!")
            return
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print("❌ Failed to open video!")
            return
        
        print("🎥 Processing video... Press 'Q' to stop")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            cv2.imshow("Dual-Model ISL - Video", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        while True:
            choice = CameraSystem.get_camera_choice()
            
            if choice == "1":
                self.run_webcam()
                break
            elif choice == "2":
                camera_url = CameraSystem.setup_ip_camera()
                if camera_url and CameraSystem.test_camera_url(camera_url):
                    self.run_webcam(camera_url)
                    break
                else:
                    print("❌ Cannot connect to IP camera. Please check:")
                    print("   • Phone and computer on same WiFi")
                    print("   • IP camera app is running")
                    print("   • URL is correct")
            elif choice == "3":
                self.run_image_mode()
                break
            elif choice == "4":
                self.run_video_mode()
                break
            elif choice == "5":
                print("👋 Goodbye!")
                break

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        app = DualModelApp()
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("🎯 Application closed")