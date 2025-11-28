# config.py - Configuration for dual-model system
import os

# Gesture mappings for MediaPipe model
MEDIAPIPE_GESTURES = {
    "A": "Letter A", "B": "Letter B", "C": "Letter C", "D": "Letter D",
    "E": "Letter E", "F": "Letter F", "G": "Letter G", "H": "Letter H",
    "I": "Letter I", "J": "Letter J", "K": "Letter K", "L": "Letter L",
    "M": "Letter M", "N": "Letter N", "O": "Letter O", "P": "Letter P",
    "Q": "Letter Q", "R": "Letter R", "S": "Letter S", "T": "Letter T",
    "U": "Letter U", "V": "Letter V", "W": "Letter W", "X": "Letter X",
    "Y": "Letter Y", "Z": "Letter Z",
    "hello": "Hello", "thanks": "Thank You", "yes": "Yes", "no": "No",
    "please": "Please", "sorry": "Sorry", "help": "Help",
    "iloveyou": "I Love You", "good": "Good", "bad": "Bad",
    "rock": "Rock", "paper": "Paper", "scissors": "Scissors",
    "thumbs_up": "Thumbs Up", "thumbs_down": "Thumbs Down",
    "victory": "Victory", "point_up": "Point Up", "point_down": "Point Down",
    "Unknown": "Unknown Gesture", "None": "No Gesture"
}

# ISL Model mappings
ISL_GESTURES = {
    "0": "Number 0", "1": "Number 1", "2": "Number 2", "3": "Number 3",
    "4": "Number 4", "5": "Number 5", "6": "Number 6", "7": "Number 7", 
    "8": "Number 8", "9": "Number 9",
    "A": "Letter A", "B": "Letter B", "C": "Letter C", "D": "Letter D",
    "E": "Letter E", "F": "Letter F", "G": "Letter G", "H": "Letter H",
    "I": "Letter I", "J": "Letter J", "K": "Letter K", "L": "Letter L",
    "M": "Letter M", "N": "Letter N", "O": "Letter O", "P": "Letter P",
    "Q": "Letter Q", "R": "Letter R", "S": "Letter S", "T": "Letter T",
    "U": "Letter U", "V": "Letter V", "W": "Letter W", "X": "Letter X",
    "Y": "Letter Y", "Z": "Letter Z",
    "hello": "Hello", "iloveyou": "I Love You", "no": "No",
    "please": "Please", "sorry": "Sorry", "thankyou": "Thank You", "yes": "Yes"
}

# Model paths
MODEL_PATHS = {
    "mediapipe": "models/gesture_recognizer.task",
    "isl": "models/isl_model.h5",
    "isl_classes": "models/isl_classes.txt"
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "mediapipe": 0.7,
    "isl": 0.6
}

# Audio settings
TEMP_AUDIO_FOLDER = "temp_audio"
AUDIO_VOLUME = 1.0

# Camera settings
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# IP Camera common URLs
COMMON_IP_CAMERAS = {
    "IP Webcam": "http://192.168.1.100:8080/video",
    "DroidCam": "http://192.168.1.100:4747/video",
    "Custom": "custom"
}