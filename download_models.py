# download_models.py - Downloads required models automatically
import os
import requests
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download file with progress indicator"""
    Path("models").mkdir(exist_ok=True)
    filepath = os.path.join("models", filename)
    
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists")
        return True
        
    print(f"📥 Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    print("🚀 Downloading Required Models...")
    
    # MediaPipe Gesture Recognizer
    gesture_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    download_file(gesture_url, "gesture_recognizer.task")
    
    # Create ISL classes file
    isl_classes = """0
1
2
3
4
5
6
7
8
9
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
hello
iloveyou
no
please
sorry
thankyou
yes"""
    
    with open("models/isl_classes.txt", "w") as f:
        f.write(isl_classes)
    
    print("✅ All models setup completed!")
    print("📝 Note: You'll need to train or obtain the ISL model (isl_model.h5) separately")
    print("   You can train it using the GitHub project's training script")

if __name__ == "__main__":
    main()