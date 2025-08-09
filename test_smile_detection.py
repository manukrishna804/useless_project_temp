#!/usr/bin/env python3
"""
Simple test script for smile detection functionality
"""

import cv2
import numpy as np

def test_smile_detection():
    """Test smile detection with webcam"""
    print("Testing smile detection...")
    
    # Load cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    if face_cascade.empty():
        print("❌ Error: Could not load face cascade classifier")
        return False
    
    if smile_cascade.empty():
        print("❌ Error: Could not load smile cascade classifier")
        return False
    
    print("✅ Cascade classifiers loaded successfully")
    
    # Test webcam access
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return False
    
    print("✅ Webcam opened successfully")
    
    # Test a few frames
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Error: Could not read frame {i}")
            cap.release()
            return False
        
        # Test face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) > 0:
            print(f"✅ Frame {i}: Face detected")
            
            # Test smile detection on first face
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20, minSize=(25, 25))
            
            if len(smiles) > 0:
                print(f"✅ Frame {i}: Smile detected!")
            else:
                print(f"ℹ️  Frame {i}: No smile detected (this is normal)")
        else:
            print(f"ℹ️  Frame {i}: No face detected (this is normal)")
    
    cap.release()
    print("✅ Smile detection test completed successfully!")
    return True

if __name__ == "__main__":
    test_smile_detection()
