import cv2
import numpy as np
import math

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Camera
cap = cv2.VideoCapture(0)

# Thresholds
ear_threshold = 0.25
required_consistency = 3  # frames

# State
blink_count = 0
closed_frames = 0

# Delay buffers
eye_closed_counter = 0
eye_open_counter = 0
eye_closed_final = False

def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio (EAR)"""
    if len(eye_points) < 6:
        return 1.0  # Return high value if not enough points
    
    # Convert to numpy array for easier calculation
    points = np.array(eye_points)
    
    # Calculate vertical distances
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    
    # Calculate horizontal distance
    horizontal = np.linalg.norm(points[0] - points[3])
    
    # Avoid division by zero
    if horizontal == 0:
        return 1.0
    
    # Calculate EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_eyes(frame):
    """Detect eyes using OpenCV cascade classifiers"""
    global blink_count, closed_frames, eye_closed_counter, eye_open_counter, eye_closed_final
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eyes_detected = 0
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes_detected = len(eyes)
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Simple blink detection based on number of eyes detected
    if eyes_detected < 2:  # Less than 2 eyes detected (likely blinking)
        closed_frames += 1
        eye_closed_counter += 1
        eye_open_counter = 0
    else:
        if closed_frames >= required_consistency:
            blink_count += 1
        closed_frames = 0
        eye_open_counter += 1
        eye_closed_counter = 0
    
    # Update final eye state
    if eye_closed_counter >= required_consistency:
        eye_closed_final = True
    elif eye_open_counter >= required_consistency:
        eye_closed_final = False

print("Starting eye-controlled screen...")
print("Press 'ESC' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Detect eyes
    detect_eyes(frame)
    
    # Decide screen content (REVERSED LOGIC - as requested)
    if eye_closed_final:
        # Show normal frame when eyes are closed (show content)
        output = frame.copy()
        cv2.putText(output, 'EYES CLOSED - Screen Visible', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # Show black frame when eyes are open (hide content)
        output = frame.copy()
        output[:] = 0  # Make it black
        cv2.putText(output, 'EYES OPEN - Screen Hidden', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Show blink count
    cv2.putText(output, f'Blinks: {blink_count}', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
    
    # Show eye state
    if eye_closed_final:
        cv2.putText(output, 'Status: Eyes Closed', (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(output, 'Status: Eyes Open', (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show final output
    cv2.imshow('Eye-Controlled Screen', output)
    
    # Exit key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total blinks detected: {blink_count}") 