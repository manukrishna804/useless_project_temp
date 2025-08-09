from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, send_from_directory
import cv2
import numpy as np
import threading
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for sessions

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained face, eye, and smile cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

class EyeDetector:
    def __init__(self):
        self.ear_threshold = 0.25
        self.required_consistency = 2  # Reduced for more responsive detection
        self.blink_count = 0
        self.closed_frames = 0
        self.eye_closed_counter = 0
        self.eye_open_counter = 0
        self.eye_closed_final = False
        self.is_streaming = False
        self.video_source = None
        self.current_frame = None
        self.lock = threading.Lock()
        self.face_detected = False
        
    def reset_stats(self):
        """Reset all eye detection statistics"""
        with self.lock:
            self.blink_count = 0
            self.closed_frames = 0
            self.eye_closed_counter = 0
            self.eye_open_counter = 0
            self.eye_closed_final = False
            self.face_detected = False
    
    def detect_eyes(self, frame):
        """Detect eyes and update blink statistics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more sensitive parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        eyes_detected = 0
        self.face_detected = len(faces) > 0
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in the face region with more sensitive parameters
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(15, 15))
            eyes_detected = len(eyes)
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # More responsive blink detection
        if not self.face_detected or eyes_detected < 2:  # No face or less than 2 eyes detected
            self.closed_frames += 1
            self.eye_closed_counter += 1
            self.eye_open_counter = 0
        else:
            if self.closed_frames >= self.required_consistency:
                self.blink_count += 1
            self.closed_frames = 0
            self.eye_open_counter += 1
            self.eye_closed_counter = 0
        
        # Update final eye state with more responsive logic
        if self.eye_closed_counter >= self.required_consistency:
            self.eye_closed_final = True
        elif self.eye_open_counter >= self.required_consistency:
            self.eye_closed_final = False
        
        return frame
    
    def process_frame(self, frame):
        """Process frame and add eye detection overlays"""
        # Flip the frame horizontally for selfie-view display (only for webcam)
        if self.video_source == 'webcam':
            frame = cv2.flip(frame, 1)
        
        # Detect eyes
        processed_frame = self.detect_eyes(frame.copy())
        
        # Decide screen content - KEY LOGIC:
        # EYES OPEN = BLACK SCREEN (hide content)
        # EYES CLOSED = SHOW VIDEO (show content)
        if self.eye_closed_final:
            # Eyes are CLOSED - SHOW the video content
            output = processed_frame.copy()
            cv2.putText(output, 'EYES CLOSED - Video Visible', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            status_color = (0, 255, 0)  # Green for visible
        else:
            # Eyes are OPEN - HIDE the video (black screen)
            output = np.zeros_like(processed_frame)  # Black screen
            cv2.putText(output, 'EYES OPEN - Video Hidden', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            status_color = (255, 255, 255)  # White text on black
        
        # Show blink count
        cv2.putText(output, f'Blinks: {self.blink_count}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Show detailed eye state
        face_status = "Face Detected" if self.face_detected else "No Face"
        cv2.putText(output, f'{face_status}', (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        eye_status = "Eyes CLOSED" if self.eye_closed_final else "Eyes OPEN"
        cv2.putText(output, f'Status: {eye_status}', (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show counters for debugging
        cv2.putText(output, f'Closed Count: {self.eye_closed_counter}', (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(output, f'Open Count: {self.eye_open_counter}', (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        with self.lock:
            self.current_frame = output.copy()
        
        return output

# Global eye detector instance
eye_detector = EyeDetector()

class SmileDetector:
    def __init__(self):
        self.smile_threshold = 0.6  # Confidence threshold for smile detection
        self.required_smiles = 3  # Number of consecutive smile detections required
        self.smile_counter = 0
        self.no_smile_counter = 0
        self.is_detecting = False
        self.cap = None
        
    def detect_smile(self, frame):
        """Detect smile in the given frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        smile_detected = False
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect smile in the face region
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20, minSize=(25, 25))
            
            if len(smiles) > 0:
                smile_detected = True
                # Draw rectangle around smile
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                    cv2.putText(roi_color, 'SMILE!', (sx, sy-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Update smile counters
        if smile_detected:
            self.smile_counter += 1
            self.no_smile_counter = 0
        else:
            self.no_smile_counter += 1
            if self.no_smile_counter > 5:  # Reset smile counter if no smile for too long
                self.smile_counter = 0
        
        return frame, smile_detected
    
    def is_happy(self):
        """Check if enough smiles have been detected"""
        return self.smile_counter >= self.required_smiles
    
    def reset(self):
        """Reset smile detection counters"""
        self.smile_counter = 0
        self.no_smile_counter = 0
        self.is_detecting = False
        if self.cap:
            self.cap.release()
            self.cap = None

# Global smile detector instance
smile_detector = SmileDetector()

def generate_frames_webcam():
    """Generate frames from webcam"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while eye_detector.is_streaming:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with eye detection
            processed_frame = eye_detector.process_frame(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.016)  # ~60 FPS for better responsiveness
    finally:
        cap.release()

def generate_frames_video(video_path):
    """Generate frames from video file with webcam eye detection overlay"""
    video_cap = cv2.VideoCapture(video_path)
    webcam_cap = cv2.VideoCapture(0)  # Webcam for eye detection
    
    try:
        while eye_detector.is_streaming:
            # Read from both video file and webcam
            video_ret, video_frame = video_cap.read()
            webcam_ret, webcam_frame = webcam_cap.read()
            
            # Loop video if ended
            if not video_ret:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # If webcam is available, detect eyes from webcam
            if webcam_ret:
                # Flip webcam for mirror effect
                webcam_frame = cv2.flip(webcam_frame, 1)
                # Detect eyes from webcam feed and get processed frame with rectangles
                webcam_processed = eye_detector.detect_eyes(webcam_frame.copy())
            
            # Decide what to show based on eye detection
            if eye_detector.eye_closed_final:
                # Eyes CLOSED - Show the video content
                output = video_frame.copy()
                
                # Add webcam overlay in corner for monitoring
                if webcam_ret:
                    small_webcam = cv2.resize(webcam_processed, (160, 120))
                    h, w = small_webcam.shape[:2]
                    output[10:10+h, 10:10+w] = small_webcam
                    # Add green border for "eyes closed" state
                    cv2.rectangle(output, (8, 8), (12+w, 12+h), (0, 255, 0), 3)
                
                cv2.putText(output, 'EYES CLOSED - Video Visible', (20, output.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                status_color = (0, 255, 0)
            else:
                # Eyes OPEN - Show black screen
                output = np.zeros_like(video_frame)
                
                # Add webcam overlay in corner for monitoring
                if webcam_ret:
                    small_webcam = cv2.resize(webcam_processed, (160, 120))
                    h, w = small_webcam.shape[:2]
                    output[10:10+h, 10:10+w] = small_webcam
                    # Add red border for "eyes open" state
                    cv2.rectangle(output, (8, 8), (12+w, 12+h), (0, 0, 255), 3)
                
                cv2.putText(output, 'EYES OPEN - Video Hidden', (20, output.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                status_color = (255, 255, 255)
            
            # Add statistics overlay
            cv2.putText(output, f'Blinks: {eye_detector.blink_count}', (20, output.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Add eye status
            eye_status = "Eyes CLOSED" if eye_detector.eye_closed_final else "Eyes OPEN"
            cv2.putText(output, f'Status: {eye_status}', (20, output.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Update current frame for stats
            with eye_detector.lock:
                eye_detector.current_frame = output.copy()
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', output)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS for video playback
    finally:
        video_cap.release()
        webcam_cap.release()

@app.route('/')
def index():
    """Redirect to login page"""
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Login page with smile detection"""
    return render_template('login.html')

@app.route('/home')
def home():
    """Home page with project showcase - requires login"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/eye-detection')
def eye_detection_app():
    """Eye detection application page"""
    return render_template('index.html')

@app.route('/watch-detector')
def watch_detector_page():
    """Watch detector fun page - requires login"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('watch_detector.html')

@app.route('/pucham.jpeg')
def serve_pucham_image():
    """Serve the pucham.jpeg image used in the watch detector page"""
    return send_from_directory(app.root_path, 'pucham.jpeg')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if eye_detector.video_source == 'webcam':
        return Response(generate_frames_webcam(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif eye_detector.video_source and os.path.exists(eye_detector.video_source):
        return Response(generate_frames_video(eye_detector.video_source),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video source available", 404

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam streaming"""
    eye_detector.video_source = 'webcam'
    eye_detector.is_streaming = True
    eye_detector.reset_stats()
    return jsonify({'status': 'success', 'message': 'Webcam started'})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video streaming"""
    eye_detector.is_streaming = False
    eye_detector.video_source = None
    return jsonify({'status': 'success', 'message': 'Stream stopped'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload and start processing video file"""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        eye_detector.video_source = video_path
        eye_detector.is_streaming = True
        eye_detector.reset_stats()
        
        return jsonify({'status': 'success', 'message': f'Video {filename} uploaded and started'})

@app.route('/stats')
def get_stats():
    """Get current eye detection statistics"""
    with eye_detector.lock:
        stats = {
            'blink_count': eye_detector.blink_count,
            'eyes_closed': eye_detector.eye_closed_final,
            'is_streaming': eye_detector.is_streaming,
            'video_source': 'webcam' if eye_detector.video_source == 'webcam' else 'video file',
            'face_detected': eye_detector.face_detected,
            'eye_closed_counter': eye_detector.eye_closed_counter,
            'eye_open_counter': eye_detector.eye_open_counter
        }
    return jsonify(stats)

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset eye detection statistics"""
    eye_detector.reset_stats()
    return jsonify({'status': 'success', 'message': 'Statistics reset'})

@app.route('/smile_check', methods=['POST'])
def smile_check():
    """Check for smile using webcam"""
    try:
        # Reset smile detector
        smile_detector.reset()
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera'})
        
        smile_detector.is_detecting = True
        smile_detector.cap = cap
        
        # Check for smile for up to 10 seconds
        start_time = time.time()
        max_duration = 10  # seconds
        
        while smile_detector.is_detecting and (time.time() - start_time) < max_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect smile
            processed_frame, smile_detected = smile_detector.detect_smile(frame)
            
            # Add status text
            status_text = f"Smiles: {smile_detector.smile_counter}/{smile_detector.required_smiles}"
            cv2.putText(processed_frame, status_text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Check if enough smiles detected
            if smile_detector.is_happy():
                smile_detector.is_detecting = False
                break
            
            # Small delay
            time.sleep(0.1)
        
        # Clean up
        cap.release()
        smile_detector.is_detecting = False
        smile_detector.cap = None
        
        # Check result
        if smile_detector.is_happy():
            session['logged_in'] = True
            return jsonify({'success': True, 'message': 'Smile detected! Welcome!'})
        else:
            return jsonify({'success': False, 'message': 'Please smile 😁'})
            
    except Exception as e:
        print(f"Error in smile detection: {e}")
        return jsonify({'success': False, 'message': 'Error during smile detection'})

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
