import cv2
import numpy as np

def create_test_video():
    """Create a simple test video with a moving circle to test the eye detection system"""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    # Create frames
    for frame_num in range(total_frames):
        # Create a frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            color_value = int(255 * y / height)
            frame[y, :] = [color_value // 3, color_value // 2, color_value]
        
        # Add moving circle
        center_x = int(width * 0.3 + (width * 0.4) * (frame_num / total_frames))
        center_y = height // 2
        radius = 30
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
        
        # Add text
        text = f"Test Video - Frame {frame_num + 1}/{total_frames}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add instructions
        instruction = "Upload this video and watch your eyes control the visibility!"
        cv2.putText(frame, instruction, (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("Test video 'test_video.mp4' created successfully!")
    print("Upload this video to test the eye detection functionality.")

if __name__ == "__main__":
    create_test_video()
