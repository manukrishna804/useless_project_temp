# 👁️ Eye Detection Video Controller - How to Use

## 🚀 Quick Start

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:** `http://localhost:5000`

## 🎯 How It Works

### **Core Concept:**
- **👁️ Eyes OPEN** = **🖤 Video HIDDEN** (black screen)
- **👁️ Eyes CLOSED** = **📺 Video VISIBLE** (content shown)

## 🎮 Two Modes of Operation

### **Mode 1: Webcam Only**
1. Click **"📷 Start Webcam"**
2. Position your face in front of the camera
3. Open/close your eyes to control the display
4. Watch the real-time statistics update

### **Mode 2: Video Upload + Eye Control**
1. Click **"📁 Upload Video"** and select any video file
2. **Sit in front of your webcam** (this is important!)
3. The uploaded video will play, but YOU control its visibility:
   - **Open your eyes** → Video becomes BLACK/HIDDEN
   - **Close your eyes** → Video becomes VISIBLE
4. **Small webcam overlay** in the corner shows your face with eye detection rectangles

## 📊 What You'll See

### **On Screen:**
- **Main video area:** Shows your uploaded video OR webcam feed
- **Webcam overlay:** Small window showing eye detection (blue rectangles around face, green around eyes)
- **Status text:** Shows current eye state and blink count
- **Border colors:** 
  - 🟢 Green border = Eyes closed (video visible)
  - 🔴 Red border = Eyes open (video hidden)

### **Statistics Panel:**
- **Total Blinks:** Real-time blink counter
- **Eye Status:** Current state (Open/Closed)
- **Stream Status:** Whether detection is active
- **Face Detected:** Whether your face is visible to the camera
- **Debug Counters:** Internal detection counters

## 🔧 Controls

- **📷 Start Webcam:** Begin webcam-only mode
- **⏹️ Stop Stream:** Stop current detection
- **📁 Upload Video:** Upload and start video with eye control
- **🔄 Reset Stats:** Reset blink counter and statistics

## 💡 Tips for Best Results

1. **Good lighting:** Ensure your face is well-lit
2. **Face the camera:** Position yourself directly in front of the webcam
3. **Stable position:** Try to keep your head relatively still
4. **Clear view:** Make sure nothing is blocking your eyes from the camera
5. **Test first:** Try webcam mode first to see if detection works well

## 🐛 Troubleshooting

**Video not responding to eyes?**
- Make sure your webcam is working and you're sitting in front of it
- Check if "Face Detected" shows "Yes" in the statistics
- Ensure good lighting on your face

**Eyes not detecting properly?**
- Move closer to or further from the camera
- Improve lighting conditions
- Make sure your entire face is visible in the webcam feed

**Video upload not working?**
- Supported formats: MP4, AVI, MOV, WMV
- File size limit: 100MB
- Make sure the file isn't corrupted

## 🎬 Test Video

A test video file (`test_video.mp4`) is created automatically when you run the application. Use this to test the eye control functionality!

---

**Enjoy controlling videos with your eyes! 👁️✨**
