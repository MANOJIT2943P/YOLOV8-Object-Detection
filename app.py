from fastapi import FastAPI, Response, Form
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import onnxruntime as ort
import logging
import os
from typing import Optional
import threading
import time
import asyncio
import json
from queue import Queue
import tempfile
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load ONNX model
model_path = os.getenv("MODEL_PATH", "best.onnx")
try:
    session = ort.InferenceSession(model_path)
    logger.info("✅ ONNX model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load ONNX model: {e}")
    raise

# Class name mappings
class_names = {
    0: "stair",
    1: "people",
    2: "door",
    3: "pothole"
}

# Colors for each class (BGR format)
colors = {
    0: (255, 0, 0),      # Blue for stair
    1: (0, 255, 0),      # Green for people
    2: (0, 0, 255),      # Red for door
    3: (255, 255, 0),    # Cyan for vehicle
}

# ============================================
# 🎬 VIDEO SOURCE CONFIGURATION
# ============================================
USE_TEST_VIDEO = os.getenv("USE_TEST_VIDEO", "false").lower() == "true"
TEST_VIDEO_PATH = os.getenv("VIDEO_SOURCE", "test.mp4")
# ============================================

# Global variables
current_camera_url = None
video_capture = None
video_capture_lock = threading.Lock()
is_streaming = False
detection_queue = Queue()  # For SSE detection events

def preprocess(frame):
    """Preprocess frame for YOLOv8 model"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
    img /= 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess(frame, outputs, conf_threshold=0.5):
    """Draw bounding boxes on detected objects for YOLOv8 ONNX output."""
    detections_count = 0
    detected_classes = []  # Track detected class names
    h, w = frame.shape[:2]

    # Transpose output tensor from (1, 84, 8400) to (1, 8400, 84)
    if isinstance(outputs, list):
        outputs = outputs[0]
    
    outputs = np.array(outputs).transpose(0, 2, 1)
    
    # Pre-allocate lists for boxes, scores, and class IDs
    boxes = []
    scores = []
    class_ids = []

    # Iterate over all 8400 predictions
    for prediction in outputs[0]:
        bbox = prediction[:4]
        class_scores = prediction[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score > conf_threshold:
            boxes.append(bbox)
            scores.append(score)
            class_ids.append(class_id)
            
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        np.array(boxes).tolist(), 
        np.array(scores).tolist(), 
        conf_threshold, 
        0.45
    )
    
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        
        for i in indices:
            x_center, y_center, width, height = boxes[i]
            x1 = int((x_center - width / 2) * w / 640)
            y1 = int((y_center - height / 2) * h / 640)
            x2 = int((x_center + width / 2) * w / 640)
            y2 = int((y_center + height / 2) * h / 640)
            
            class_id = class_ids[i]
            class_name = class_names.get(class_id, f"Class_{class_id}")
            detected_classes.append(class_name)  # Add to detected classes
            color = colors.get(class_id, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {scores[i]:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                          (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections_count += 1
    
    return frame, detections_count, detected_classes

def gen_frames():
    """Generate frames with YOLOv8 inference from video source"""
    global is_streaming, current_camera_url, video_capture
    
    # Determine video source
    if USE_TEST_VIDEO:
        if not os.path.exists(TEST_VIDEO_PATH):
            logger.error(f"❌ Test video not found: {TEST_VIDEO_PATH}")
            return
        video_source = TEST_VIDEO_PATH
        logger.info(f"🎬 [TESTING MODE] Using video file: {TEST_VIDEO_PATH}")
    else:
        if current_camera_url is None:
            logger.error("❌ No camera URL configured")
            return
        video_source = current_camera_url
        logger.info(f"🔌 Connecting to camera: {current_camera_url}")
    
    # Open video capture
    cap = cv2.VideoCapture(video_source)
    
    # Set buffer size to 1 for real-time streaming
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # For network streams, set timeout
    if not USE_TEST_VIDEO and video_source.startswith(('http', 'rtsp')):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video source: {video_source}")
        is_streaming = False
        return
    
    logger.info("✅ Video source opened successfully!")
    is_streaming = True
    
    # Store capture object
    with video_capture_lock:
        video_capture = cap
    
    frame_count = 0
    consecutive_failures = 0
    max_failures = 10
    last_announcement = {}  # Track last announcement per class
    announcement_cooldown = 3  # Seconds between announcements
    
    try:
        while is_streaming:
            success, frame = cap.read()
            
            if not success:
                consecutive_failures += 1
                logger.warning(f"⚠ Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error("❌ Too many consecutive failures, stopping stream")
                    break
                
                if USE_TEST_VIDEO:
                    logger.info("🔄 Video ended, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    consecutive_failures = 0
                    continue
                else:
                    time.sleep(0.1)
                    continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            frame_count += 1
            
            # Run inference
            try:
                input_tensor = preprocess(frame)
                outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
                annotated_frame, detection_count, detected_classes = postprocess(frame, outputs)
                
                # Check for new detections to announce
                current_time = time.time()
                for class_name in set(detected_classes):  # Unique classes only
                    if class_name not in last_announcement or \
                       (current_time - last_announcement[class_name]) > announcement_cooldown:
                        last_announcement[class_name] = current_time
                        
                        # Push to queue for SSE
                        detection_queue.put({
                            "class": class_name,
                            "timestamp": current_time
                        })
                        logger.info(f"🔊 Detected: {class_name}")
                
                if USE_TEST_VIDEO:
                    cv2.putText(annotated_frame, f"[TEST] Frame: {frame_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                annotated_frame = frame
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                logger.error("❌ Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        logger.error(f"❌ Error in gen_frames: {e}")
    
    finally:
        with video_capture_lock:
            if video_capture is not None:
                video_capture.release()
                video_capture = None
        is_streaming = False
        logger.info("🔌 Video source closed")

async def detection_event_stream():
    """Stream detection events to frontend via SSE"""
    while True:
        if not detection_queue.empty():
            detection = detection_queue.get()
            yield f"data: {json.dumps(detection)}\n\n"
        await asyncio.sleep(0.1)

@app.get("/detection_stream")
async def detection_stream():
    """Server-Sent Events endpoint for real-time detections"""
    return StreamingResponse(
        detection_event_stream(),
        media_type="text/event-stream"
    )

@app.get("/text_to_speech/{text}")
def text_to_speech(text: str):
    """Generate speech for detected object"""
    try:
        # Create speech
        speech_text = f"{text} detected"
        tts = gTTS(text=speech_text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)
        
        # Read the file
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Delete temp file
        os.unlink(temp_path)
        
        return Response(content=audio_data, media_type="audio/mpeg")
    
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/set_camera_url")
async def set_camera_url(camera_url: str = Form(...)):
    """Set camera URL from user input"""
    global current_camera_url, is_streaming, video_capture
    
    if USE_TEST_VIDEO:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Server is in test mode. Cannot set camera URL."}
        )
    
    # Stop current stream if running
    if is_streaming:
        is_streaming = False
        time.sleep(0.5)
        
        with video_capture_lock:
            if video_capture is not None:
                video_capture.release()
                video_capture = None
    
    if not camera_url or camera_url.strip() == "":
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Camera URL is required"}
        )
    
    current_camera_url = camera_url.strip()
    logger.info(f"✅ Camera URL set to: {current_camera_url}")
    return {"success": True, "message": "Camera URL configured", "url": current_camera_url}

@app.get("/video_feed")
def video_feed():
    """Video streaming route"""
    if not USE_TEST_VIDEO and current_camera_url is None:
        return Response(
            content="No camera URL configured. Please set camera URL first.",
            status_code=400
        )
    
    logger.info("📹 Video feed requested")
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stop_stream")
def stop_stream():
    """Stop the current video stream"""
    global is_streaming, video_capture
    
    is_streaming = False
    
    with video_capture_lock:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
    
    logger.info("🛑 Stream stop requested")
    return {"success": True, "message": "Stream stopped"}

@app.get("/stream_status")
def stream_status():
    """Get current streaming status"""
    return {
        "is_streaming": is_streaming,
        "current_url": current_camera_url,
        "test_mode": USE_TEST_VIDEO
    }

@app.get("/")
def home():
    """Home page with camera URL input and stream"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESP32-CAM YOLOv8 Detection with Speech</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                max-height: 100vh;
                overflow: hidden;
            }

            .container {
                background: white;
                border-radius: 15px;
                padding: 15px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                max-width: 1400px;
                width: 100%;
                max-height: 95vh;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }

            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 5px;
                font-size: 1.5rem;
            }

            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 10px;
                font-size: 0.9rem;
            }

            .camera-setup {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 15px;
                border: 2px solid #667eea;
            }

            .camera-setup h2 {
                margin: 0 0 10px 0;
                color: #333;
                font-size: 16px;
            }

            .input-group {
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }

            .input-group input {
                flex: 1;
                padding: 10px 12px;
                border: 2px solid #e9ecef;
                border-radius: 5px;
                font-size: 14px;
                transition: border-color 0.3s;
            }

            .input-group input:focus {
                outline: none;
                border-color: #667eea;
            }

            .button-group {
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }

            .btn {
                padding: 10px 25px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s;
                font-weight: 600;
            }

            .btn-primary {
                background: #667eea;
                color: white;
            }

            .btn-primary:hover:not(:disabled) {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }

            .btn-danger {
                background: #dc3545;
                color: white;
            }

            .btn-danger:hover:not(:disabled) {
                background: #c82333;
            }

            .btn-success {
                background: #28a745;
                color: white;
            }

            .btn-success:hover:not(:disabled) {
                background: #218838;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
            }

            .btn-warning {
                background: #ffc107;
                color: #333;
            }

            .btn-warning:hover:not(:disabled) {
                background: #e0a800;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(255, 193, 7, 0.4);
            }

            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }

            .status-message {
                padding: 10px 15px;
                border-radius: 5px;
                margin-top: 10px;
                text-align: center;
                display: none;
                font-size: 13px;
            }

            .status-message.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }

            .status-message.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }

            .status-message.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }

            .video-container {
                position: relative;
                width: 100%;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                max-height: 50vh;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }

            .video-container img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
                max-height: 50vh;
            }

            .video-placeholder {
                color: #aaa;
                text-align: center;
                padding: 30px;
            }

            .video-placeholder h3 {
                margin: 0 0 10px 0;
                color: #aaa;
                font-size: 1rem;
            }

            .video-placeholder p {
                font-size: 0.85rem;
                margin: 5px 0;
            }

            .info-panel {
                margin-top: 15px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }

            .info-card {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }

            .info-card h3 {
                margin: 0 0 8px 0;
                color: #333;
                font-size: 13px;
            }

            .info-card p {
                margin: 0;
                color: #666;
                font-size: 11px;
                word-break: break-all;
            }

            .class-legend {
                margin-top: 10px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
            }

            .class-legend h3 {
                margin: 0 0 8px 0;
                color: #333;
                font-size: 14px;
            }

            .class-item {
                display: inline-block;
                margin: 4px 8px 4px 0;
                padding: 4px 8px;
                background: #e9ecef;
                border-radius: 5px;
                font-size: 12px;
            }

            .example-urls {
                background: #e7f3ff;
                padding: 8px 12px;
                border-radius: 5px;
                margin-top: 8px;
                font-size: 11px;
                color: #004085;
                line-height: 1.4;
            }

            .example-urls strong {
                display: block;
                margin-bottom: 4px;
            }

            /* Voice indicator badge */
            .voice-indicator {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
                vertical-align: middle;
            }

            .voice-indicator.enabled {
                background: #d4edda;
                color: #155724;
            }

            .voice-indicator.disabled {
                background: #f8d7da;
                color: #721c24;
            }

            /* Scrollbar styling for container */
            .container::-webkit-scrollbar {
                width: 8px;
            }

            .container::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }

            .container::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 10px;
            }

            .container::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📷 ESP32-CAM YOLOv8 Detection with Speech 🔊</h1>
            <p class="subtitle">Connect your camera and get real-time object detection with audio alerts</p>
            
            <div class="camera-setup">
                <h2>🔗 Camera Connection</h2>
                <div class="input-group">
                    <input type="text" id="cameraUrl" 
                           placeholder="Enter your ESP32-CAM or IP Camera URL"
                           value="http://192.168.1.100:81/stream">
                </div>
                
                <div class="example-urls">
                    <strong>📝 Example URLs:</strong>
                    • ESP32-CAM: http://192.168.1.100:81/stream<br>
                    • IP Camera (RTSP): rtsp://username:password@192.168.1.100:554/stream<br>
                    • USB Webcam: 0 (device index)
                </div>
                
                <div class="button-group">
                    <button class="btn btn-primary" onclick="startDetection()" id="startBtn">
                        ▶ Start Detection
                    </button>
                    <button class="btn btn-danger" onclick="stopDetection()" id="stopBtn" disabled>
                        ⏹ Stop Detection
                    </button>
                    <button class="btn btn-success" onclick="toggleVoice()" id="voiceBtn">
                        🔊 Voice: ON
                    </button>
                </div>
                
                <div class="status-message" id="statusMessage"></div>
            </div>
            
            <div class="video-container" id="videoContainer">
                <div class="video-placeholder">
                    <h3>👆 Enter your camera URL and click "Start Detection"</h3>
                    <p>Make sure your camera is powered on and connected to the same network</p>
                    <p>🔊 Audio alerts will play when objects are detected (toggle voice button to enable/disable)</p>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="info-card">
                    <h3>📹 Camera URL</h3>
                    <p id="currentUrl">Not configured</p>
                </div>
                <div class="info-card">
                    <h3>🧠 Model</h3>
                    <p>YOLOv8 (ONNX)</p>
                </div>
                <div class="info-card">
                    <h3>🎯 Input Size</h3>
                    <p>640x640 pixels</p>
                </div>
                <div class="info-card">
                    <h3>⚡ Status</h3>
                    <p id="streamStatus">Stopped</p>
                </div>
                <div class="info-card">
                    <h3>🔊 Voice Alerts</h3>
                    <p id="voiceStatus">
                        <span class="voice-indicator enabled">ENABLED</span>
                    </p>
                </div>
            </div>
            
            <div class="class-legend">
                <h3>🏷 Detected Classes (with audio alerts):</h3>
                <span class="class-item">🪜 Stair</span>
                <span class="class-item">👤 People</span>
                <span class="class-item">🚪 Door</span>
                <span class="class-item">🕳 Pothole</span>
            </div>
        </div>
        
        <script>
    let streamCheckInterval = null;
    let streamImg = null;
    let detectionListener = null;
    let voiceEnabled = true;  // Voice is ON by default
    
    function showStatus(message, type) {
        const statusEl = document.getElementById('statusMessage');
        statusEl.textContent = message;
        statusEl.className = 'status-message ' + type;
        statusEl.style.display = 'block';
        
        setTimeout(() => {
            statusEl.style.display = 'none';
        }, 5000);
    }
    
    function toggleVoice() {
        voiceEnabled = !voiceEnabled;
        
        const voiceBtn = document.getElementById('voiceBtn');
        const voiceStatus = document.getElementById('voiceStatus');
        
        if (voiceEnabled) {
            voiceBtn.textContent = '🔊 Voice: ON';
            voiceBtn.className = 'btn btn-success';
            voiceStatus.innerHTML = '<span class="voice-indicator enabled">ENABLED</span>';
            showStatus('🔊 Voice alerts enabled', 'success');
        } else {
            voiceBtn.textContent = '🔇 Voice: OFF';
            voiceBtn.className = 'btn btn-warning';
            voiceStatus.innerHTML = '<span class="voice-indicator disabled">DISABLED</span>';
            showStatus('🔇 Voice alerts disabled', 'info');
        }
    }
    
    function playDetectionSound(className) {
        if (!voiceEnabled) {
            console.log('Voice disabled, skipping sound for:', className);
            return;
        }
        
        try {
            const audio = new Audio('/text_to_speech/' + encodeURIComponent(className));
            audio.play().catch(e => console.error('Audio play failed:', e));
            
            showStatus('🔊 Detected: ' + className, 'info');
        } catch (e) {
            console.error('Error playing sound:', e);
        }
    }
    
    function startDetectionListener() {
        const eventSource = new EventSource('/detection_stream');
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                playDetectionSound(data.class);
            } catch (e) {
                console.error('Error parsing detection event:', e);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('SSE Error:', error);
        };
        
        return eventSource;
    }
    
    async function startDetection() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        startBtn.disabled = true;
        showStatus('🔄 Starting video stream...', 'info');
        
        try {
            // Check if in test mode
            const statusResponse = await fetch('/stream_status');
            const statusData = await statusResponse.json();
            
            if (!statusData.test_mode) {
                // Production mode - need camera URL
                const cameraUrl = document.getElementById('cameraUrl').value.trim();
                
                if (!cameraUrl) {
                    showStatus('⚠ Please enter a camera URL', 'error');
                    startBtn.disabled = false;
                    return;
                }
                
                // Set camera URL
                const formData = new FormData();
                formData.append('camera_url', cameraUrl);
                
                const response = await fetch('/set_camera_url', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (!result.success) {
                    showStatus('❌ ' + result.message, 'error');
                    startBtn.disabled = false;
                    return;
                }
                
                document.getElementById('currentUrl').textContent = result.url;
            } else {
                // Test mode
                document.getElementById('currentUrl').textContent = 'Test Video (Local File)';
            }
            
            // Update video feed
            const videoContainer = document.getElementById('videoContainer');
            const timestamp = new Date().getTime();
            
            videoContainer.innerHTML = '<img id="streamImg" src="/video_feed?t=' + timestamp + '" alt="Video Stream" onerror="handleStreamError()">';
            
            streamImg = document.getElementById('streamImg');
            
            // Update UI
            document.getElementById('streamStatus').textContent = '🟢 Running';
            
            stopBtn.disabled = false;
            
            const voiceStatusMsg = voiceEnabled ? '🔊' : '🔇';
            showStatus('✅ Detection started! Voice alerts: ' + (voiceEnabled ? 'ON' : 'OFF'), 'success');
            
            // Start detection listener for audio alerts
            detectionListener = startDetectionListener();
            
            // Start status checking
            startStatusCheck();
            
        } catch (error) {
            showStatus('❌ Error: ' + error.message, 'error');
            startBtn.disabled = false;
        }
    }
    
    function handleStreamError() {
        console.error('Stream image failed to load');
        showStatus('⚠ Stream connection lost', 'error');
    }
    
    async function stopDetection() {
        try {
            await fetch('/stop_stream');
            
            if (detectionListener) {
                detectionListener.close();
                detectionListener = null;
            }
            
            const videoContainer = document.getElementById('videoContainer');
            videoContainer.innerHTML = `
                <div class="video-placeholder">
                    <h3>⏹ Detection Stopped</h3>
                    <p>Click "Start Detection" to resume</p>
                    <p>🔊 Audio alerts can be toggled using the Voice button</p>
                </div>
            `;
            
            document.getElementById('streamStatus').textContent = 'Stopped';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            showStatus('⏹ Detection stopped', 'info');
            
            if (streamCheckInterval) {
                clearInterval(streamCheckInterval);
                streamCheckInterval = null;
            }
        } catch (error) {
            showStatus('❌ Error stopping stream: ' + error.message, 'error');
        }
    }
    
    function startStatusCheck() {
        if (streamCheckInterval) {
            clearInterval(streamCheckInterval);
        }
        
        streamCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/stream_status');
                const status = await response.json();
                
                if (!status.is_streaming) {
                    document.getElementById('streamStatus').textContent = '🔴 Disconnected';
                    showStatus('⚠ Stream disconnected', 'error');
                    
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    
                    if (detectionListener) {
                        detectionListener.close();
                        detectionListener = null;
                    }
                    
                    clearInterval(streamCheckInterval);
                    streamCheckInterval = null;
                }
            } catch (error) {
                console.error('Status check error:', error);
            }
        }, 3000);
    }
    
    document.getElementById('cameraUrl').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            startDetection();
        }
    });
    
    // ⭐ AUTO-START IN TEST MODE ⭐
    window.addEventListener('load', function() {
        fetch('/stream_status')
            .then(response => response.json())
            .then(data => {
                if (data.test_mode) {
                    console.log('✅ Test mode detected, auto-starting...');
                    
                    // Hide camera input fields in test mode
                    document.querySelector('.input-group').style.display = 'none';
                    document.querySelector('.example-urls').style.display = 'none';
                    
                    // Show test mode indicator
                    const subtitle = document.querySelector('.subtitle');
                    subtitle.innerHTML = '🎬 <strong>TEST MODE</strong> - Using local video file with audio alerts (toggle voice on/off)';
                    subtitle.style.color = '#dc3545';
                    
                    // Auto-start detection after 1 second
                    setTimeout(() => {
                        startDetection();
                    }, 1000);
                }
            })
            .catch(error => console.error('Error checking test mode:', error));
    });
</script>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info("🚀 Starting ESP32-CAM YOLOv8 Detection Server")
    logger.info(f"🌐 Open in browser: http://localhost:{port}")
    
    if USE_TEST_VIDEO:
        logger.info(f"🎬 [BACKEND TEST MODE] Using video file: {TEST_VIDEO_PATH}")
        logger.info("💡 To use camera input, set USE_TEST_VIDEO = False")
    else:
        logger.info("📷 Production mode: Users can enter camera URL via web interface")
    
    uvicorn.run(app, host=host, port=port, log_level=log_level)