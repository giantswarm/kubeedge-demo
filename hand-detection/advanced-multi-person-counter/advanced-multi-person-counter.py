import cv2
import mediapipe as mp
import time
import math
import json
from os import getenv
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("Warning: paho-mqtt not installed. MQTT publishing disabled.")
    MQTT_AVAILABLE = False

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Variables for tracking
current_raised_hands = 0
last_count_display = time.time()
frame_count = 0

# Add this at the top with other variables
hand_count_history = []
SMOOTHING_WINDOW = 3

def load_config(config_file=None):
    """Load configuration from JSON file - fail if invalid"""
    # Use environment variable if set, otherwise use default
    if config_file is None:
        config_file = getenv("CONFIG_PATH", "config.json")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from: {config_file}")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Required config file '{config_file}' not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file '{config_file}': {e}")
    
class MQTTPublisher:
    def __init__(self, config):
        self.broker_host = config["mqtt"]["broker_host"]
        self.broker_port = config["mqtt"]["broker_port"]
        self.topic = config["mqtt"]["topic"]
        self.timeout = config["mqtt"]["timeout"]
        self.client = None
        self.connected = False
        
        if MQTT_AVAILABLE:
            self.setup_mqtt()
    
    def setup_mqtt(self):
        """Initialize MQTT client"""
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
        try:
            print(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, self.timeout)
            self.client.loop_start()
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Continuing without MQTT...")
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"Connected to MQTT broker. Publishing to topic: {self.topic}")
        else:
            print(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("Disconnected from MQTT broker")
    
    def publish(self, hand_count):
        """Publish hand count data to MQTT topic"""
        if not MQTT_AVAILABLE or not self.connected:
            return
        
        # Create message payload in the required format
        payload = {
            "handsraised": str(hand_count),
            "status": "online"
        }
        
        try:
            self.client.publish(self.topic, json.dumps(payload))
            print(f"Published to MQTT: {payload}")
        except Exception as e:
            print(f"Failed to publish to MQTT: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()

def is_hand_raised_by_position(hand_landmarks, image_height):
    """More strict hand detection"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # More strict thresholds
    wrist_raised = wrist.y < 0.55  # Changed from 0.65 to 0.55 (higher position required)
    hand_upright = middle_finger_tip.y < (wrist.y - 0.1)  # Bigger gap required
    
    return wrist_raised and hand_upright

def estimate_shoulder_height(pose_landmarks):
    """
    Estimate average shoulder height from pose landmarks
    """
    if pose_landmarks:
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return (left_shoulder.y + right_shoulder.y) / 2
    return 0.5  # Default middle of frame

def is_hand_above_shoulders(hand_landmarks, shoulder_height):
    """Add margin to shoulder detection"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    # Add 5% margin above shoulders
    return wrist.y < (shoulder_height - 0.05)

def count_raised_hands_comprehensive(hand_results, pose_results, image_height):
    """
    Count raised hands using multiple criteria for better accuracy
    """
    raised_count = 0
    
    if not hand_results.multi_hand_landmarks:
        return 0
    
    # Get shoulder height estimate from pose detection
    shoulder_height = estimate_shoulder_height(pose_results.pose_landmarks) if pose_results.pose_landmarks else 0.5
    
    # Analyze each detected hand
    for hand_landmarks in hand_results.multi_hand_landmarks:
        # Use multiple criteria
        position_raised = is_hand_raised_by_position(hand_landmarks, image_height)
        above_shoulders = is_hand_above_shoulders(hand_landmarks, shoulder_height)
        
        # Hand is considered raised if it meets both criteria
        if position_raised and above_shoulders:
            raised_count += 1
    
    return raised_count

def count_raised_hands_simple(hand_results, image_height):
    """
    Simple method: count hands in upper portion of frame
    Fallback when pose detection fails
    """
    raised_count = 0
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if is_hand_raised_by_position(hand_landmarks, image_height):
                raised_count += 1
    
    return raised_count

def count_raised_hands_with_confidence(hand_results, pose_results, image_height):
    """Count hands with confidence scoring"""
    raised_count = 0
    
    if not hand_results.multi_hand_landmarks:
        return 0
    
    shoulder_height = estimate_shoulder_height(pose_results.pose_landmarks) if pose_results.pose_landmarks else 0.5
    
    for hand_landmarks in hand_results.multi_hand_landmarks:
        confidence_score = 0
        
        # Test 1: Position-based detection
        if is_hand_raised_by_position(hand_landmarks, image_height):
            confidence_score += 1
        
        # Test 2: Above shoulders
        if is_hand_above_shoulders(hand_landmarks, shoulder_height):
            confidence_score += 1
        
        # Test 3: Hand orientation (fingers pointing up)
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        if middle_tip.y < wrist.y - 0.15:  # Strong upward orientation
            confidence_score += 1
        
        # Only count if passes at least 2 out of 3 tests
        if confidence_score >= 2:
            raised_count += 1
    
    return raised_count

class HandCountSmoother:
    def __init__(self, window_size=5, threshold=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
    
    def smooth_count(self, current_count):
        # Add current count to history
        self.history.append(current_count)
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Count frequency of each value
        from collections import Counter
        counts = Counter(self.history)
        most_common = counts.most_common(1)[0]
        
        # Return most common count if it appears enough times
        if len(self.history) >= 3 and most_common[1] / len(self.history) >= self.threshold:
            return most_common[0]
        
        # Otherwise return current count
        return current_count

# Load configuration
config = load_config()

# For webcam input with optimized settings:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['resolution']['width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['resolution']['height'])
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize MQTT and set visual display (default: False)
show_visual = config.get("visual_display", False)
mqtt_publisher = MQTTPublisher(config)

print("Advanced Multi-Person Hand Counter Started!")
print("Detects raised hands from multiple people with improved accuracy")
print(f"Visual display: {'ON' if show_visual else 'OFF'}")
print(f"Publishes to MQTT broker: {config['mqtt']['broker_host']}:{config['mqtt']['broker_port']}")
print(f"Topic: {config['mqtt']['topic']}")
print("Press 'q' to quit, 's' to toggle visual display")

# Initialize both hands and pose detection
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=12,
    min_detection_confidence=0.8,  # Increased from 0.6
    min_tracking_confidence=0.8    # Increased from 0.6
) as hands, \
mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,  # Increased from 0.5  
    min_tracking_confidence=0.7,   # Increased from 0.5
    model_complexity=1             # Use full model instead of lite
) as pose:

    smoother = HandCountSmoother(window_size=5, threshold=0.6)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with both models
        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        
        # Store previous count for comparison
        previous_count = current_raised_hands
        
        # Count raised hands using comprehensive method
        image_height, image_width = image.shape[:2]
        
        if pose_results.pose_landmarks:
            # Use comprehensive method when pose is detected
            raw_count = count_raised_hands_comprehensive(hand_results, pose_results, image_height)
            smoothed_count = smoother.smooth_count(raw_count)
            current_raised_hands = smoothed_count
        else:
            # Fallback to simple method
            current_raised_hands = count_raised_hands_simple(hand_results, image_height)
        
        # Print count immediately when it changes
        if current_raised_hands != previous_count:
            print(f"Hands raised: {current_raised_hands}")
            # Publish to MQTT when count changes
            mqtt_publisher.publish(current_raised_hands)
        
        # Visual feedback
        if show_visual and hand_results.multi_hand_landmarks:
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Get shoulder height for visual reference
            shoulder_height = estimate_shoulder_height(pose_results.pose_landmarks) if pose_results.pose_landmarks else 0.5
            
            # Draw shoulder line
            shoulder_y = int(shoulder_height * image_height)
            cv2.line(image_bgr, (0, shoulder_y), (image_width, shoulder_y), (255, 255, 0), 2)
            cv2.putText(image_bgr, "Shoulder Line", (10, shoulder_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw hand landmarks and status
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check if this hand is raised
                position_raised = is_hand_raised_by_position(hand_landmarks, image_height)
                above_shoulders = is_hand_above_shoulders(hand_landmarks, shoulder_height)
                is_raised = position_raised and above_shoulders
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw status circle at wrist
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * image_width)
                wrist_y = int(wrist.y * image_height)
                
                color = (0, 255, 0) if is_raised else (0, 0, 255)  # Green if raised, red if not
                cv2.circle(image_bgr, (wrist_x, wrist_y), 10, color, -1)
            
            # Draw pose landmarks if available
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display count on image
            cv2.putText(image_bgr, f"Raised Hands: {current_raised_hands}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Advanced Multi-Person Hand Counter', image_bgr)
        
        # Display current count every 5 seconds
        current_time = time.time()
        if current_time - last_count_display >= 5:
            print(f"[Status] Current hands raised: {current_raised_hands}")
            if hand_results.multi_hand_landmarks:
                print(f"[Info] Detected {len(hand_results.multi_hand_landmarks)} total hands")
            last_count_display = current_time
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            show_visual = not show_visual
            if not show_visual:
                cv2.destroyAllWindows()
            print(f"Visual display: {'ON' if show_visual else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
mqtt_publisher.disconnect()
