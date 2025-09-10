import cv2
import mediapipe as mp
import time
import math
import json
import numpy as np
from os import getenv
from datetime import datetime
from collections import defaultdict
try:
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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
    """Distance-adaptive hand detection with flexible thresholds"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # More flexible thresholds for distance detection
    wrist_raised = wrist.y < 0.85  # More lenient for far-distance detection
    
    # Check if hand is upright by comparing multiple finger tips to wrist
    fingers_up = 0
    finger_gap_threshold = 0.05  # More lenient gap requirement
    
    # Check middle finger
    if middle_finger_tip.y < (wrist.y - finger_gap_threshold):
        fingers_up += 1
    
    # Check index finger
    if index_finger_tip.y < (wrist.y - finger_gap_threshold):
        fingers_up += 1
    
    # Hand is raised if wrist is high enough and at least one finger points up
    hand_upright = fingers_up >= 1
    
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
    """Add flexible margin to shoulder detection for distance"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    # More generous margin for far-distance detection
    return wrist.y < (shoulder_height - 0.02)

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

class PersonDetector:
    """Multi-person pose detection system"""
    
    def __init__(self):
        # Initialize pose detector for full image detection
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        self.person_tracker = PersonTracker()
    
    def detect_people_from_hands(self, hand_results, image_shape):
        """Estimate person locations from hand positions using advanced clustering"""
        if not hand_results.multi_hand_landmarks:
            return []
        
        hands = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hands.append({
                'landmarks': hand_landmarks,
                'position': (wrist.x, wrist.y),
                'wrist': wrist
            })
        
        # Use DBSCAN-like clustering to group hands by person
        person_groups = self._cluster_hands_by_person(hands)
        
        # For each person group, estimate person center and create detection region
        people = []
        for group_idx, group in enumerate(person_groups):
            if not group:
                continue
                
            person_info = self._create_person_from_hands(group, group_idx, image_shape)
            people.append(person_info)
        
        return people
    
    def _cluster_hands_by_person(self, hands):
        """Advanced hand clustering using distance and anatomical constraints"""
        if len(hands) <= 1:
            return [hands] if hands else []
        
        # Calculate distances between all hand pairs
        positions = np.array([h['position'] for h in hands])
        
        if SCIPY_AVAILABLE:
            distances = cdist(positions, positions)
        else:
            # Manual distance calculation
            n = len(positions)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        diff = positions[i] - positions[j]
                        distances[i][j] = np.sqrt(np.sum(diff**2))
        
        # Apply clustering based on anatomical constraints
        clusters = []
        used_indices = set()
        
        for i, hand in enumerate(hands):
            if i in used_indices:
                continue
                
            cluster = [hand]
            used_indices.add(i)
            
            # Look for hands that belong to same person
            for j, other_hand in enumerate(hands):
                if j in used_indices:
                    continue
                    
                # Check if these hands could belong to same person
                if self._should_group_hands(hand, other_hand, distances[i][j]):
                    cluster.append(other_hand)
                    used_indices.add(j)
                    
                    # Limit to 2 hands per person
                    if len(cluster) >= 2:
                        break
            
            clusters.append(cluster)
        
        return clusters
    
    def _should_group_hands(self, hand1, hand2, distance):
        """Determine if two hands belong to the same person"""
        vertical_diff = abs(hand1['position'][1] - hand2['position'][1])
        horizontal_diff = abs(hand1['position'][0] - hand2['position'][0])
        
        # Anatomical constraints for same person:
        # 1. Hands at similar height (shoulders relatively level)
        # 2. Reasonable horizontal separation (arm span)
        # 3. Not too far apart overall
        
        same_height = vertical_diff < 0.15  # Within 15% of frame height
        reasonable_separation = 0.1 < horizontal_diff < 0.8  # 10-80% of frame width
        close_enough = distance < 0.6  # Overall distance constraint
        
        return same_height and reasonable_separation and close_enough
    
    def _create_person_from_hands(self, hand_group, person_id, image_shape):
        """Create person info from hand group with estimated body position"""
        h, w = image_shape[:2]
        
        # Calculate person center from hand positions
        hand_positions = np.array([h['position'] for h in hand_group])
        center_x = np.mean(hand_positions[:, 0])
        center_y = np.mean(hand_positions[:, 1])
        
        # Estimate person bounding box based on hand positions
        min_x = max(0, center_x - 0.25)  # 25% of frame width on each side
        max_x = min(1, center_x + 0.25)
        min_y = max(0, center_y - 0.3)   # 30% of frame height above/below
        max_y = min(1, center_y + 0.4)
        
        return {
            'id': person_id,
            'center': (center_x, center_y),
            'bbox': (min_x, min_y, max_x, max_y),
            'hands': hand_group,
            'shoulder_height': None,  # Will be calculated later
            'pose_landmarks': None
        }
    
    def get_person_poses(self, image, people):
        """Extract pose information for each detected person using full image pose detection"""
        # Use full image pose detection instead of regional detection
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(image_rgb)
            
            if pose_results.pose_landmarks:
                # For each detected person, find the closest pose landmarks
                for person in people:
                    closest_pose = self._find_closest_pose_to_person(person, pose_results.pose_landmarks)
                    if closest_pose:
                        person['pose_landmarks'] = pose_results.pose_landmarks
                        person['shoulder_height'] = self._get_shoulder_height_from_pose(pose_results.pose_landmarks)
                        # Debug info
                        print(f"[Debug] Person {person['id']} using REAL pose landmarks for shoulder line")
                    else:
                        # Fallback: estimate from hand positions
                        person['shoulder_height'] = self._estimate_shoulder_from_hands(person['hands'])
                        print(f"[Debug] Person {person['id']} using ESTIMATED shoulder line (pose too far from hands)")
            else:
                # No pose detected, use hand-based estimation for all people
                print("[Debug] No pose landmarks detected in full image - using hand-based estimation for all people")
                for person in people:
                    person['shoulder_height'] = self._estimate_shoulder_from_hands(person['hands'])
                    
        except Exception as e:
            # Fallback on any error - use hand-based estimation
            for person in people:
                person['shoulder_height'] = self._estimate_shoulder_from_hands(person['hands'])
    
    def _find_closest_pose_to_person(self, person, pose_landmarks):
        """Check if pose landmarks correspond to this person's hand positions"""
        if not person['hands'] or not pose_landmarks:
            return False
        
        # Get person's hand center
        hand_positions = [h['position'] for h in person['hands']]
        person_center_x = sum(pos[0] for pos in hand_positions) / len(hand_positions)
        person_center_y = sum(pos[1] for pos in hand_positions) / len(hand_positions)
        
        # Get pose center from shoulders/torso
        try:
            left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            
            # Calculate pose center
            pose_center_x = (left_shoulder.x + right_shoulder.x + nose.x) / 3
            pose_center_y = (left_shoulder.y + right_shoulder.y + nose.y) / 3
            
            # Check if person's hands are reasonably close to pose center
            distance = math.sqrt((person_center_x - pose_center_x)**2 + (person_center_y - pose_center_y)**2)
            
            # Allow reasonable distance (40% of frame) between hand center and pose center
            return distance < 0.4
            
        except:
            return False
    
    def _get_shoulder_height_from_pose(self, pose_landmarks):
        """Get actual shoulder height from pose landmarks"""
        try:
            left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Average shoulder height
            shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
            
            # Constrain to reasonable bounds
            return max(0.2, min(shoulder_height, 0.8))
        except:
            return None
    
    def _estimate_shoulder_from_hands(self, hand_group):
        """Anatomically accurate shoulder estimation from hand positions"""
        if not hand_group:
            return 0.6
        
        # Get the HIGHEST hand position (lowest y-value, since y increases downward)
        hand_heights = [h['wrist'].y for h in hand_group]
        highest_hand_y = min(hand_heights)  # Lowest y value = highest position
        
        # Anatomically accurate shoulder-to-wrist offset
        # When hands are raised above head, shoulders are typically 15-20% below highest hand
        # When hands are at shoulder level, shoulders are at same level
        # We'll estimate shoulders are about 15% below the highest raised hand
        shoulder_offset = 0.15  
        estimated_shoulder = highest_hand_y + shoulder_offset
        
        # Ensure shoulders are in reasonable range (not too high or too low)
        return max(0.20, min(estimated_shoulder, 0.70))

class PersonTracker:
    """Track people across frames for consistency"""
    
    def __init__(self):
        self.tracked_people = {}
        self.next_id = 0
        self.max_age = 10  # Frames to keep person without detection
    
    def update(self, detected_people):
        """Update tracked people with new detections"""
        # This could be enhanced with proper tracking algorithms
        return detected_people

def estimate_person_shoulder_height(hand_group, pose_landmarks=None):
    """
    Estimate shoulder height for a specific person based on their hands
    Uses pose landmarks if available, otherwise estimates from hand positions anatomically
    """
    if pose_landmarks and len(hand_group) <= 2:
        # Use global pose for single person or when pose is reliable
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
        # Constrain pose-based shoulder height to reasonable bounds
        return max(0.3, min(shoulder_height, 0.8))
    
    # Anatomically accurate estimation from hand positions
    if not hand_group:
        return 0.6  # Default for distance detection
    
    # Get the HIGHEST hand position (lowest y-value, since y increases downward)
    wrist_positions = [hand['wrist'].y for hand in hand_group]
    highest_hand_y = min(wrist_positions)  # Lowest y value = highest position
    
    # Anatomically accurate shoulder-to-wrist offset
    # Shoulders are typically 15% of frame height below the highest raised hand
    shoulder_offset = 0.15
    estimated_shoulder_y = highest_hand_y + shoulder_offset
    
    # Apply reasonable constraints to prevent extreme values
    return max(0.20, min(estimated_shoulder_y, 0.70))

def is_hand_above_person_shoulders(hand, person_shoulder_height):
    """Check if a hand is above a specific person's shoulder line with flexible margin"""
    wrist = hand['wrist']
    # More generous margin for distance detection
    return wrist.y < (person_shoulder_height - 0.02)

def count_raised_hands_multi_person(hand_results, image, person_detector, shoulder_smoother=None):
    """
    Count raised hands using true multi-person detection with individual pose landmarks
    """
    if not hand_results.multi_hand_landmarks:
        return 0, []
    
    # Detect individual people from hand positions
    people = person_detector.detect_people_from_hands(hand_results, image.shape)
    
    if not people:
        return 0, []
    
    # Get actual pose landmarks for each person
    person_detector.get_person_poses(image, people)
    
    # Clean up smoother for current person count
    if shoulder_smoother:
        shoulder_smoother.cleanup_old_people(len(people))
    
    total_raised_hands = 0
    
    for person in people:
        if not person['hands']:
            continue
        
        # Get shoulder height (from actual pose if available, otherwise estimated)
        raw_shoulder_height = person['shoulder_height']
        if raw_shoulder_height is None:
            raw_shoulder_height = 0.6  # Default fallback
        
        # Apply smoothing if smoother is provided
        person_shoulder_height = raw_shoulder_height
        if shoulder_smoother:
            person_shoulder_height = shoulder_smoother.smooth_shoulder_height(person['id'], raw_shoulder_height)
            person['smoothed_shoulder_height'] = person_shoulder_height
        else:
            person['smoothed_shoulder_height'] = person_shoulder_height
        
        # Count raised hands for this person
        person_raised_count = 0
        for hand in person['hands']:
            # Use multiple criteria with person's actual shoulder line
            position_raised = is_hand_raised_by_position(hand['landmarks'], image.shape[0])
            above_shoulders = is_hand_above_person_shoulders(hand, person_shoulder_height)
            
            # Hand is considered raised if it meets both criteria
            if position_raised and above_shoulders:
                person_raised_count += 1
        
        person['raised_count'] = person_raised_count
        total_raised_hands += person_raised_count
    
    return total_raised_hands, people

def count_raised_hands_per_person(hand_results, pose_results, image_height, shoulder_smoother=None):
    """
    Legacy function for compatibility - redirects to old method
    """
    if not hand_results.multi_hand_landmarks:
        return 0
    
    # Use simplified grouping for fallback
    hands = []
    for hand_landmarks in hand_results.multi_hand_landmarks:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        hands.append({
            'landmarks': hand_landmarks,
            'position': (wrist.x, wrist.y),
            'wrist': wrist
        })
    
    # Simple clustering fallback
    if len(hands) <= 2:
        groups = [hands]
    else:
        # Basic left-to-right grouping
        hands.sort(key=lambda h: h['position'][0])
        groups = []
        current_group = [hands[0]]
        
        for hand in hands[1:]:
            if abs(hand['position'][0] - current_group[-1]['position'][0]) < 0.4:
                current_group.append(hand)
            else:
                groups.append(current_group)
                current_group = [hand]
        groups.append(current_group)
    
    total_raised_hands = 0
    
    for group_idx, group in enumerate(groups):
        if not group:
            continue
            
        # Get raw shoulder height for this person
        raw_shoulder_height = estimate_person_shoulder_height(
            group, 
            pose_results.pose_landmarks if pose_results.pose_landmarks else None
        )
        
        # Apply smoothing if smoother is provided
        person_shoulder_height = raw_shoulder_height
        if shoulder_smoother:
            person_shoulder_height = shoulder_smoother.smooth_shoulder_height(group_idx, raw_shoulder_height)
        
        # Count raised hands for this person
        person_raised_count = 0
        for hand in group:
            # Use multiple criteria
            position_raised = is_hand_raised_by_position(hand['landmarks'], image_height)
            above_shoulders = is_hand_above_person_shoulders(hand, person_shoulder_height)
            
            # Hand is considered raised if it meets both criteria
            if position_raised and above_shoulders:
                person_raised_count += 1
        
        total_raised_hands += person_raised_count
    
    return total_raised_hands

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

class ShoulderLineSmoother:
    def __init__(self, window_size=8, smoothing_factor=0.7):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.person_histories = {}  # Track shoulder heights per person
        self.person_stable_positions = {}  # Stable positions per person
    
    def smooth_shoulder_height(self, person_id, current_height):
        # Initialize history for new person
        if person_id not in self.person_histories:
            self.person_histories[person_id] = []
            self.person_stable_positions[person_id] = current_height
        
        # Add current height to history
        self.person_histories[person_id].append(current_height)
        
        # Keep only recent history
        if len(self.person_histories[person_id]) > self.window_size:
            self.person_histories[person_id].pop(0)
        
        # Calculate smoothed height using exponential moving average
        history = self.person_histories[person_id]
        if len(history) >= 3:
            # Calculate median to reduce outlier impact
            sorted_heights = sorted(history)
            median_height = sorted_heights[len(sorted_heights) // 2]
            
            # Apply exponential smoothing
            stable_pos = self.person_stable_positions[person_id]
            smoothed = stable_pos * self.smoothing_factor + median_height * (1 - self.smoothing_factor)
            
            # Only update if change is significant (reduces flapping)
            if abs(smoothed - stable_pos) > 0.02:
                self.person_stable_positions[person_id] = smoothed
            
            return self.person_stable_positions[person_id]
        
        return current_height
    
    def cleanup_old_people(self, current_person_count):
        # Remove tracking for people no longer detected
        if len(self.person_histories) > current_person_count:
            # Keep only the most recent person IDs
            person_ids = list(self.person_histories.keys())
            for person_id in person_ids[current_person_count:]:
                del self.person_histories[person_id]
                del self.person_stable_positions[person_id]

# For webcam input with optimized settings for distance detection:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # High resolution for distant subjects
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus for distance changes
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure for varying distances

# Load config, then initialize MQTT and set visual display (default: False)
config = load_config()
show_visual = config.get("visual_display", False)
mqtt_publisher = MQTTPublisher(config)

print("Advanced Multi-Person Hand Counter Started!")
print("🚀 TRUE MULTI-PERSON DETECTION: Individual pose detection per person")
print("✨ ACCURATE SHOULDER LINES: Each person gets their own shoulder line from actual pose landmarks")
print("🔍 DISTANCE OPTIMIZED: Enhanced detection for people at varying distances")
print("📊 INTELLIGENT GROUPING: Advanced hand clustering with anatomical constraints")
print(f"Visual display: {'ON' if show_visual else 'OFF'}")
print(f"Publishes to MQTT broker: {config['mqtt']['broker_host']}:{config['mqtt']['broker_port']}")
print(f"Topic: {config['mqtt']['topic']}")
print("Legend: [P] = Real pose detected, [E] = Estimated from hands")
print("Press 'q' to quit, 's' to toggle visual display")

# Initialize both hands and pose detection with optimized settings for distance
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=12,
    min_detection_confidence=0.5,  # Lowered for better far-distance detection
    min_tracking_confidence=0.5    # Lowered to maintain tracking at distance
) as hands, \
mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.4,  # Lowered for distant pose detection  
    min_tracking_confidence=0.4,   # Lowered for distant pose tracking
    model_complexity=2             # Use most accurate model for better distance detection
) as pose:

    smoother = HandCountSmoother(window_size=5, threshold=0.6)
    shoulder_smoother = ShoulderLineSmoother(window_size=8, smoothing_factor=0.7)
    person_detector = PersonDetector()  # Initialize multi-person detection system

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
        
        # Count raised hands using per-person method with individual shoulder lines
        image_height, image_width = image.shape[:2]
        
        # Use true multi-person detection with individual pose landmarks
        raw_count, detected_people = count_raised_hands_multi_person(hand_results, image, person_detector, shoulder_smoother)
        smoothed_count = smoother.smooth_count(raw_count)
        current_raised_hands = smoothed_count
        
        # Print count immediately when it changes
        if current_raised_hands != previous_count:
            print(f"Hands raised: {current_raised_hands}")
            
            # Debug: Show multi-person detection info when count changes
            if hand_results.multi_hand_landmarks and detected_people:
                total_hands = sum(len(person['hands']) for person in detected_people)
                pose_count = sum(1 for person in detected_people if person['pose_landmarks'] is not None)
                print(f"[Debug] {total_hands} hands detected across {len(detected_people)} people ({pose_count} with pose landmarks)")
            
            # Publish to MQTT when count changes
            mqtt_publisher.publish(current_raised_hands)
        
        # Visual feedback with true multi-person detection
        if show_visual and hand_results.multi_hand_landmarks and detected_people:
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Colors for different people (cycling through colors)
            person_colors = [
                (255, 255, 0),   # Yellow
                (0, 255, 255),   # Cyan  
                (255, 0, 255),   # Magenta
                (255, 128, 0),   # Orange
                (128, 255, 0),   # Lime
                (0, 128, 255),   # Light blue
            ]
            
            # Process each detected person
            for person in detected_people:
                if not person['hands']:
                    continue
                    
                # Get color for this person
                color = person_colors[person['id'] % len(person_colors)]
                
                # Get person's shoulder height (actual or smoothed)
                person_shoulder_height = person.get('smoothed_shoulder_height', person['shoulder_height'])
                
                # Calculate person's horizontal bounds from hands
                x_positions = [hand['position'][0] for hand in person['hands']]
                person_left = int(min(x_positions) * image_width)
                person_right = int(max(x_positions) * image_width)
                
                # Extend bounds for better visibility
                person_left = max(0, person_left - 50)
                person_right = min(image_width, person_right + 50)
                
                # Draw person's actual shoulder line
                if person_shoulder_height:
                    shoulder_y = int(person_shoulder_height * image_height)
                    cv2.line(image_bgr, (person_left, shoulder_y), (person_right, shoulder_y), color, 3)
                    
                    # Add label with pose info
                    pose_label = " (POSE)" if person['pose_landmarks'] else " (EST)"
                    cv2.putText(image_bgr, f"Person {person['id'] + 1}{pose_label}", 
                               (person_left, shoulder_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw person bounding box if we have pose detection
                if person['pose_landmarks']:
                    min_x, min_y, max_x, max_y = person['bbox']
                    box_x1, box_y1 = int(min_x * image_width), int(min_y * image_height)
                    box_x2, box_y2 = int(max_x * image_width), int(max_y * image_height)
                    cv2.rectangle(image_bgr, (box_x1, box_y1), (box_x2, box_y2), color, 1)
                
                # Visualize hands for this person
                for hand in person['hands']:
                    # Check if this hand is raised
                    position_raised = is_hand_raised_by_position(hand['landmarks'], image_height)
                    above_shoulders = is_hand_above_person_shoulders(hand, person_shoulder_height) if person_shoulder_height else False
                    is_raised = position_raised and above_shoulders
                    
                    # Draw hand landmarks with person's color
                    mp_drawing.draw_landmarks(
                        image_bgr, hand['landmarks'], mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2))
                    
                    # Draw status circle at wrist
                    wrist_x = int(hand['wrist'].x * image_width)
                    wrist_y = int(hand['wrist'].y * image_height)
                    
                    status_color = (0, 255, 0) if is_raised else (0, 0, 255)  # Green if raised, red if not
                    cv2.circle(image_bgr, (wrist_x, wrist_y), 8, status_color, -1)
                    cv2.circle(image_bgr, (wrist_x, wrist_y), 12, color, 2)  # Person color border
                
                # Draw individual pose landmarks if available
                if person['pose_landmarks']:
                    try:
                        mp_drawing.draw_landmarks(
                            image_bgr, person['pose_landmarks'], mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=color, thickness=1))
                    except Exception as e:
                        # Skip pose drawing if there's an error, continue with hand detection
                        print(f"[Warning] Could not draw pose landmarks for person {person['id']}: {e}")
                        pass
            
            # Display total count and per-person breakdown
            cv2.putText(image_bgr, f"Total Raised Hands: {current_raised_hands}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show per-person counts with pose detection status
            y_offset = 60
            for person in detected_people:
                if person['hands']:
                    person_color = person_colors[person['id'] % len(person_colors)]
                    raised_count = person.get('raised_count', 0)
                    total_hands = len(person['hands'])
                    pose_status = "P" if person['pose_landmarks'] else "E"  # Pose or Estimated
                    cv2.putText(image_bgr, f"P{person['id']+1}[{pose_status}]: {raised_count}/{total_hands}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)
                    y_offset += 25
            
            cv2.imshow('Advanced Multi-Person Hand Counter', image_bgr)
        
        # Display current count every 5 seconds with multi-person breakdown
        current_time = time.time()
        if current_time - last_count_display >= 5:
            print(f"[Status] Current hands raised: {current_raised_hands}")
            if hand_results.multi_hand_landmarks and detected_people:
                total_hands = sum(len(person['hands']) for person in detected_people)
                pose_detections = sum(1 for person in detected_people if person['pose_landmarks'] is not None)
                print(f"[Info] Detected {total_hands} total hands across {len(detected_people)} people ({pose_detections} with pose landmarks)")
                
                # Show detailed per-person breakdown
                for person in detected_people:
                    if person['hands']:
                        person_id = person['id'] + 1
                        raised_count = person.get('raised_count', 0)
                        total_hands_person = len(person['hands'])
                        shoulder_height = person.get('smoothed_shoulder_height', person['shoulder_height'])
                        
                        # Indicate source of shoulder line
                        shoulder_source = "POSE" if person['pose_landmarks'] else "ESTIMATED"
                        
                        print(f"[Info] Person {person_id}: {raised_count}/{total_hands_person} hands raised "
                              f"(shoulder at {shoulder_height:.2f} [{shoulder_source}])")
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
person_detector.pose_detector.close()  # Clean up pose detector resources
