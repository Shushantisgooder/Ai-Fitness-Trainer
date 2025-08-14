import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
from collections import deque

# CONFIGURATION - Update these paths
MODEL_PATH = "C:/Users/Shushant/Documents/GitHub/Ai-Fitness-Coach/testing/models/model-2.pt"  # Your trained model
TEST_IMAGE = "testing-material/squat-8.jpg"  # Test image path
TEST_VIDEO = "testing-material/Side-Curl-Rana.mp4"  # Test video path

# Output directory
OUTPUT_DIR = Path("runs/fitness_trainer_test")


class ReadyPositionDetector:
    """
    Detects when user is in ready position and stable before starting exercise tracking
    Enhanced with velocity-based stability detection
    """
    
    def __init__(self, exercise_type="dumbell-curl", velocity_tracker=None):
        self.exercise_type = exercise_type
        self.is_ready = False
        self.is_stable = False
        self.tracking_active = False
        self.velocity_tracker = velocity_tracker  # Reference to VelocityTracker instance
        
        # Ready position criteria for different exercises
        self.ready_position_criteria = {
            "dumbell-curl": {
                "primary_angles": {
                    "left_elbow": {"target": 180, "tolerance": 15},    # Nearly straight
                    "right_elbow": {"target": 180, "tolerance": 15}    # Nearly straight
                },
                "stability_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
                "min_stable_frames": 6,        # Must be stable for 1 second at 30fps
                "angle_variance_threshold": 10,   # Max angle variance for stability
                "min_confidence": 0.3,          # Minimum pose confidence
                
                # NEW: Velocity-based stability criteria
                "velocity_stability": {
                    "required_slow_joints": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
                    "min_slow_percentage": 0.50,  # 75% of tracked joints must be "slow"
                    "allow_medium_joints": ["left_wrist", "right_wrist",],  # These can be medium speed
                    "max_fast_joints": 0,  # No joints should be moving fast
                    "velocity_weight": 0.6,  # How much to weight velocity vs angle variance (0.6 = 60% velocity, 40% angles)
                }
            }
        }
        
        # Get current exercise config
        self.config = self.ready_position_criteria.get(exercise_type, 
                                                      self.ready_position_criteria["dumbell-curl"])
        
        # Tracking variables
        self.angle_history = {}
        self.velocity_stability_history = []  # NEW: Track velocity stability over time
        self.stability_counter = 0
        self.position_check_history = []
        self.confidence_history = []
        
        print(f"Ready Position Detector initialized for {exercise_type}")
        if velocity_tracker:
            print("Velocity-based stability detection ENABLED")
        else:
            print("WARNING: No velocity tracker provided - using angle-only stability")
        print(f"Waiting for user to assume ready position...")
    
    def check_ready_position(self, angles, keypoints, confidences, frame_count, velocities=None):
        """
        Check if user is in ready position and stable
        ENHANCED: Now uses velocity data for improved stability detection
        
        Args:
            angles: Dictionary of current body angles
            keypoints: Current keypoint positions
            confidences: Keypoint confidence scores
            frame_count: Current frame number
            velocities: Optional velocity data (will try to get from velocity_tracker if not provided)
            
        Returns:
            dict: Status information about ready position detection
        """
        # Get velocity data if not provided
        if velocities is None and self.velocity_tracker is not None:
            # Get the most recent velocity data from the tracker
            if hasattr(self.velocity_tracker, 'velocity_history') and self.velocity_tracker.velocity_history:
                velocities = self.velocity_tracker.velocity_history[-1]
        
        # Check minimum confidence first
        avg_confidence = np.mean([confidences[i] for i in [5, 6, 7, 8, 9, 10]])  # Arms
        
        if avg_confidence < self.config["min_confidence"]:
            return self._get_status("low_confidence", 
                                  f"Low pose confidence: {avg_confidence:.2f}")
        
        # Check if angles meet ready position criteria
        position_correct = self._check_angle_criteria(angles)
        
        if not position_correct:
            self.stability_counter = 0  # Reset stability counter
            self._clear_velocity_history()  # Reset velocity history
            return self._get_status("wrong_position", 
                                  "Move to ready position: arms extended down")
        
        # Position is correct, now check stability using both methods
        stability_status = self._check_hybrid_stability(angles, keypoints, velocities, frame_count)
        
        if stability_status["stable"]:
            self.stability_counter += 1
            
            if self.stability_counter >= self.config["min_stable_frames"]:
                # User is ready and stable!
                if not self.tracking_active:
                    self.tracking_active = True
                    print("🎯 READY POSITION DETECTED - TRACKING STARTED!")
                    print(f"Stability method: {stability_status['method']}")
                
                return self._get_status("ready", 
                                      f"Ready! Tracking active - {stability_status['details']}")
            else:
                frames_needed = self.config["min_stable_frames"] - self.stability_counter
                return self._get_status("stabilizing", 
                                      f"Hold steady... ({frames_needed} frames) - {stability_status['details']}")
        else:
            self.stability_counter = max(0, self.stability_counter - 2)  # Decay counter
            return self._get_status("unstable", 
                                  f"Too much movement - {stability_status['reason']}")
    
    def _check_angle_criteria(self, angles):
        """Check if current angles meet ready position criteria"""
        for angle_name, criteria in self.config["primary_angles"].items():
            if angle_name not in angles:
                return False
            
            current_angle = angles[angle_name]
            target = criteria["target"]
            tolerance = criteria["tolerance"]
            
            if not (target - tolerance <= current_angle <= target + tolerance):
                return False
        
        return True
    
    def _check_hybrid_stability(self, angles, keypoints, velocities, frame_count):
        """
        NEW: Check stability using both angle variance and velocity data
        Uses weighted combination for more robust detection
        """
        # Get traditional angle-based stability
        angle_stability = self._check_angle_stability(angles)
        
        # Get velocity-based stability if available
        velocity_stability = self._check_velocity_stability(velocities) if velocities else None
        
        # Determine stability using hybrid approach
        if velocity_stability is not None and self.velocity_tracker is not None:
            # Use weighted combination
            weight = self.config["velocity_stability"]["velocity_weight"]
            
            # Convert boolean to score (1.0 = stable, 0.0 = unstable)
            angle_score = 1.0 if angle_stability["stable"] else 0.0
            velocity_score = 1.0 if velocity_stability["stable"] else 0.0
            
            # Calculate combined stability score
            combined_score = (velocity_score * weight) + (angle_score * (1.0 - weight))
            
            # Require high confidence for stability
            is_stable = combined_score >= 0.8
            
            if is_stable:
                details = f"Hybrid: V({velocity_stability['score']:.1f}) + A({angle_score:.1f}) = {combined_score:.2f}"
                return {
                    "stable": True,
                    "method": "hybrid",
                    "details": details,
                    "velocity_data": velocity_stability,
                    "angle_data": angle_stability
                }
            else:
                # Determine primary reason for instability
                if velocity_score < 0.5:
                    reason = f"Movement too fast: {velocity_stability['reason']}"
                else:
                    reason = f"Angle variance: {angle_stability['reason']}"
                
                return {
                    "stable": False,
                    "method": "hybrid",
                    "reason": reason,
                    "score": combined_score
                }
        else:
            # Fall back to angle-only method
            return {
                "stable": angle_stability["stable"],
                "method": "angle_only",
                "details": f"Angle variance check: {angle_stability['reason']}",
                "reason": angle_stability["reason"] if not angle_stability["stable"] else ""
            }
    
    def _check_angle_stability(self, angles):
        """Traditional angle variance stability check"""
        # Add current angles to history
        for angle_name in self.config["primary_angles"]:
            if angle_name in angles:
                if angle_name not in self.angle_history:
                    self.angle_history[angle_name] = []
                
                self.angle_history[angle_name].append(angles[angle_name])
                
                # Maintain history window
                if len(self.angle_history[angle_name]) > 15:  # Last 15 frames
                    self.angle_history[angle_name].pop(0)
        
        # Check variance for each tracked angle
        for angle_name in self.config["primary_angles"]:
            if (angle_name in self.angle_history and 
                len(self.angle_history[angle_name]) >= 10):  # Need minimum history
                
                variance = np.var(self.angle_history[angle_name])
                if variance > self.config["angle_variance_threshold"]:
                    return {"stable": False, "reason": f"{angle_name} variance: {variance:.1f}"}
        
        return {"stable": True, "reason": "All angles stable"}
    
    def _check_velocity_stability(self, velocities):
        """
        NEW: Check stability based on velocity classifications
        Returns stability assessment based on how many joints are moving slowly
        """
        if not velocities:
            return {"stable": False, "reason": "No velocity data", "score": 0.0}
        
        config = self.config["velocity_stability"]
        
        # Count joint movements by speed category
        slow_joints = []
        medium_joints = []
        fast_joints = []
        
        # Check required slow joints
        for joint_name in config["required_slow_joints"]:
            if joint_name in velocities:
                speed_cat = velocities[joint_name].get('speed_category', 'unknown')
                if speed_cat == 'slow':
                    slow_joints.append(joint_name)
                elif speed_cat == 'medium':
                    medium_joints.append(joint_name)
                elif speed_cat == 'fast':
                    fast_joints.append(joint_name)
        
        # Check allowed medium joints
        for joint_name in config.get("allow_medium_joints", []):
            if joint_name in velocities:
                speed_cat = velocities[joint_name].get('speed_category', 'unknown')
                if speed_cat == 'medium':
                    medium_joints.append(joint_name)
                elif speed_cat == 'fast':
                    fast_joints.append(joint_name)
        
        # Calculate stability metrics
        total_required = len(config["required_slow_joints"])
        slow_percentage = len(slow_joints) / total_required if total_required > 0 else 0
        
        # Store in history for trend analysis
        velocity_stability_data = {
            "slow_joints": slow_joints,
            "medium_joints": medium_joints,
            "fast_joints": fast_joints,
            "slow_percentage": slow_percentage
        }
        
        self.velocity_stability_history.append(velocity_stability_data)
        if len(self.velocity_stability_history) > 10:  # Keep last 10 frames
            self.velocity_stability_history.pop(0)
        
        # Check stability criteria
        reasons = []
        
        # Must have enough slow joints
        if slow_percentage < config["min_slow_percentage"]:
            reasons.append(f"Only {slow_percentage:.1%} slow joints (need {config['min_slow_percentage']:.1%})")
        
        # Must not have too many fast joints
        if len(fast_joints) > config["max_fast_joints"]:
            reasons.append(f"{len(fast_joints)} fast joints (max {config['max_fast_joints']}): {fast_joints}")
        
        is_stable = len(reasons) == 0
        
        # Calculate stability score (0-1)
        score = min(1.0, slow_percentage) * (1.0 - min(1.0, len(fast_joints) * 0.5))
        
        return {
            "stable": is_stable,
            "score": score,
            "reason": "; ".join(reasons) if reasons else f"{len(slow_joints)}/{total_required} joints slow",
            "slow_joints": slow_joints,
            "medium_joints": medium_joints,
            "fast_joints": fast_joints,
            "slow_percentage": slow_percentage
        }
    
    def _clear_velocity_history(self):
        """Clear velocity stability history when position is wrong"""
        self.velocity_stability_history = []
    
    def _get_status(self, status_type, message):
        """Return standardized status dictionary with enhanced info"""
        return {
            "ready": status_type == "ready",
            "tracking_active": self.tracking_active,
            "status": status_type,
            "message": message,
            "stability_progress": min(100, (self.stability_counter / self.config["min_stable_frames"]) * 100),
            "has_velocity_tracker": self.velocity_tracker is not None,
            "detection_method": "hybrid" if self.velocity_tracker else "angle_only"
        }
    
    def reset(self):
        """Reset the ready position detector"""
        self.is_ready = False
        self.is_stable = False
        self.tracking_active = False
        self.angle_history = {}
        self.velocity_stability_history = []  # NEW: Reset velocity history
        self.stability_counter = 0
        self.position_check_history = []
        self.confidence_history = []
        print("Ready Position Detector reset (including velocity data)")
    
    def get_stability_debug_info(self):
        """
        NEW: Get detailed debugging information about stability detection
        Useful for tuning and troubleshooting
        """
        info = {
            "angle_history_length": {name: len(history) for name, history in self.angle_history.items()},
            "velocity_history_length": len(self.velocity_stability_history),
            "stability_counter": self.stability_counter,
            "required_stable_frames": self.config["min_stable_frames"],
            "current_config": self.config
        }
        
        if self.velocity_stability_history:
            latest_velocity = self.velocity_stability_history[-1]
            info["latest_velocity_stability"] = latest_velocity
        
        return info

def draw_waiting_for_ready_display(image, ready_status_info, position=(200, 600)):
    """
    NEW FUNCTION: Draw display when waiting for ready position
    
    Args:
        image: OpenCV image
        ready_status_info: Dictionary from fitness_scorer.check_ready_and_start_tracking()
        position: (x, y) position for display
    """
    x, y = position
    ready_status = ready_status_info.get('ready_status')
    
    if not ready_status:
        return image
    
    # Create semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x-10, y-30), (x+500, y+120), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Status colors and messages based on ready position status
    status = ready_status.get('status', 'unknown')
    
    if status == 'ready' and ready_status.get('tracking_active'):
        status_color = (0, 255, 0)  # Green
        status_text = "🎯 READY - TRACKING WILL START!"
        instruction_text = "Excellent! Begin your exercise now."
    elif status == 'stabilizing':
        status_color = (0, 255, 255)  # Yellow
        status_text = "⏳ GET READY - Hold Position"
        instruction_text = "Hold steady in ready position..."
    elif status == 'wrong_position':
        status_color = (0, 165, 255)  # Orange
        status_text = "📍 MOVE TO READY POSITION"
        instruction_text = "Position arms extended down at sides"
    elif status == 'unstable':
        status_color = (60, 76, 231)  # Red
        status_text = "⚡ TOO MUCH MOVEMENT"
        instruction_text = "Hold steady - minimize movement"
    elif status == 'low_confidence':
        status_color = (128, 128, 128)  # Gray
        status_text = "📹 IMPROVE CAMERA VIEW"
        instruction_text = "Ensure you're fully visible and well-lit"
    else:
        status_color = (128, 128, 128)  # Gray
        status_text = f"Status: {status.upper()}"
        instruction_text = ready_status.get('message', 'Preparing...')
    
    # Draw main status
    cv2.putText(image, status_text, (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Draw instruction
    cv2.putText(image, instruction_text, (x, y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw detailed message from ready detector
    detail_message = ready_status.get('message', '')
    if detail_message and len(detail_message) < 60:  # Only show if not too long
        cv2.putText(image, detail_message, (x, y + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Draw progress bar for stabilizing
    if status == 'stabilizing':
        progress = ready_status.get('stability_progress', 0)
        bar_width = 300
        bar_height = 10
        bar_x = x
        bar_y = y + 80
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
        
        # Progress bar
        progress_width = int((progress / 100) * bar_width)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), status_color, -1)
        
        # Progress text
        cv2.putText(image, f"{progress:.0f}% Ready", (bar_x + bar_width + 15, bar_y + 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    
    # Show preparation frame count
    prep_frames = ready_status_info.get('preparation_frames', 0)
    cv2.putText(image, f"Preparation frames: {prep_frames}", (x, y + 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return image

class IntegratedRPECalculator:
    """
    Integrated RPE Calculator that uses data from other system components
    instead of duplicating their functionality
    """
    
    def __init__(self, exercise_type="dumbell-curl"):
        self.exercise_type = exercise_type
        
        # NEW: Add ready position detector
        self.ready_detector = ReadyPositionDetector(exercise_type)
        self.tracking_started = False  # Flag to track if we've started actual RPE tracking
        
        # RPE calculation weights - focusing on what matters most
        self.rpe_weights = {
            'velocity_decay': 0.35,
            'form_degradation': 0.10,
            'rom_decline': 0.35,
            'rep_progression': 0.20 
        }
        
        self.rep_velocity_averages = []
        self.current_rep_velocities = []
        self.last_completed_rep = 0
        
        # Baseline tracking for comparisons
        self.baseline_form_score = None
        self.baseline_velocity_data = {}
        self.baseline_rom_data = None
        
        # Historical tracking for trends
        self.form_score_history = []
        self.velocity_history = []
        self.rom_history = []
        
        # Configuration
        self.analysis_window = 3
        self.min_data_points = 2
        
        print(f"Integrated RPE Calculator initialized for {exercise_type}")
        print("Waiting for ready position before starting RPE tracking...")
    
    def update_rpe_data(self, rep_counter_data, fitness_scorer_data, velocity_data, 
                       body_measurements=None, angles=None, keypoints=None, 
                       confidences=None, frame_count=None):
        """
        Update RPE calculation with data from integrated systems
        ENHANCED: Now waits for ready position before tracking
        """
        try:
            # NEW: Check ready position first if not already tracking
            if not self.tracking_started:
                if angles is not None and keypoints is not None and confidences is not None and frame_count is not None:
                    ready_status = self.ready_detector.check_ready_position(
                        angles, keypoints, confidences, frame_count
                    )
                    
                    if ready_status['ready'] and ready_status['tracking_active']:
                        self.tracking_started = True
                        print("🚀 RPE TRACKING STARTED - User in ready position!")
                        
                        # Initialize baselines now that tracking has started
                        if fitness_scorer_data and 'current_points' in fitness_scorer_data:
                            self.baseline_form_score = fitness_scorer_data['current_points']
                    else:
                        # Not ready yet, skip RPE data collection
                        return
                else:
                    # Missing required data for ready position check
                    print("RPE: Waiting for complete pose data to check ready position...")
                    return
            
            # EXISTING CODE: Continue with normal RPE data collection only if tracking started
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            current_rep_count = rep_counter_data.get('total_reps', 0)
            
            # Store form score data
            if fitness_scorer_data and 'current_points' in fitness_scorer_data:
                current_form = fitness_scorer_data['current_points']
                self.form_score_history.append({
                    'time': current_time,
                    'score': current_form,
                    'rep_count': current_rep_count
                })
                
                if self.baseline_form_score is None and current_form > 0:
                    self.baseline_form_score = current_form
            
            # Track velocities per rep
            if velocity_data and current_rep_count >= 0:
                self.current_rep_velocities.append({
                    'time': current_time,
                    'data': velocity_data.copy(),
                    'rep_count': current_rep_count
                })
                
                if current_rep_count > self.last_completed_rep and current_rep_count > 0:
                    rep_avg_velocity = self._calculate_rep_average_velocity()
                    
                    if rep_avg_velocity is not None:
                        self.rep_velocity_averages.append({
                            'rep_number': self.last_completed_rep + 1,
                            'average_velocities': rep_avg_velocity,
                            'time': current_time
                        })
                        
                        print(f"Rep {self.last_completed_rep + 1} completed. Average velocities recorded.")
                    
                    self.last_completed_rep = current_rep_count
                    self.current_rep_velocities = []
            
            # Store velocity data
            if velocity_data:
                self.velocity_history.append({
                    'time': current_time,
                    'data': velocity_data.copy(),
                    'rep_count': current_rep_count,
                    'body_scale': body_measurements.get('reference_scale') if body_measurements else None
                })
                
                if not self.baseline_velocity_data and len(self.velocity_history) >= 3:
                    self.baseline_velocity_data = self._calculate_baseline_velocity()
            
            # Store ROM data
            if rep_counter_data and 'total_reps' in rep_counter_data:
                rom_entry = {
                    'time': current_time,
                    'total_reps': rep_counter_data['total_reps'],
                    'current_phase': rep_counter_data['current_phase']
                }
                self.rom_history.append(rom_entry)
            
            # Maintain history sizes
            max_history = 50
            for history_list in [self.form_score_history, self.velocity_history, self.rom_history]:
                if len(history_list) > max_history:
                    history_list.pop(0)
            
            if len(self.rep_velocity_averages) > 20:
                self.rep_velocity_averages.pop(0)
                    
        except Exception as e:
            print(f"Error in update_rpe_data: {e}")
    
    def calculate_rpe(self, rep_counter, fitness_scorer, velocity_tracker):
        """
        Calculate RPE using data from integrated system components
        ENHANCED: Only calculates RPE after ready position detected
        """
        try:
            # NEW: Check if tracking has started
            if not self.tracking_started:
                return {
                    'rpe': 1.0,  # Minimal RPE while waiting
                    'confidence': 'waiting',
                    'breakdown': {},
                    'interpretation': 'Waiting for ready position...',
                    'fatigue_indicators': [],
                    'data_sources': {
                        'ready_status': self.ready_detector._get_status("waiting", "Assume ready position to start tracking"),
                        'reps': 0,
                        'form_score': 0,
                        'rom_consistency': 0,
                        'velocity_data_points': 0
                    }
                }
            
            # EXISTING CODE: Continue with normal RPE calculation
            rep_data = rep_counter.get_rep_summary()
            rom_data = rep_counter.get_range_of_motion_data()
            form_data = fitness_scorer.get_score_display()
            current_velocities = getattr(velocity_tracker, 'velocity_history', [])
            body_measurements = getattr(velocity_tracker.body_calculator, 'reference_scale', None)
            
            # Calculate RPE components using integrated data
            rpe_components = {}
            
            rpe_components['velocity_decay'] = self._analyze_velocity_decay_integrated(current_velocities, body_measurements)
            rpe_components['form_degradation'] = self._analyze_form_degradation_integrated(form_data)
            rpe_components['rom_decline'] = self._analyze_rom_decline_integrated(rom_data)
            rpe_components['rep_progression'] = self._analyze_rep_progression_integrated(rep_data)
            
            # Calculate weighted RPE
            final_rpe = self._calculate_weighted_rpe(rpe_components)
            
            # Apply contextual adjustments
            final_rpe = self._apply_contextual_adjustments_integrated(final_rpe, rpe_components, rep_data)
            final_rpe = max(1.0, min(10.0, final_rpe))
            
            # Calculate confidence
            confidence = self._calculate_confidence_integrated(rpe_components, rep_data, rom_data)
            
            return {
                'rpe': round(final_rpe, 1),
                'confidence': confidence,
                'breakdown': rpe_components,
                'interpretation': self._interpret_rpe(final_rpe),
                'fatigue_indicators': self._get_fatigue_indicators_integrated(rpe_components),
                'data_sources': {
                    'reps': rep_data['total_reps'],
                    'form_score': form_data['points'],
                    'rom_consistency': rom_data.get('rom_consistency', 0),
                    'velocity_data_points': len(current_velocities),
                    'tracking_active': self.tracking_started
                }
            }
            
        except Exception as e:
            print(f"Integrated RPE calculation error: {e}")
            return self._get_error_result(str(e))
        

    def _calculate_rep_average_velocity(self):
        """Calculate average velocity for the current rep based on stored velocity data"""
        if not self.current_rep_velocities:
            return None
        
        # Focus on key joints for the exercise
        key_joints = ['left_wrist', 'right_wrist']
        joint_averages = {}
        
        for joint in key_joints:
            speeds = []
            
            # Extract all speeds for this joint during the rep
            for vel_entry in self.current_rep_velocities:
                vel_data = vel_entry['data']
                if joint in vel_data and 'speed' in vel_data[joint]:
                    speeds.append(vel_data[joint]['speed'])
            
            # Calculate average speed for this joint during the rep
            if speeds:
                joint_averages[joint] = {
                    'average_speed': np.mean(speeds),
                    'max_speed': max(speeds),
                    'min_speed': min(speeds),
                    'speed_variance': np.var(speeds),
                    'data_points': len(speeds)
                }
        
        return joint_averages if joint_averages else None
    
    def _analyze_velocity_decay_integrated(self, velocity_history, body_scale):
        """
        Analyze velocity decay using rep-based average velocities
        MODIFIED: Now compares early reps to ONLY the most recent completed rep
        """
        try:
            # Need at least 2 completed reps to compare
            if len(self.rep_velocity_averages) < 2:
                return None
            
            # Focus on key joints for the exercise
            key_joints = ['left_wrist', 'right_wrist']
            
            # Get early reps and the single most recent rep for comparison
            num_reps = len(self.rep_velocity_averages)
            
            if num_reps < 3:
                # With only 2 reps, compare first vs second
                early_reps = [self.rep_velocity_averages[0]]
                recent_rep = [self.rep_velocity_averages[1]]  # Still use list format for consistency
            else:
                # With 3+ reps, compare early third vs ONLY the most recent rep
                split_point = max(1, num_reps // 3)  # At least 1 rep in early group
                early_reps = self.rep_velocity_averages[:split_point]
                recent_rep = [self.rep_velocity_averages[-1]]  # CHANGED: Only the last rep
            
            velocity_declines = []
            
            for joint in key_joints:
                # Calculate average velocity for early reps
                early_speeds = []
                for rep_data in early_reps:
                    avg_velocities = rep_data['average_velocities']
                    if joint in avg_velocities:
                        early_speeds.append(avg_velocities[joint]['average_speed'])
                
                # Get velocity for the single most recent rep
                recent_speeds = []
                for rep_data in recent_rep:  # This loop will only run once now
                    avg_velocities = rep_data['average_velocities']
                    if joint in avg_velocities:
                        recent_speeds.append(avg_velocities[joint]['average_speed'])
                
                # Compare early average vs most recent rep average for this joint
                if early_speeds and recent_speeds:
                    early_avg = np.mean(early_speeds)
                    recent_avg = recent_speeds[0]  # Since there's only one recent rep now
                    
                    if early_avg > 0:
                        # Calculate percentage decline (positive = getting slower)
                        velocity_decline = ((early_avg - recent_avg) / early_avg) * 100
                        velocity_declines.append({
                            'joint': joint,
                            'decline_percent': velocity_decline,
                            'early_avg': early_avg,
                            'recent_avg': recent_avg,
                            'early_reps': len(early_speeds),
                            'recent_reps': 1  # Always 1 now since we only use the most recent rep
                        })
                        
                        print(f"Velocity analysis - {joint}: {early_avg:.1f} -> {recent_avg:.1f} ({velocity_decline:+.1f}% change)")
            
            if not velocity_declines:
                return None
            
            # Calculate overall velocity decline across all joints
            decline_percentages = [decline['decline_percent'] for decline in velocity_declines]
            avg_velocity_decline = np.mean(decline_percentages)
            
            # Convert velocity decline to RPE scale
            # Positive decline = getting slower = higher fatigue = higher RPE
            # Negative decline = getting faster = lower fatigue = lower RPE
            
            if avg_velocity_decline <= -10:  # Getting significantly faster (unusual)
                rpe_score = 2.0
            elif avg_velocity_decline <= -5:  # Getting slightly faster
                rpe_score = 2.5                
            elif avg_velocity_decline <= 0:  # Maintaining or slightly faster
                rpe_score = 3.0
            elif avg_velocity_decline <= 5:  # Slight slowdown
                rpe_score = 5.0
            elif avg_velocity_decline <= 10:  # Moderate slowdown
                rpe_score = 6.0
            elif avg_velocity_decline <= 15:  # Noticeable slowdown
                rpe_score = 7.0
            elif avg_velocity_decline <= 20:  # Significant slowdown
                rpe_score = 8.0
            elif avg_velocity_decline <= 30:  # Major slowdown
                rpe_score = 9.0
            else:  # Severe slowdown
                rpe_score = 10.0
            
            print(f"Rep-based velocity decay analysis: {avg_velocity_decline:.1f}% decline -> RPE {rpe_score}")
            print(f"Comparison: Early reps average vs Rep #{num_reps} only")
            
            return rpe_score
                    
        except Exception as e:
            print(f"Error in rep-based velocity decay analysis: {e}")
            return None

    
    def get_rep_velocity_debug_info(self):
        """Get debug information about rep-based velocity tracking"""
        return {
            'completed_reps_tracked': len(self.rep_velocity_averages),
            'current_rep_data_points': len(self.current_rep_velocities),
            'last_completed_rep': self.last_completed_rep,
            'rep_averages': self.rep_velocity_averages[-5:],  # Last 5 reps
        }


    def _analyze_form_degradation_integrated(self, form_data):
        """Analyze form degradation using FitnessScorer data"""
        try:
            if not form_data or 'percentage' not in form_data:
                return None
            
            current_percentage = form_data['percentage']
            
            # Convert form percentage to RPE scale (inverse relationship)
            if current_percentage >= 90:
                return 3.0  # Excellent form = low RPE
            elif current_percentage >= 80:
                return 7.0
            elif current_percentage >= 70:
                return 8.5
            elif current_percentage >= 60:
                return 9.0
            elif current_percentage >= 50:
                return 9.5
            else:
                return 10.0  # Poor form = high RPE
                
        except Exception as e:
            print(f"Error in form degradation analysis: {e}")
            return None
    
    def _analyze_rom_decline_integrated(self, rom_data):
        """Analyze ROM decline using RepCounter ROM data"""
        try:
            if not rom_data or not rom_data.get('rep_data'):
                return None
            
            rep_roms = rom_data['rep_data']
            if len(rep_roms) < 2:
                return None
            
            # Compare early vs ONLY the most recent ROM
            total_reps = len(rep_roms)
            early_roms = [rep['range'] for rep in rep_roms[:max(1, total_reps//2)]]
            most_recent_rom = rep_roms[-1]['range']  # CHANGED: Only the last rep's ROM
            
            if not early_roms or most_recent_rom is None:
                return None
            
            early_avg_rom = np.mean(early_roms)
            recent_rom = most_recent_rom  # CHANGED: No averaging needed, just the single recent ROM
            
            if early_avg_rom <= 0:
                return None
            
            rom_decline = ((early_avg_rom - recent_rom) / early_avg_rom) * 100
            
            # Also consider ROM consistency
            consistency = rom_data.get('rom_consistency', 100)
            
            # Combine decline and consistency
            base_rpe = 4.0
            
            # ROM decline component
            if rom_decline <= 0:
                base_rpe -= 1.0   #3.0
            elif rom_decline <= 5:
                base_rpe += 1.0   #5.0
            elif rom_decline <= 10:
                base_rpe += 2.0   #6.0
            elif rom_decline <= 20:
                base_rpe += 3.0   #7.0
            elif rom_decline <= 30:
                base_rpe += 4.0   #8.0
            elif rom_decline <= 40:
                base_rpe += 5.0   #9.0
            else:
                base_rpe += 6.0  #10.0
            
            # Consistency component
            if consistency >= 80:
                base_rpe -= 0.5
            elif consistency < 60:
                base_rpe += 1.0
            
            print(f"ROM analysis: Early avg {early_avg_rom:.1f} vs Most recent rep {recent_rom:.1f} ({rom_decline:+.1f}% change)")
            print(f"Comparison: Early reps average vs Rep #{total_reps} only")
            
            return max(3.0, min(10.0, base_rpe))
            
        except Exception as e:
            print(f"Error in ROM decline analysis: {e}")
            return None
    
    def _analyze_rep_progression_integrated(self, rep_data):
        """Analyze rep progression using RepCounter data"""
        try:
            if not rep_data:
                return None
            
            total_reps = rep_data.get('total_reps', 0)
            current_phase = rep_data.get('current_phase', 'neutral')
            
            # Base RPE on rep count and phase
            base_rpe = 3.0
            
            if total_reps >= 12:
                base_rpe = 10.0
            elif total_reps >= 10:
                base_rpe = 9.0
            elif total_reps >= 8:
                base_rpe = 8.0
            elif total_reps >= 6:
                base_rpe = 7.0
            elif total_reps >= 4:
                base_rpe = 6.0
            
            # Slight adjustment for phase difficulty
            if current_phase == 'eccentric':
                base_rpe += 0.2
            elif current_phase == 'concentric':
                base_rpe += 0.3
            
            return base_rpe
            
        except Exception as e:
            print(f"Error in rep progression analysis: {e}")
            return None
    
    def _calculate_weighted_rpe(self, components):
        """Calculate weighted RPE from components"""
        weighted_rpe = 0
        total_weight = 0
        
        for component, score in components.items():
            if score is not None and isinstance(score, (int, float)):
                weight = self.rpe_weights.get(component, 0.25)
                weighted_rpe += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 5.0  # Default moderate RPE
        
        return weighted_rpe / total_weight
    
    def _apply_contextual_adjustments_integrated(self, base_rpe, components, rep_data):
        """Apply contextual adjustments using integrated data"""
        adjusted_rpe = base_rpe
        
        # High fatigue indicators boost
        high_fatigue_count = sum([
            1 for score in components.values() 
            if score is not None and isinstance(score, (int, float)) and score >= 7.5
        ])
        
        if high_fatigue_count >= 2:
            adjusted_rpe += 0.8
        elif high_fatigue_count >= 1:
            adjusted_rpe += 0.4
        
        # Rep count adjustment
        total_reps = rep_data.get('total_reps', 0)
        if total_reps > 6:
            fatigue_multiplier = 1 + ((total_reps - 6) * 0.05)
            adjusted_rpe *= min(fatigue_multiplier, 1.25)
        
        return adjusted_rpe
    
    def _calculate_confidence_integrated(self, components, rep_data, rom_data):
        """Calculate confidence based on data availability"""
        data_quality_score = 0
        
        # Component availability
        available_components = sum([1 for score in components.values() if score is not None])
        data_quality_score += (available_components / len(components)) * 0.4
        
        # Rep data quality
        total_reps = rep_data.get('total_reps', 0)
        if total_reps >= 6:
            data_quality_score += 0.3
        elif total_reps >= 2:
            data_quality_score += 0.15
        
        # ROM data quality
        if rom_data and rom_data.get('rep_data'):
            rom_reps = len(rom_data['rep_data'])
            if rom_reps >= 6:
                data_quality_score += 0.3
            elif rom_reps >= 2:
                data_quality_score += 0.15
        
        if data_quality_score >= 0.8:
            return 'high'
        elif data_quality_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _get_fatigue_indicators_integrated(self, components):
        """Get fatigue indicators from integrated analysis"""
        indicators = []
        
        for component, score in components.items():
            if score is not None and isinstance(score, (int, float)) and score >= 6.5:
                component_name = component.replace('_', ' ').title()
                indicators.append(f"{component_name}: {score:.1f}")
        
        return indicators
    
    def _calculate_baseline_velocity(self):
        """Calculate baseline velocity from early data"""
        if len(self.velocity_history) < 3:
            return {}
        
        early_data = self.velocity_history[:3]
        baseline = {}
        
        for vel_frame in early_data:
            for joint, vel_data in vel_frame['data'].items():
                if joint not in baseline:
                    baseline[joint] = []
                if 'speed' in vel_data:
                    baseline[joint].append(vel_data['speed'])
        
        # Average the baseline data
        for joint in baseline:
            if baseline[joint]:
                baseline[joint] = np.mean(baseline[joint])
        
        return baseline
    
    def _interpret_rpe(self, rpe):
        """Provide text interpretation of RPE score"""
        if rpe <= 2:
            return "Very easy - could do many more reps"
        elif rpe <= 3:
            return "Easy - minimal effort required"
        elif rpe <= 4:
            return "Moderately easy - comfortable pace"
        elif rpe <= 5:
            return "Moderate - some effort required"
        elif rpe <= 6:
            return "Moderately hard - noticeable effort"
        elif rpe <= 7:
            return "Hard - significant effort, 3-4 reps left"
        elif rpe <= 8:
            return "Very hard - 1-2 reps left in reserve"
        elif rpe <= 9:
            return "Extremely hard - could maybe do 1 more rep"
        else:
            return "Maximum effort - at or near failure"
    
    def _get_error_result(self, error_msg):
        """Return error result structure"""
        return {
            'rpe': 5.0,
            'confidence': 'error',
            'breakdown': {},
            'interpretation': f'Error: {error_msg}',
            'fatigue_indicators': [],
            'data_sources': {}
        }
    

    def get_ready_position_status(self):
        """Get current ready position detector status"""
        return {
            'tracking_started': self.tracking_started,
            'ready_detector_active': self.ready_detector.tracking_active,
            'current_stability_progress': getattr(self.ready_detector, 'stability_counter', 0)
        }
    
    def force_start_tracking(self):
        """Manually force RPE tracking to start (for testing/debugging)"""
        self.tracking_started = True
        self.ready_detector.tracking_active = True
        print("⚡ RPE TRACKING FORCE STARTED")


    def reset_rpe_data(self):
        """Reset RPE calculation data"""
        # NEW: Reset ready position detector
        self.ready_detector.reset()
        self.tracking_started = False
        
        # EXISTING: Reset RPE data
        self.baseline_form_score = None
        self.baseline_velocity_data = {}
        self.baseline_rom_data = None
        self.form_score_history = []
        self.velocity_history = []
        self.rom_history = []
        
        # Reset rep-based velocity tracking
        self.rep_velocity_averages = []
        self.current_rep_velocities = []
        self.last_completed_rep = 0
        
        print("Integrated RPE data reset - waiting for ready position")


def draw_integrated_rpe_display(image, velocity_tracker, rep_counter, position=(50, 600)):
    """
    Draw integrated RPE display using data from all system components
    """
    try:
        # Get integrated RPE analysis
        rpe_result = velocity_tracker.get_integrated_rpe_analysis(rep_counter)
        
        x, y = position
        
        # Create background
        overlay = image.copy()
        cv2.rectangle(overlay, (x-5, y-25), (x+580, y+280), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(image, "INTEGRATED RPE ANALYSIS", (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # RPE Score with color coding
        rpe_score = rpe_result.get('rpe', 5.0)
        if rpe_score <= 4:
            rpe_color = (0, 255, 0)  # Green
        elif rpe_score <= 6:
            rpe_color = (0, 255, 255)  # Yellow
        elif rpe_score <= 8:
            rpe_color = (0, 165, 255)  # Orange
        else:
            rpe_color = (0, 0, 255)  # Red
        
        cv2.putText(image, f"RPE: {rpe_score}/10", (x, y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, rpe_color, 2)
        
        # Confidence and data sources
        confidence = rpe_result.get('confidence', 'low')
        confidence_colors = {'high': (0, 255, 0), 'medium': (0, 255, 255), 
                           'low': (0, 0, 255), 'error': (128, 128, 128)}
        confidence_color = confidence_colors.get(confidence, (128, 128, 128))
        
        cv2.putText(image, f"Confidence: {confidence.title()}", (x, y + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
        
        # Data sources
        data_sources = rpe_result.get('data_sources', {})
        cv2.putText(image, f"Reps: {data_sources.get('reps', 0)} | Form: {data_sources.get('form_score', 0):.0f}%", 
                    (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.putText(image, f"ROM Consistency: {data_sources.get('rom_consistency', 0):.1f}%", 
                    (x, y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # RPE Breakdown
        breakdown = rpe_result.get('breakdown', {})
        y_breakdown = y + 150
        cv2.putText(image, "Component Analysis:", (x, y_breakdown), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)
        
        for i, (component, score) in enumerate(breakdown.items()):
            if score is not None:
                component_name = component.replace('_', ' ').title()
                score_color = (0, 255, 0) if score <= 5 else (0, 165, 255) if score <= 7 else (0, 0, 255)
                cv2.putText(image, f"{component_name}: {score:.1f}", 
                           (x + 10, y_breakdown + 20 + (i * 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, score_color, 1)
        
        # Interpretation
        interpretation = rpe_result.get('interpretation', 'Moderate effort')
        cv2.putText(image, interpretation, (x, y + 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Fatigue indicators
        fatigue_indicators = rpe_result.get('fatigue_indicators', [])
        if fatigue_indicators:
            cv2.putText(image, f"Fatigue: {', '.join(fatigue_indicators[:2])}", (x, y + 265), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
        
        return image
        
    except Exception as e:
        print(f"Error in integrated RPE display: {e}")
        cv2.putText(image, f"RPE Display Error: {str(e)}", position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return image


class RepCounter:
    """
    Counts repetitions by tracking eccentric and concentric phases of movement
    Supports different exercise types with configurable angle thresholds
    """
    
    def __init__(self, exercise_type="dumbell-curl", ready_position_detector=None):
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.half_rep_count = 0  # Total half reps (eccentric + concentric)
        
        # NEW: Ready position integration
        self.ready_position_detector = ready_position_detector
        self.tracking_enabled = False  # NEW: Only count when this is True
        
        # Phase tracking
        self.current_phase = "neutral"  # "neutral", "eccentric", "concentric"
        self.previous_phase = "neutral"
        
        # Angle history for smoothing and change detection
        self.angle_history = []
        self.history_window = 5
        
        # NEW: Angle change tracking
        self.previous_smoothed_angle = None
        self.angle_direction = None  # "increasing", "decreasing", None
        self.previous_direction = None
        
        # NEW: Range of motion tracking
        self.current_max_angle = None
        self.current_min_angle = None
        self.rom_data = []  # List of (min_angle, max_angle) tuples for each rep
        
        # NEW: Direction confirmation to avoid noise
        self.direction_confirmation_count = 0
        self.min_direction_confirmation = 4
        self.min_angle_change_threshold = 1  # Minimum angle change to consider significant
        
        # Exercise-specific configurations
        self.exercise_configs = {
            "dumbell-curl": {
                "primary_angle": "left_elbow",  # Main angle to track
                "secondary_angle": "right_elbow",  # Secondary angle for validation
                "start_threshold": 170,  # Starting position (extended) - kept for reference
                "end_threshold": 60,     # End position (flexed) - kept for reference
                "hysteresis": 10,        # Prevents oscillation
                "require_both_arms": True,  # Both arms must move together
                # NEW: Define which direction is eccentric vs concentric
                "eccentric_direction": "increasing",  # For curls, eccentric is angle decreasing (flexing)
                "concentric_direction": "decreasing"   # For curls, concentric is angle increasing (extending)
            }
        }
        
        # Get current exercise config
        self.config = self.exercise_configs.get(exercise_type, self.exercise_configs["dumbell-curl"])
        
        # Rep validation
        self.last_rep_frame = 0
        self.min_frames_between_reps = 15  # Minimum frames between complete reps
        
        print(f"RepCounter initialized for {exercise_type}")
        if ready_position_detector:
            print("Ready position detector integration ENABLED - waiting for ready signal")
        else:
            print("WARNING: No ready position detector - counting will start immediately")
        print(f"Tracking: {self.config['primary_angle']} (direction-based counting)")
    
    def update_rep_count(self, angles, frame_count, keypoints=None, confidences=None):
        """
        Update rep count based on current angles - ENHANCED with ready position integration
        
        Args:
            angles: Dictionary of current body angles
            frame_count: Current frame number
            keypoints: Current keypoint positions (for ready position detection)
            confidences: Keypoint confidence scores (for ready position detection)
            
        Returns:
            dict: Rep counting information and events
        """
        # NEW: Check ready position status first
        ready_status = self._check_ready_position_status(angles, keypoints, confidences, frame_count)
        
        # If we're not ready to track, return early
        if not self.tracking_enabled:
            return self._get_status_dict_with_ready_info(ready_status)
        
        primary_angle = angles.get(self.config['primary_angle'])
        secondary_angle = angles.get(self.config['secondary_angle'])
        
        if primary_angle is None:
            return self._get_status_dict_with_ready_info(ready_status)
        
        # Add angle to history for smoothing
        self.angle_history.append(primary_angle)
        if len(self.angle_history) > self.history_window:
            self.angle_history.pop(0)
        
        # Use smoothed angle
        smoothed_angle = sum(self.angle_history) / len(self.angle_history)
        
        # NEW: Detect angle direction and phase changes
        direction_changed, rep_events = self._detect_direction_change(smoothed_angle, frame_count)
        
        result = {
            'total_reps': self.rep_count,
            'half_reps': self.half_rep_count,
            'current_phase': self.current_phase,
            'primary_angle': primary_angle,
            'smoothed_angle': smoothed_angle,
            'angle_direction': self.angle_direction,
            'current_max_angle': self.current_max_angle,
            'current_min_angle': self.current_min_angle,
            'events': rep_events,
            'direction_changed': direction_changed,
            # NEW: Add ready position information
            'ready_status': ready_status,
            'tracking_enabled': self.tracking_enabled
        }
        
        return result
    
    # NEW: Main function to detect direction changes
    def _detect_direction_change(self, smoothed_angle, frame_count):
        """Detect changes in angle direction and count half reps"""
        events = []
        direction_changed = False
        
        if self.previous_smoothed_angle is None:
            self.previous_smoothed_angle = smoothed_angle
            return False, events
        
        # Calculate angle change
        angle_change = smoothed_angle - self.previous_smoothed_angle
        
        # Only consider significant changes to avoid noise
        if abs(angle_change) < self.min_angle_change_threshold:
            self.previous_smoothed_angle = smoothed_angle
            return False, events
        
        # Determine current direction
        current_direction = "increasing" if angle_change > 0 else "decreasing"
        
        # Confirm direction change
        if current_direction != self.angle_direction:
            self.direction_confirmation_count += 1
            
            if self.direction_confirmation_count >= self.min_direction_confirmation:
                # Direction change confirmed
                self.previous_direction = self.angle_direction
                self.angle_direction = current_direction
                self.direction_confirmation_count = 0
                direction_changed = True
                
                # Record extremum and process half rep
                events = self._process_direction_change(smoothed_angle, frame_count)
                
        else:
            self.direction_confirmation_count = 0
        
        # Update angle tracking for range of motion
        self._update_angle_extremes(smoothed_angle)
        
        self.previous_smoothed_angle = smoothed_angle
        return direction_changed, events
    
    # NEW: Process direction changes and count half reps
    def _process_direction_change(self, current_angle, frame_count):
        """Process confirmed direction change and update counters"""
        events = []
        
        if self.previous_direction is None:
            # First direction detected, just set phase
            self._set_phase_from_direction(self.angle_direction)
            return events
        
        # Record the extremum angle when direction changes
        self._record_extremum_angle(current_angle)
        
        # Count half rep
        self.half_rep_count += 1
        
        # Update phase based on new direction
        self._set_phase_from_direction(self.angle_direction)
        
        # Create event for half rep
        phase_name = "eccentric" if self.angle_direction == self.config["eccentric_direction"] else "concentric"
        events.append({
            'type': 'half_rep',
            'message': f'{phase_name.capitalize()} phase started (Half rep #{self.half_rep_count})',
            'half_reps': self.half_rep_count,
            'direction': self.angle_direction,
            'phase': self.current_phase,
            'angle': current_angle,
            'frame': frame_count
        })
        
        # Check for full rep (every 2 half reps)
        if self.half_rep_count % 2 == 0 and self.half_rep_count > 0:
            if frame_count - self.last_rep_frame >= self.min_frames_between_reps:
                self.rep_count += 1
                self.last_rep_frame = frame_count
                
                # Record ROM data for completed rep
                if self.current_min_angle is not None and self.current_max_angle is not None:
                    self.rom_data.append((self.current_min_angle, self.current_max_angle))
                    # Reset for next rep
                    self.current_min_angle = None
                    self.current_max_angle = None
                
                events.append({
                    'type': 'rep_complete',
                    'message': f'Rep #{self.rep_count} completed!',
                    'total_reps': self.rep_count,
                    'half_reps': self.half_rep_count,
                    'frame': frame_count
                })
        
        return events
    
    
    # NEW: Set movement phase based on direction
    def _set_phase_from_direction(self, direction):
        """Set current phase based on angle direction"""
        self.previous_phase = self.current_phase
        
        if direction == self.config["eccentric_direction"]:
            self.current_phase = "eccentric"
        elif direction == self.config["concentric_direction"]:
            self.current_phase = "concentric"
        else:
            self.current_phase = "neutral"
    
    # NEW: Update min/max angle tracking
    def _update_angle_extremes(self, angle):
        """Update current min and max angles for ROM tracking"""
        if self.current_min_angle is None or angle < self.current_min_angle:
            self.current_min_angle = angle
        
        if self.current_max_angle is None or angle > self.current_max_angle:
            self.current_max_angle = angle
    
    # NEW: Record extremum when direction changes
    def _record_extremum_angle(self, angle):
        """Record extremum angle when movement direction changes"""
        # This angle represents a peak or valley in the movement
        pass  # The extremum is already captured in current_min_angle/current_max_angle
    
    # NEW: Get range of motion data
    def get_range_of_motion_data(self):
        """Get range of motion data for completed reps"""
        if not self.rom_data:
            return {
                'average_rom': 0,
                'min_rom': 0,
                'max_rom': 0,
                'rom_consistency': 0,
                'rep_data': []
            }
        
        rom_ranges = [max_angle - min_angle for min_angle, max_angle in self.rom_data]
        
        return {
            'average_rom': sum(rom_ranges) / len(rom_ranges),
            'min_rom': min(rom_ranges),
            'max_rom': max(rom_ranges),
            'rom_consistency': (min(rom_ranges) / max(rom_ranges)) * 100 if max(rom_ranges) > 0 else 0,
            'rep_data': [
                {
                    'rep': i + 1,
                    'min_angle': min_angle,
                    'max_angle': max_angle,
                    'range': max_angle - min_angle
                }
                for i, (min_angle, max_angle) in enumerate(self.rom_data)
            ]
        }
    
    def _get_status_dict(self):
        """Return current status when angle data is unavailable"""
        return {
            'total_reps': self.rep_count,
            'half_reps': self.half_rep_count,
            'current_phase': self.current_phase,
            'primary_angle': None,
            'smoothed_angle': None,
            'angle_direction': self.angle_direction,
            'current_max_angle': self.current_max_angle,
            'current_min_angle': self.current_min_angle,
            'events': [],
            'direction_changed': False
        }
    
    def _check_ready_position_status(self, angles, keypoints, confidences, frame_count):
        """
        Check ready position status and update tracking state
        
        Returns:
            dict: Ready position status information
        """
        if self.ready_position_detector is None:
            # No ready position detector - start tracking immediately
            if not self.tracking_enabled:
                self.tracking_enabled = True
                print("🎯 NO READY DETECTOR - TRACKING STARTED IMMEDIATELY")
            
            return {
                "ready": True,
                "tracking_active": True,
                "status": "no_detector",
                "message": "Tracking active (no ready detector)",
                "detection_method": "none"
            }
        
        # Get ready position status
        ready_status = self.ready_position_detector.check_ready_position(
            angles, keypoints, confidences, frame_count
        )
        
        # Update tracking state based on ready status
        if ready_status["ready"] and ready_status["tracking_active"]:
            if not self.tracking_enabled:
                self.tracking_enabled = True
                print("🎯 READY POSITION CONFIRMED - REP COUNTING STARTED!")
        elif not ready_status["ready"]:
            # If user leaves ready position, we might want to keep tracking
            # or stop tracking depending on your preference
            # For now, let's keep tracking once started
            pass
        
        return ready_status
    

    def _get_status_dict_with_ready_info(self, ready_status):
        """Return current status when tracking is disabled or angle data unavailable"""
        return {
            'total_reps': self.rep_count,
            'half_reps': self.half_rep_count,
            'current_phase': self.current_phase,
            'primary_angle': None,
            'smoothed_angle': None,
            'angle_direction': self.angle_direction,
            'current_max_angle': self.current_max_angle,
            'current_min_angle': self.current_min_angle,
            'events': [],
            'direction_changed': False,
            # NEW: Add ready position information
            'ready_status': ready_status,
            'tracking_enabled': self.tracking_enabled
        }


    def set_ready_position_detector(self, ready_position_detector):
        """
        Set or update the ready position detector
        
        Args:
            ready_position_detector: ReadyPositionDetector instance
        """
        self.ready_position_detector = ready_position_detector
        print("Ready position detector updated for RepCounter")


    def force_start_tracking(self):
        """
        Force start tracking regardless of ready position status
        Useful for manual override or testing
        """
        self.tracking_enabled = True
        print("🎯 REP TRACKING FORCE STARTED!")

    def stop_tracking(self):
        """
        Stop rep counting tracking
        """
        self.tracking_enabled = False
        print("⏸️ REP TRACKING STOPPED")

    def is_tracking_enabled(self):
        """
        Check if rep counting is currently enabled
        
        Returns:
            bool: True if tracking is enabled
        """
        return self.tracking_enabled


    def reset_count(self):
        """Reset all counters and tracking state"""
        self.rep_count = 0
        self.half_rep_count = 0
        self.current_phase = "neutral"
        self.previous_phase = "neutral"
        self.angle_history = []
        
        # NEW: Reset tracking state
        self.tracking_enabled = False  # Must wait for ready position again
        
        # NEW: Reset direction tracking
        self.previous_smoothed_angle = None
        self.angle_direction = None
        self.previous_direction = None
        self.direction_confirmation_count = 0
        
        # NEW: Reset ROM tracking
        self.current_max_angle = None
        self.current_min_angle = None
        self.rom_data = []
        
        self.last_rep_frame = 0
        
        # NEW: Reset ready position detector if available
        if self.ready_position_detector:
            self.ready_position_detector.reset()
        
        print("Rep counter reset - waiting for ready position")

    # FIXED: Added the missing get_rep_summary method
    def get_rep_summary(self):
        """Get formatted summary of rep counting"""
        return {
            'exercise': self.exercise_type,
            'total_reps': self.rep_count,
            'half_reps': self.half_rep_count,
            'current_phase': self.current_phase,
            'tracking_angle': self.config['primary_angle'],
            'angle_direction': self.angle_direction,
            'current_range': f"{self.current_min_angle}° - {self.current_max_angle}°" if self.current_min_angle is not None and self.current_max_angle is not None else "Not available",
            'tracking_enabled': self.tracking_enabled,
            'ready_detector': "Active" if self.ready_position_detector else "None"
        }




class FitnessScorer:
    """
    Fitness scoring system that starts at 100 points and deducts for incorrect form
    Uses angles, velocities, and joint steadiness to assess exercise performance
    ENHANCED: Only starts tracking when Ready Position Detector confirms user is ready
    """
    
    def __init__(self, exercise_type="dumbell-curl", starting_points=100, center_mode="reset", ready_position_detector=None):
        self.exercise_type = exercise_type
        self.starting_points = starting_points
        self.current_points = starting_points
        self.deduction_history = []
        self.frame_count = 0
        self.center_mode = center_mode
        
        # NEW: Ready position integration
        self.ready_position_detector = ready_position_detector
        self.tracking_started = False  # NEW: Flag to control when tracking begins
        self.pre_ready_frames = 0  # NEW: Count frames before ready position
        
        # MODIFIED: Pass ready_position_detector to RepCounter
        self.rep_counter = RepCounter(exercise_type, ready_position_detector)
        
        # Exercise-specific angle ranges (in degrees)
        self.angle_rules = {
            "dumbell-curl": {
                "left_elbow": {"min": 20, "max": 180, "deduction": 5},
                "right_elbow": {"min": 20, "max": 180, "deduction": 5},
            }
        }
        
        self.rom_rules = {
            "dumbell-curl": {
                "left_elbow": {
                    "optimal_min_range": (30, 65),     
                    "optimal_max_range": (170, 180),   
                    "contraction_deduction": 2.0,      
                    "extension_deduction": 2.0,        
                    "frequency_penalty": 30,           
                    "severe_contraction_threshold": 90,  
                    "severe_extension_threshold": 140,   
                },
                "right_elbow": {
                    "optimal_min_range": (30, 65),
                    "optimal_max_range": (170, 180),
                    "contraction_deduction": 2.0,
                    "extension_deduction": 2.0,  
                    "frequency_penalty": 30,
                    "severe_contraction_threshold": 90,
                    "severe_extension_threshold": 140,
                }
            }
        }
        
        # NEW: ROM tracking variables  
        self.rom_violations = {}
        self.last_rom_check_frame = {}

        # Velocity rules (speed categories from your adaptive system)
        self.velocity_rules = {
            "excessive_speed": {
                 "joints": {
                    "left_wrist": {"allowed_speeds": ["medium", "slow"], "deduction": 0.1},
                    "right_wrist": {"allowed_speeds": ["medium", "slow"], "deduction": 0.1}
                         },
                "max_fast_frames": 15,  
                "max_medium_frames": 30,  
            }
        }
        
        # NEW: Steadiness detection rules
        self.steadiness_rules = {
            "joints": {},  
            "global_settings": {
                "update_center_frames": 30,  
                "min_frames_before_penalty": 30,  
                "consecutive_violation_threshold": 20  
            }
        }
        
        # Tracking variables
        self.fast_movement_counter = {}
        
        # NEW: Steadiness tracking variables
        self.steadiness_centers = {}  
        self.steadiness_violations = {}  
        self.steadiness_history = {}  

        # Initialize joint steadiness requirements (but tracking won't start until ready)
        self.set_joint_steadiness_requirements("left_shoulder", 0.125, 1, True, center_mode="fixed")
        self.set_joint_steadiness_requirements("right_shoulder", 0.125, 1, True, center_mode="fixed")
        self.set_joint_steadiness_requirements("left_elbow", 0.25, 1, True, center_mode="fixed")
        self.set_joint_steadiness_requirements("right_elbow", 0.25, 1, True, center_mode="fixed")
        
        print(f"FitnessScorer initialized - waiting for ready position before tracking starts")
    
    def check_ready_and_start_tracking(self, angles, keypoints, confidences, frame_count, velocities=None):
        """
        NEW METHOD: Check ready position and start tracking when ready
        This method MUST be called before evaluate_frame()
        
        Args:
            angles: Current body angles
            keypoints: Current keypoints
            confidences: Current confidence scores
            frame_count: Current frame number
            velocities: Optional velocity data
            
        Returns:
            dict: Ready position status and tracking start information
        """
        self.pre_ready_frames += 1
        
        if not self.tracking_started and self.ready_position_detector is not None:
            # Check if user is in ready position
            ready_status = self.ready_position_detector.check_ready_position(
                angles, keypoints, confidences, frame_count, velocities
            )
            
            # Start tracking when ready position is detected
            if ready_status['ready'] and ready_status['tracking_active']:
                self.tracking_started = True
                self.frame_count = 0  # Reset frame count to start fresh
                
                # CRITICAL: Initialize steadiness centers NOW with current positions
                self._initialize_steadiness_centers_on_ready(keypoints, confidences)
                
                print(f"🎯 FITNESS TRACKING STARTED after {self.pre_ready_frames} preparation frames!")
                print(f"   Steadiness centers initialized for {len(self.steadiness_centers)} joints")
                
                return {
                    'tracking_started': True,
                    'ready_status': ready_status,
                    'preparation_frames': self.pre_ready_frames,
                    'message': 'Fitness tracking now active!'
                }
            else:
                # Still waiting for ready position
                return {
                    'tracking_started': False,
                    'ready_status': ready_status,
                    'preparation_frames': self.pre_ready_frames,
                    'message': f'Waiting for ready position... ({ready_status.get("message", "Getting ready")})'
                }
        
        # If already tracking or no ready position detector
        return {
            'tracking_started': self.tracking_started,
            'ready_status': None,
            'preparation_frames': self.pre_ready_frames,
            'message': 'Tracking active' if self.tracking_started else 'No ready position detector'
        }
    
    def _initialize_steadiness_centers_on_ready(self, keypoints, confidences, confidence_threshold=0.3):
        """
        NEW METHOD: Initialize steadiness centers when ready position is detected
        This ensures centers are set from a proper starting position, not random movement
        
        CRITICAL REASONING:
        - We want steadiness centers to represent the "ideal" position when user is ready
        - Setting centers during movement/setup would create incorrect baselines
        - By setting centers at ready position, we ensure accurate steadiness tracking
        """
        # Keypoint name mapping
        keypoint_mapping = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
            'neck': 17, 'chest': 18, 'mid_spine': 19, 'lower_spine': 20, 'tail_bone': 21,
            'left_toes': 22, 'right_toes': 23
        }
        
        centers_initialized = 0
        
        for joint_name, rule in self.steadiness_rules["joints"].items():
            if joint_name not in keypoint_mapping:
                continue
                
            joint_idx = keypoint_mapping[joint_name]
            if joint_idx >= len(keypoints) or joint_idx >= len(confidences):
                continue
                
            # Only initialize if confidence is high enough
            if confidences[joint_idx] > confidence_threshold:
                # Set center to current position (ready position)
                self.steadiness_centers[joint_name] = keypoints[joint_idx].copy()
                
                # Initialize tracking variables
                self.steadiness_violations[joint_name] = 0
                self.steadiness_history[joint_name] = [keypoints[joint_idx].copy()]
                
                centers_initialized += 1
                print(f"   Initialized {joint_name} center at position {keypoints[joint_idx]}")
        
        print(f"   Total steadiness centers initialized: {centers_initialized}")
    
    def evaluate_frame(self, angles, velocities, keypoints=None, body_scale=None, confidences=None):
        """
        Evaluate a single frame and deduct points for incorrect form
        ENHANCED: Only performs evaluation when tracking has started
        CRITICAL: Must call check_ready_and_start_tracking() first!
        """
        # CRITICAL CHECK: Only evaluate if tracking has started
        if not self.tracking_started:
            return {
                'current_points': self.current_points,
                'frame_deductions': [],
                'total_deduction': 0.0,
                'feedback': ['Waiting for ready position...'],
                'score_percentage': 100.0,  # Keep at 100% until tracking starts
                'rep_info': self.rep_counter.get_rep_summary(),  # Return empty rep info
                'tracking_active': False  # NEW: Indicate tracking is not active
            }
        
        # Increment frame count ONLY after tracking starts
        self.frame_count += 1
        frame_deductions = []
        feedback_messages = []
        
        # 1. ANGLE-BASED EVALUATION (only when tracking active)
        angle_deductions = self._evaluate_angles(angles)
        frame_deductions.extend(angle_deductions)
        
        # 2. VELOCITY-BASED EVALUATION (only when tracking active)
        velocity_deductions = self._evaluate_velocities(velocities)
        frame_deductions.extend(velocity_deductions)
        
        # 3. STEADINESS-BASED EVALUATION (only when tracking active)
        if keypoints is not None:
            steadiness_deductions = self._evaluate_steadiness(keypoints, body_scale)
            frame_deductions.extend(steadiness_deductions)
        
        # 4. REP COUNTING (enhanced to work with ready position)
        rep_info = self.rep_counter.update_rep_count(angles, self.frame_count, keypoints, confidences)
        
        # 5. RANGE OF MOTION EVALUATION
        rom_deductions = self._evaluate_range_of_motion(rep_info)
        frame_deductions.extend(rom_deductions)
        
        # 6. APPLY DEDUCTIONS
        total_frame_deduction = sum([d['amount'] for d in frame_deductions])
        self.current_points = max(0, self.current_points - total_frame_deduction)
        
        # 7. STORE HISTORY
        if frame_deductions:
            self.deduction_history.append({
                'frame': self.frame_count,
                'deductions': frame_deductions,
                'total_deduction': total_frame_deduction,
                'points_remaining': self.current_points
            })
        
        # 8. GENERATE FEEDBACK
        feedback_messages = self._generate_feedback(frame_deductions)
        
        # Add rep event messages to feedback
        for event in rep_info['events']:
            if event['type'] == 'rep_complete':
                feedback_messages.insert(0, f"🎉 {event['message']}")
            elif event['type'] in ['eccentric_complete', 'concentric_complete']:
                feedback_messages.append(f"📊 {event['message']}")
        
        return {
            'current_points': self.current_points,
            'frame_deductions': frame_deductions,
            'total_deduction': total_frame_deduction,
            'feedback': feedback_messages,
            'score_percentage': (self.current_points / self.starting_points) * 100,
            'rep_info': rep_info,
            'tracking_active': True  # NEW: Indicate tracking is active
        }
    
    def _evaluate_steadiness(self, keypoints, body_scale=None):
        """
        Evaluate joint steadiness against defined radius requirements
        ENHANCED: Only evaluates steadiness if centers have been properly initialized
        """
        deductions = []
        
        # CRITICAL: Only evaluate steadiness if tracking has started
        if not self.tracking_started:
            return deductions
        
        # Keypoint name mapping
        keypoint_mapping = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
            'neck': 17, 'chest': 18, 'mid_spine': 19, 'lower_spine': 20, 'tail_bone': 21,
            'left_toes': 22, 'right_toes': 23
        }
        
        for joint_name, rule in self.steadiness_rules["joints"].items():
            if joint_name not in keypoint_mapping:
                continue
                
            joint_idx = keypoint_mapping[joint_name]
            if joint_idx >= len(keypoints):
                continue
                
            current_position = keypoints[joint_idx]
            
            # CRITICAL: Only evaluate if center has been initialized
            if joint_name not in self.steadiness_centers or self.steadiness_centers[joint_name] is None:
                continue
            
            # Initialize tracking for this joint if needed
            if joint_name not in self.steadiness_history:
                self.steadiness_history[joint_name] = []
                self.steadiness_violations[joint_name] = 0
            
            # Add current position to history
            self.steadiness_history[joint_name].append(current_position.copy())
            
            # Maintain history window
            max_history = self.steadiness_rules["global_settings"]["update_center_frames"]
            if len(self.steadiness_history[joint_name]) > max_history:
                self.steadiness_history[joint_name].pop(0)
            
            # MODIFIED: Handle center updates based on joint's center_mode
            joint_center_mode = rule.get("center_mode", "reset")
            
            if joint_center_mode == "fixed":
                # FIXED MODE: Never update center after initial ready position
                # Center was set in _initialize_steadiness_centers_on_ready()
                pass  # Do nothing - keep the original center
            
            else:  # joint_center_mode == "reset"
                # RESET MODE: Update center periodically ONLY AFTER tracking started
                if self.frame_count % self.steadiness_rules["global_settings"]["update_center_frames"] == 0:
                    if len(self.steadiness_history[joint_name]) >= 5:
                        positions = np.array(self.steadiness_history[joint_name])
                        self.steadiness_centers[joint_name] = np.mean(positions, axis=0)
            
            # Check steadiness (same as before, but now we know center is valid)
            center = self.steadiness_centers[joint_name]
            distance = np.linalg.norm(current_position - center)
            
            # Get effective radius
            if rule["adaptive_radius"] and body_scale is not None:
                effective_radius = rule["radius"] * body_scale
            else:
                effective_radius = rule["radius"]
            
            # Check if outside allowed radius
            if distance > effective_radius:
                self.steadiness_violations[joint_name] += 1
                
                consecutive_threshold = self.steadiness_rules["global_settings"]["consecutive_violation_threshold"]
                min_frames = self.steadiness_rules["global_settings"]["min_frames_before_penalty"]
                
                if (self.steadiness_violations[joint_name] >= consecutive_threshold and 
                    self.frame_count >= min_frames):
                    
                    deductions.append({
                        'type': 'steadiness',
                        'joint': joint_name,
                        'amount': rule["deduction"],
                        'reason': f"{joint_name.replace('_', ' ').title()}: Moving outside steady position (distance: {distance:.1f}, limit: {effective_radius:.1f})",
                        'severity': 'medium' if distance < effective_radius * 1.5 else 'high',
                        'distance': distance,
                        'limit': effective_radius
                    })
                    
                    self.steadiness_violations[joint_name] = 0
            else:
                self.steadiness_violations[joint_name] = 0
        
        return deductions
    
    # [Include all other existing methods unchanged...]
    def set_joint_speed_requirements(self, joint_name, allowed_speeds, deduction_amount=0.2):
        if "excessive_speed" not in self.velocity_rules:
            self.velocity_rules["excessive_speed"] = {"joints": {}, "max_bad_frames": 5}
        
        self.velocity_rules["excessive_speed"]["joints"][joint_name] = {
            "allowed_speeds": allowed_speeds,
            "deduction": deduction_amount
        }
        
        print(f"Set {joint_name} to allow speeds: {allowed_speeds} (penalty: {deduction_amount} points)")
    
    def set_joint_steadiness_requirements(self, joint_name, radius, deduction_amount=0.3, adaptive_radius=True, center_mode=None):
        effective_center_mode = center_mode if center_mode is not None else self.center_mode
        
        self.steadiness_rules["joints"][joint_name] = {
            "radius": radius,
            "deduction": deduction_amount,
            "adaptive_radius": adaptive_radius,
            "center_mode": effective_center_mode
        }
        
        # Initialize tracking for this joint (but don't set center until ready)
        self.steadiness_centers[joint_name] = None  # Will be set when ready
        self.steadiness_violations[joint_name] = 0
        self.steadiness_history[joint_name] = []
        
        radius_type = "body-scale units" if adaptive_radius else "pixels"
        center_type = "fixed from ready position" if effective_center_mode == "fixed" else "resets every 60 frames"
        print(f"Set {joint_name} steadiness: radius {radius} {radius_type}, center {center_type} (penalty: {deduction_amount} points)")
    
    def get_joint_steadiness_requirements(self):
        return self.steadiness_rules["joints"]
    
    def remove_joint_steadiness_requirements(self, joint_name):
        if joint_name in self.steadiness_rules["joints"]:
            del self.steadiness_rules["joints"][joint_name]
            
            if joint_name in self.steadiness_centers:
                del self.steadiness_centers[joint_name]
            if joint_name in self.steadiness_violations:
                del self.steadiness_violations[joint_name]
            if joint_name in self.steadiness_history:
                del self.steadiness_history[joint_name]
                
            print(f"Removed steadiness monitoring for {joint_name}")
    
    def set_joint_rom_requirements(self, joint_name, optimal_min, optimal_max, 
                                 contraction_deduction=2.0, extension_deduction=2.0, frequency_penalty=30):
        if self.exercise_type not in self.rom_rules:
            self.rom_rules[self.exercise_type] = {}
            
        self.rom_rules[self.exercise_type][joint_name] = {
            "optimal_min": optimal_min,
            "optimal_max": optimal_max,
            "contraction_deduction": contraction_deduction,
            "extension_deduction": extension_deduction,
            "frequency_penalty": frequency_penalty
        }
        
        print(f"Set ROM requirements for {joint_name}: {optimal_min}°-{optimal_max}° "
              f"(penalties: {contraction_deduction}/{extension_deduction} points)")

    def get_rom_violations_summary(self):
        return {
            'violations': self.rom_violations,
            'total_contraction_violations': sum([v['insufficient_contraction'] for v in self.rom_violations.values()]),
            'total_extension_violations': sum([v['insufficient_extension'] for v in self.rom_violations.values()]),
        }

    def _evaluate_range_of_motion(self, rep_info):
        """ROM evaluation - only when tracking is active"""
        if not self.tracking_started:
            return []
            
        deductions = []
        rep_complete_events = [event for event in rep_info['events'] if event['type'] == 'rep_complete']
        
        if not rep_complete_events:
            return deductions
        
        rom_data = self.rep_counter.get_range_of_motion_data()
        if not rom_data['rep_data']:
            return deductions
        
        latest_rep = rom_data['rep_data'][-1]
        actual_min = latest_rep['min_angle']
        actual_max = latest_rep['max_angle']
        
        current_rom_rules = self.rom_rules.get(self.exercise_type, self.rom_rules["dumbell-curl"])
        primary_joint = self.rep_counter.config['primary_angle']
        
        if primary_joint in current_rom_rules:
            rule = current_rom_rules[primary_joint]
            
            if primary_joint not in self.rom_violations:
                self.rom_violations[primary_joint] = {
                    'insufficient_contraction': 0,
                    'insufficient_extension': 0,
                    'last_contraction_check': 0,
                    'last_extension_check': 0
                }
            
            best_min, acceptable_min = rule['optimal_min_range']
            
            if actual_min > acceptable_min:
                frames_since_check = self.frame_count - self.rom_violations[primary_joint]['last_contraction_check']
                if frames_since_check >= rule.get('frequency_penalty', 30):
                    
                    self.rom_violations[primary_joint]['insufficient_contraction'] += 1
                    self.rom_violations[primary_joint]['last_contraction_check'] = self.frame_count
                    
                    if actual_min > rule.get('severe_contraction_threshold', 70):
                        severity = 'high'
                        penalty_multiplier = 2.0
                        performance_msg = "severely limited"
                    else:
                        severity = 'medium'
                        penalty_multiplier = 1.0
                        performance_msg = "limited"
                    
                    deductions.append({
                        'type': 'rom_contraction',
                        'joint': primary_joint,
                        'amount': rule['contraction_deduction'] * penalty_multiplier,
                        'reason': f"{primary_joint.replace('_', ' ').title()}: Contraction {performance_msg} - reached {actual_min:.1f}° (target: {best_min}°-{acceptable_min}°)",
                        'severity': severity,
                        'actual_angle': actual_min,
                        'optimal_range': rule['optimal_min_range'],
                        'phase': 'rep_complete'
                    })
            
            acceptable_max, best_max = rule['optimal_max_range']
            
            if actual_max < acceptable_max:
                frames_since_check = self.frame_count - self.rom_violations[primary_joint]['last_extension_check']
                if frames_since_check >= rule.get('frequency_penalty', 30):
                    
                    self.rom_violations[primary_joint]['insufficient_extension'] += 1
                    self.rom_violations[primary_joint]['last_extension_check'] = self.frame_count
                    
                    if actual_max < rule.get('severe_extension_threshold', 140):
                        severity = 'high'
                        penalty_multiplier = 2.0
                        performance_msg = "severely limited"
                    else:
                        severity = 'medium'
                        penalty_multiplier = 1.0
                        performance_msg = "limited"
                    
                    deductions.append({
                        'type': 'rom_extension',
                        'joint': primary_joint,
                        'amount': rule['extension_deduction'] * penalty_multiplier,
                        'reason': f"{primary_joint.replace('_', ' ').title()}: Extension {performance_msg} - reached {actual_max:.1f}° (target: {acceptable_max}°-{best_max}°)",
                        'severity': severity,
                        'actual_angle': actual_max,
                        'optimal_range': rule['optimal_max_range'],
                        'phase': 'rep_complete'
                    })
        
        return deductions

    def _evaluate_angles(self, angles):
        """Angle evaluation - only when tracking is active"""
        if not self.tracking_started:
            return []
            
        deductions = []
        current_rules = self.angle_rules.get(self.exercise_type, self.angle_rules["dumbell-curl"])
        
        for angle_name, angle_value in angles.items():
            if angle_name in current_rules:
                rule = current_rules[angle_name]
                
                if angle_value < rule["min"] or angle_value > rule["max"]:
                    deduction_amount = rule["deduction"]
                    
                    if self.frame_count % 10 == 0:
                        deductions.append({
                            'type': 'angle',
                            'joint': angle_name,
                            'amount': deduction_amount,
                            'reason': f"{angle_name.replace('_', ' ').title()}: {angle_value:.1f}° (should be {rule['min']}-{rule['max']}°)",
                            'severity': self._calculate_severity(angle_value, rule["min"], rule["max"])
                        })
        
        return deductions
    
    def set_joint_center_mode(self, joint_name, center_mode):
        if joint_name in self.steadiness_rules["joints"]:
            old_mode = self.steadiness_rules["joints"][joint_name].get("center_mode", "reset")
            self.steadiness_rules["joints"][joint_name]["center_mode"] = center_mode
            
            if center_mode == "fixed" and old_mode != "fixed":
                self.steadiness_centers[joint_name] = None
                print(f"Switched {joint_name} to fixed center mode - center will be recalculated")
            elif center_mode == "reset" and old_mode != "reset":
                print(f"Switched {joint_name} to reset center mode - center will update every 60 frames")
        else:
            print(f"Joint {joint_name} not found in steadiness rules")

    def set_global_center_mode(self, center_mode):
        self.center_mode = center_mode
        print(f"Global center mode set to: {center_mode}")

    def _evaluate_velocities(self, velocities):
        """Velocity evaluation - only when tracking is active"""
        if not self.tracking_started:
            return []
            
        deductions = []
        
        for joint_name, velocity_data in velocities.items():
            if 'speed_category' not in velocity_data:
                continue
                
            speed_category = velocity_data['speed_category']
            
            if joint_name in self.velocity_rules["excessive_speed"]["joints"]:
                joint_rule = self.velocity_rules["excessive_speed"]["joints"][joint_name]
                
                if speed_category not in joint_rule["allowed_speeds"]:
                    if joint_name not in self.fast_movement_counter:
                        self.fast_movement_counter[joint_name] = {"fast": 0, "medium": 0}
                    
                    if speed_category == 'fast':
                        self.fast_movement_counter[joint_name]["fast"] += 1
                        self.fast_movement_counter[joint_name]["medium"] = 0
                    elif speed_category == 'medium':
                        self.fast_movement_counter[joint_name]["medium"] += 1
                        self.fast_movement_counter[joint_name]["fast"] = 0
                    
                    max_fast = self.velocity_rules["excessive_speed"]["max_fast_frames"]
                    if self.fast_movement_counter[joint_name]["fast"] > max_fast:
                        deductions.append({
                            'type': 'velocity',
                            'joint': joint_name,
                            'amount': joint_rule["deduction"],
                            'reason': f"{joint_name.replace('_', ' ').title()}: Moving too fast for too long",
                            'severity': 'high'
                        })
                    
                    max_medium = self.velocity_rules["excessive_speed"].get("max_medium_frames", 8)
                    if (speed_category == 'medium' and 
                        self.fast_movement_counter[joint_name]["medium"] > max_medium):
                        deductions.append({
                            'type': 'velocity',
                            'joint': joint_name,
                            'amount': joint_rule["deduction"] * 0.5,
                            'reason': f"{joint_name.replace('_', ' ').title()}: Moving at medium speed for too long",
                            'severity': 'medium'
                        })
                        
                else:
                    if joint_name not in self.fast_movement_counter:
                        self.fast_movement_counter[joint_name] = {"fast": 0, "medium": 0}
                    else:
                        self.fast_movement_counter[joint_name]["fast"] = 0
                        self.fast_movement_counter[joint_name]["medium"] = 0
        
        return deductions
    
    def _calculate_severity(self, value, min_val, max_val):
        if min_val <= value <= max_val:
            return 'none'
        
        if value < min_val:
            deviation = min_val - value
        else:
            deviation = value - max_val
        
        range_size = max_val - min_val
        deviation_ratio = deviation / range_size
        
        if deviation_ratio > 0.5:
            return 'high'
        elif deviation_ratio > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_feedback(self, frame_deductions):
        """Generate user-friendly feedback messages"""
        feedback = []
        
        angle_issues = [d for d in frame_deductions if d['type'] == 'angle']
        velocity_issues = [d for d in frame_deductions if d['type'] == 'velocity']
        steadiness_issues = [d for d in frame_deductions if d['type'] == 'steadiness']
        
        high_severity = [d for d in frame_deductions if d.get('severity') == 'high']
        
        if high_severity:
            feedback.append("⚠️ CRITICAL: " + high_severity[0]['reason'])
        
        if angle_issues:
            feedback.append("📐 Form: " + angle_issues[0]['reason'])
        
        if velocity_issues:
            feedback.append("⚡ Speed: " + velocity_issues[0]['reason'])
        
        if steadiness_issues:
            feedback.append("🎯 Stability: " + steadiness_issues[0]['reason'])
        
        rom_issues = [d for d in frame_deductions if d['type'] in ['rom_contraction', 'rom_extension']]
        
        if rom_issues:
            feedback.append("🎯 ROM: " + rom_issues[0]['reason'])

        return feedback
    
    def get_steadiness_debug_info(self):
        """Get debug information about steadiness tracking"""
        debug_info = {}
        
        for joint_name in self.steadiness_rules["joints"]:
            rule = self.steadiness_rules["joints"][joint_name]
            debug_info[joint_name] = {
                'center_position': self.steadiness_centers.get(joint_name),
                'center_mode': rule.get('center_mode', 'reset'),
                'violation_count': self.steadiness_violations.get(joint_name, 0),
                'history_length': len(self.steadiness_history.get(joint_name, [])),
                'rule': rule,
                'tracking_started': self.tracking_started  # NEW: Show if tracking is active
            }
        
        return debug_info
    
    def get_score_display(self):
        """Get formatted score display with tracking status"""
        percentage = (self.current_points / self.starting_points) * 100
        
        if percentage >= 90:
            grade = "A"
            color = (0, 255, 0)  # Green
        elif percentage >= 80:
            grade = "B"
            color = (0, 255, 255)  # Yellow
        elif percentage >= 70:
            grade = "C"
            color = (0, 165, 255)  # Orange
        else:
            grade = "D"
            color = (0, 0, 255)  # Red
        
        # Modify display text based on tracking status
        if not self.tracking_started:
            display_text = f"Score: Ready? ({int(self.current_points)}/100)"
            color = (128, 128, 128)  # Gray when not tracking
        else:
            display_text = f"Score: {int(self.current_points)}/100 ({grade})"
        
        return {
            'points': int(self.current_points),
            'percentage': percentage,
            'grade': grade,
            'color': color,
            'display_text': display_text,
            'tracking_started': self.tracking_started  # NEW: Include tracking status
        }
    
    def reset_score(self):
        """Reset score to starting value and tracking state"""
        self.current_points = self.starting_points
        self.deduction_history = []
        self.frame_count = 0
        self.fast_movement_counter = {}
        self.rep_counter.reset_count()
        self.rom_violations = {}
        
        # NEW: Reset tracking state
        self.tracking_started = False
        self.pre_ready_frames = 0
        
        # Reset steadiness tracking
        for joint_name in list(self.steadiness_centers.keys()):
            joint_rule = self.steadiness_rules["joints"].get(joint_name, {})
            center_mode = joint_rule.get("center_mode", "reset")
            
            # Always reset center when score is reset (user will need to get ready again)
            self.steadiness_centers[joint_name] = None
        
        self.steadiness_violations = {}
        self.steadiness_history = {}
        
        # Reset ready position detector too
        if self.ready_position_detector:
            self.ready_position_detector.reset()
        
        print("Fitness score and tracking state reset - user must assume ready position again")
    
    def get_rep_info(self):
        """Get current rep counting information"""
        return self.rep_counter.get_rep_summary()
    
    def get_summary_report(self):
        """Get detailed performance summary including tracking status"""
        if not self.tracking_started:
            return f"""
PERFORMANCE SUMMARY
===================
Status: Waiting for ready position
Preparation Frames: {self.pre_ready_frames}
Score: {int(self.current_points)}/100 (not yet tracking)

Ready for exercise tracking once proper position is assumed.
            """
        
        if not self.deduction_history:
            base_report = "Perfect form! No deductions."
        else:
            total_deductions = sum([entry['total_deduction'] for entry in self.deduction_history])
            
            angle_deductions = sum([
                deduction['amount'] for entry in self.deduction_history 
                for deduction in entry['deductions'] if deduction['type'] == 'angle'
            ])
            velocity_deductions = sum([
                deduction['amount'] for entry in self.deduction_history 
                for deduction in entry['deductions'] if deduction['type'] == 'velocity'
            ])
            steadiness_deductions = sum([
                deduction['amount'] for entry in self.deduction_history 
                for deduction in entry['deductions'] if deduction['type'] == 'steadiness'
            ])
            
            base_report = f"""
PERFORMANCE SUMMARY
===================
Final Score: {int(self.current_points)}/100 ({((self.current_points/self.starting_points)*100):.1f}%)
Total Deductions: {total_deductions} points
Preparation Frames: {self.pre_ready_frames}
Tracking Frames: {self.frame_count}

Breakdown:
- Form Issues: -{angle_deductions} points
- Movement Issues: -{velocity_deductions} points
- Stability Issues: -{steadiness_deductions} points
- Issues Detected: {len(self.deduction_history)} frames

Steadiness Monitoring:
- Monitored Joints: {list(self.steadiness_rules['joints'].keys())}
            """
        
        rep_summary = self.rep_counter.get_rep_summary()
        rep_report = f"""

REP COUNTING SUMMARY
===================
Exercise: {rep_summary['exercise']}
Total Reps: {rep_summary['total_reps']}
Half Reps: {rep_summary['half_reps']}
Current Phase: {rep_summary['current_phase']}
Tracking Angle: {rep_summary['tracking_angle']}
Angle Range: {rep_summary['current_range']}
        """
        
        rom_summary = self.get_rom_violations_summary()
        rom_report = f"""

RANGE OF MOTION ANALYSIS
=======================
Total Contraction Issues: {rom_summary['total_contraction_violations']}
Total Extension Issues: {rom_summary['total_extension_violations']}

Per Joint:"""
        
        for joint_name, violations in rom_summary['violations'].items():
            rom_report += f"""
- {joint_name.replace('_', ' ').title()}: {violations['insufficient_contraction']} contraction, {violations['insufficient_extension']} extension issues"""

        return base_report + rep_report + rom_report
def draw_score_display_with_tracking_status(image, fitness_scorer, position=(900, 50)):
    """
    Enhanced score display that shows tracking status
    """
    score_info = fitness_scorer.get_score_display()
    tracking_started = score_info.get('tracking_started', False)
    x, y = position
    
    # Create semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x-5, y-25), (x+250, y+80), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Draw score text with tracking status
    if tracking_started:
        cv2.putText(image, score_info['display_text'], (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_info['color'], 2)
        status_text = "🎯 TRACKING ACTIVE"
        status_color = (0, 255, 0)
    else:
        cv2.putText(image, score_info['display_text'], (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        status_text = "⏳ WAITING FOR READY"
        status_color = (255, 255, 0)
    
    # Draw tracking status
    cv2.putText(image, status_text, (x, y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    
    # Draw progress bar (only when tracking)
    if tracking_started:
        bar_width = 200
        bar_height = 10
        bar_x = x
        bar_y = y + 15
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
        
        # Progress bar
        progress_width = int((score_info['percentage'] / 100) * bar_width)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), score_info['color'], -1)
        
        # Progress text
        cv2.putText(image, f"{score_info['percentage']:.1f}%", (bar_x + bar_width + 10, bar_y + 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        # Show "waiting" indicator instead of progress bar
        cv2.putText(image, "Score tracking will begin when ready position is detected", 
                    (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    return image

def draw_feedback_messages(image, feedback_messages, position=(750, 150)):
    """
    Draw feedback messages on the image
    
    Args:
        image: OpenCV image  
        feedback_messages: List of feedback strings
        position: (x, y) starting position
    """
    if not feedback_messages:
        return image
    
    x, y_start = position
    line_height = 25
    
    # Create background
    bg_height = len(feedback_messages) * line_height + 10
    overlay = image.copy()
    cv2.rectangle(overlay, (x-5, y_start-20), (x+500, y_start + bg_height), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Draw messages
    for i, message in enumerate(feedback_messages):
        y = y_start + (i * line_height)
        
        # Color based on message type
        if "CRITICAL" in message:
            color = (0, 0, 255)  # Red
        elif "Form" in message:
            color = (0, 165, 255)  # Orange
        elif "Speed" in message:
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 255, 255)  # White
        
        cv2.putText(image, message, (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image

class VelocityTracker:
    def __init__(self, smoothing_window=5):
        self.previous_keypoints = None
        self.previous_timestamp = None
        self.velocity_history = [] 
        self.smoothing_window = smoothing_window
        self.body_calculator = BodyProportionCalculator()
        
        # MODIFIED: Create ready detector first, then pass it to fitness scorer
        self.ready_detector = ReadyPositionDetector(exercise_type="dumbell-curl", velocity_tracker=self)
        self.fitness_scorer = FitnessScorer(exercise_type="dumbell-curl", starting_points=100, ready_position_detector=self.ready_detector)
        self.rpe_calculator = IntegratedRPECalculator(exercise_type="dumbell-curl")
        
        # Store last frame data for ready position detection
        self.last_angles = None
        self.last_keypoints = None 
        self.last_confidences = None
        self.frame_count = 0

    def get_integrated_rpe_analysis(self, rep_counter):
        """
        Get RPE analysis using integrated data from all system components
        ENHANCED: Passes required data for ready position detection
        """
        try:
            # Get all required data
            rep_data = rep_counter.get_rep_summary()
            rom_data = rep_counter.get_range_of_motion_data()
            form_data = self.fitness_scorer.get_score_display()
            body_measurements = {'reference_scale': self.body_calculator.reference_scale}
            current_velocities = self.velocity_history[-5:] if len(self.velocity_history) >= 5 else self.velocity_history
            
            # NEW: Get additional data needed for ready position detection
            # These should come from your main processing loop - for now we'll handle missing data gracefully
            angles = getattr(self, 'last_angles', None)  # You'll need to store this in VelocityTracker
            keypoints = getattr(self, 'last_keypoints', None)  # You'll need to store this in VelocityTracker  
            confidences = getattr(self, 'last_confidences', None)  # You'll need to store this in VelocityTracker
            frame_count = getattr(self, 'frame_count', 0)  # You'll need to store this in VelocityTracker
            
            # Update RPE data with ready position check
            self.rpe_calculator.update_rpe_data(
                rep_counter_data={'total_reps': rep_data['total_reps'], 'current_phase': rep_data['current_phase']},
                fitness_scorer_data={'current_points': self.fitness_scorer.current_points},
                velocity_data=current_velocities[-1] if current_velocities else {},
                body_measurements=body_measurements,
                # NEW: Pass ready position detection data
                angles=angles,
                keypoints=keypoints,
                confidences=confidences,
                frame_count=frame_count
            )
            
            # Calculate integrated RPE
            rpe_result = self.rpe_calculator.calculate_rpe(
                rep_counter=rep_counter,
                fitness_scorer=self.fitness_scorer,
                velocity_tracker=self
            )
            
            return rpe_result
            
        except Exception as e:
            print(f"Error in integrated RPE analysis: {e}")
            return {
                'rpe': 5.0,
                'confidence': 'error',
                'interpretation': f'RPE calculation error: {str(e)}',
                'breakdown': {},
                'fatigue_indicators': [],
                'data_sources': {}
            }


    def calculate_velocities(self, current_keypoints, current_timestamp, confidences, confidence_threshold=0.3):
        """Calculate velocities for all keypoints with adaptive scaling"""
        velocities = {}
        
        # *** NEW CODE TO ADD ***
        # Store the current frame data for RPE ready position detection
        self.last_angles = None  # We'll calculate this below
        self.last_keypoints = current_keypoints.copy() if current_keypoints is not None else None
        self.last_confidences = confidences.copy() if confidences is not None else None
        self.frame_count += 1  # Increment frame counter
        
        if self.previous_keypoints is None or self.previous_timestamp is None:
            # First frame - initialize
            self.previous_keypoints = current_keypoints.copy()
            self.previous_timestamp = current_timestamp
            # IMPORTANT: Calculate body proportions even on first frame
            self.body_calculator.update_body_proportions(current_keypoints, confidences, confidence_threshold)
            
            # *** NEW CODE TO ADD ***
            # Calculate angles for the first frame too (needed for RPE)
            if current_keypoints is not None and confidences is not None:
                self.last_angles = calculate_body_angles(current_keypoints, confidences, confidence_threshold)
            
            return velocities
        
        # Calculate time difference
        dt = current_timestamp - self.previous_timestamp
        if dt <= 0:
            return velocities
        
        # UPDATE BODY PROPORTIONS: This is crucial for adaptive scaling
        body_measurements = self.body_calculator.update_body_proportions(
            current_keypoints, confidences, confidence_threshold
        )
        
        # *** NEW CODE TO ADD ***
        # Calculate angles for RPE ready position detection
        if current_keypoints is not None and confidences is not None:
            self.last_angles = calculate_body_angles(current_keypoints, confidences, confidence_threshold)
        
        # GET ADAPTIVE THRESHOLDS: These change based on person size and video resolution
        velocity_thresholds = self.body_calculator.get_velocity_thresholds()
        
        # Keypoint names for reference
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'neck', 'chest', 'mid_spine', 'lower_spine', 'tail_bone',
            'left_toes', 'right_toes'
        ]
        
        for i, (current_point, prev_point, conf, name) in enumerate(
            zip(current_keypoints, self.previous_keypoints, confidences, keypoint_names)
        ):
            if conf > confidence_threshold:
                # Calculate displacement
                dx = current_point[0] - prev_point[0]
                dy = current_point[1] - prev_point[1]
                
                # Calculate velocity components
                vx = dx / dt
                vy = dy / dt
                
                # Calculate speed (magnitude)
                speed = np.sqrt(vx**2 + vy**2)
                
                # CLASSIFY SPEED USING ADAPTIVE THRESHOLDS
                if speed < velocity_thresholds['slow']:
                    speed_category = 'slow'
                elif speed < velocity_thresholds['medium']:
                    speed_category = 'medium'
                else:
                    speed_category = 'fast'
                
                velocities[name] = {
                    'vx': vx,
                    'vy': vy, 
                    'speed': speed,
                    'speed_category': speed_category,  # NEW: Add speed classification
                    'confidence': conf,
                    'thresholds': velocity_thresholds  # NEW: Include current thresholds for debugging
                }
        
        # Store current as previous for next iteration
        self.previous_keypoints = current_keypoints.copy()
        self.previous_timestamp = current_timestamp
        
        # Add to history for smoothing
        self.velocity_history.append(velocities)
        if len(self.velocity_history) > self.smoothing_window:
            self.velocity_history.pop(0)
        
        return self._smooth_velocities()
    
    def _smooth_velocities(self):
        """Apply smoothing to velocity measurements - ENHANCED VERSION"""
        if len(self.velocity_history) < 2:
            return self.velocity_history[-1] if self.velocity_history else {}
        
        smoothed = {}
        
        # Get all keypoint names from recent history
        all_keypoints = set()
        for frame_velocities in self.velocity_history:
            all_keypoints.update(frame_velocities.keys())
        
        for keypoint_name in all_keypoints:
            vx_values = []
            vy_values = []
            speed_values = []
            categories = []  # NEW: Track speed categories for smoothing
            
            for frame_velocities in self.velocity_history:
                if keypoint_name in frame_velocities:
                    vx_values.append(frame_velocities[keypoint_name]['vx'])
                    vy_values.append(frame_velocities[keypoint_name]['vy'])
                    speed_values.append(frame_velocities[keypoint_name]['speed'])
                    if 'speed_category' in frame_velocities[keypoint_name]:
                        categories.append(frame_velocities[keypoint_name]['speed_category'])
            
            if vx_values:  # Only smooth if we have data
                # Find the most recent confidence value and thresholds
                recent_confidence = 0.0
                recent_thresholds = {'slow': 50, 'medium': 150, 'fast': 300}  # fallback
                
                for frame_velocities in reversed(self.velocity_history):
                    if keypoint_name in frame_velocities:
                        recent_confidence = frame_velocities[keypoint_name]['confidence']
                        if 'thresholds' in frame_velocities[keypoint_name]:
                            recent_thresholds = frame_velocities[keypoint_name]['thresholds']
                        break
                
                # Smooth the speed and reclassify
                smoothed_speed = np.mean(speed_values)
                
                # Classify smoothed speed
                if smoothed_speed < recent_thresholds['slow']:
                    smoothed_category = 'slow'
                elif smoothed_speed < recent_thresholds['medium']:
                    smoothed_category = 'medium'
                else:
                    smoothed_category = 'fast'
                
                smoothed[keypoint_name] = {
                    'vx': np.mean(vx_values),
                    'vy': np.mean(vy_values),
                    'speed': smoothed_speed,
                    'speed_category': smoothed_category,
                    'confidence': recent_confidence,
                    'thresholds': recent_thresholds
                }
        
        return smoothed
    
class BodyProportionCalculator:
    """Calculate body proportions for adaptive velocity scaling"""
    
    def __init__(self):
        self.shoulder_elbow_distances = []
        self.hip_knee_distances = []
        self.smoothing_window = 5
        self.reference_scale = None
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_body_proportions(self, keypoints, confidences, confidence_threshold=0.3):
        """
        Calculate current body proportions using shoulder-elbow and hip-knee distances
        
        Args:
            keypoints: Array of keypoint coordinates [24, 2]
            confidences: Array of confidence scores [24]
            confidence_threshold: Minimum confidence for valid measurements
        
        Returns:
            dict: Current body measurements and reference scale
        """
        measurements = {}
        
        # Keypoint indices for reference:
        # 5: left_shoulder, 6: right_shoulder
        # 7: left_elbow, 8: right_elbow  
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        
        # Calculate shoulder to elbow distances (both sides)
        shoulder_elbow_distances = []
        
        # Left side: shoulder to elbow
        if (confidences[5] > confidence_threshold and 
            confidences[7] > confidence_threshold):
            left_shoulder_elbow = self.calculate_distance(keypoints[5], keypoints[7])
            shoulder_elbow_distances.append(left_shoulder_elbow)
            measurements['left_shoulder_elbow'] = left_shoulder_elbow
        
        # Right side: shoulder to elbow  
        if (confidences[6] > confidence_threshold and 
            confidences[8] > confidence_threshold):
            right_shoulder_elbow = self.calculate_distance(keypoints[6], keypoints[8])
            shoulder_elbow_distances.append(right_shoulder_elbow)
            measurements['right_shoulder_elbow'] = right_shoulder_elbow
        
        # Calculate hip to knee distances (both sides)
        hip_knee_distances = []
        
        # Left side: hip to knee
        if (confidences[11] > confidence_threshold and 
            confidences[13] > confidence_threshold):
            left_hip_knee = self.calculate_distance(keypoints[11], keypoints[13])
            hip_knee_distances.append(left_hip_knee)
            measurements['left_hip_knee'] = left_hip_knee
        
        # Right side: hip to knee
        if (confidences[12] > confidence_threshold and 
            confidences[14] > confidence_threshold):
            right_hip_knee = self.calculate_distance(keypoints[12], keypoints[14])
            hip_knee_distances.append(right_hip_knee)
            measurements['right_hip_knee'] = right_hip_knee
        
        # Store measurements for smoothing
        if shoulder_elbow_distances:
            avg_shoulder_elbow = np.mean(shoulder_elbow_distances)
            self.shoulder_elbow_distances.append(avg_shoulder_elbow)
            measurements['avg_shoulder_elbow'] = avg_shoulder_elbow
        
        if hip_knee_distances:
            avg_hip_knee = np.mean(hip_knee_distances)
            self.hip_knee_distances.append(avg_hip_knee)
            measurements['avg_hip_knee'] = avg_hip_knee
        
        # Maintain smoothing window
        if len(self.shoulder_elbow_distances) > self.smoothing_window:
            self.shoulder_elbow_distances.pop(0)
        if len(self.hip_knee_distances) > self.smoothing_window:
            self.hip_knee_distances.pop(0)
        
        # Calculate smoothed reference scale
        self.reference_scale = self._calculate_reference_scale()
        measurements['reference_scale'] = self.reference_scale
        
        return measurements
    
    def _calculate_reference_scale(self):
        """
        Calculate a reference scale based on smoothed body measurements
        
        The reference scale combines shoulder-elbow and hip-knee measurements
        to create a robust body size indicator that adapts to:
        - Different video resolutions
        - Different person sizes  
        - Different distances from camera
        
        Returns:
            float: Reference scale in pixels, or None if insufficient data
        """
        scale_components = []
        
        # Use smoothed shoulder-elbow distance
        if self.shoulder_elbow_distances:
            avg_shoulder_elbow = np.mean(self.shoulder_elbow_distances)
            scale_components.append(avg_shoulder_elbow)
        
        # Use smoothed hip-knee distance
        if self.hip_knee_distances:
            avg_hip_knee = np.mean(self.hip_knee_distances)
            scale_components.append(avg_hip_knee)
        
        if not scale_components:
            return None
        
        # Combine measurements with weighted average
        # Hip-knee tends to be more stable, so weight it higher
        if len(scale_components) == 2:  # Both measurements available
            reference_scale = (scale_components[0] * 0.4 + scale_components[1] * 0.6)
        else:  # Only one measurement available
            reference_scale = scale_components[0]
        
        return reference_scale
    
    def get_velocity_thresholds(self, base_slow=0.45, base_medium=2.15, base_fast=5):
        """
        Calculate adaptive velocity thresholds based on current body scale
        
        Args:
            base_slow: Base threshold for slow movement (relative to body scale)
            base_medium: Base threshold for medium movement (relative to body scale)  
            base_fast: Base threshold for fast movement (relative to body scale)
        
        Returns:
            dict: Adaptive thresholds in pixels/second
        """
        if self.reference_scale is None:
            # Fallback to default pixel-based thresholds
            return {
                'slow': 50,
                'medium': 150, 
                'fast': 300
            }
        
        # Scale thresholds based on body proportions
        return {
            'slow': self.reference_scale * base_slow,
            'medium': self.reference_scale * base_medium,
            'fast': self.reference_scale * base_fast
        }


def velocity_info_box(image, velocities, body_measurements=None, position_offset=(200, 10)):
    """Display velocity information on image with adaptive scaling info"""
    x_offset, y_start = position_offset
    y_offset = 12
    y_current = y_start
    
    # Sort by speed for better visualization
    sorted_velocities = sorted(velocities.items(), 
                              key=lambda x: x[1]['speed'], reverse=True)
    
    # Display title with reference scale info
    cv2.putText(image, "Adaptive Velocities:", (x_offset, y_current), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y_current += y_offset + 5
    
    # FIXED: Proper null checking for body measurements
    if body_measurements and body_measurements.get('reference_scale') is not None:
        scale_value = body_measurements['reference_scale']
        scale_text = f"Body Scale: {scale_value:.1f}px"
        scale_color = (0, 255, 0)  # Green when available
    else:
        scale_text = "Body Scale: Calculating..."
        scale_color = (0, 255, 255)  # Yellow when calculating
    
    cv2.putText(image, scale_text, (x_offset, y_current), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, scale_color, 1)
    y_current += y_offset
    
    # Display thresholds (get from first velocity entry)
    if velocities:
        first_velocity = next(iter(velocities.values()))
        if 'thresholds' in first_velocity:
            thresholds = first_velocity['thresholds']
            threshold_text = f"Thresholds: S<{thresholds['slow']:.0f} M<{thresholds['medium']:.0f} F>{thresholds['medium']:.0f}"
            cv2.putText(image, threshold_text, (x_offset, y_current), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (92, 92, 205), 1)
            y_current += y_offset + 3
    
    # Display keypoint velocities with adaptive color coding
    for i, (keypoint_name, velocity) in enumerate(sorted_velocities):
        speed = velocity['speed']
        category = velocity.get('speed_category', 'unknown')
        
        # Color based on ADAPTIVE category
        if category == 'slow':
            color = (113, 204, 46)  # Green
        elif category == 'medium':
            color = (226, 173, 93)  # Blue/Orange  
        elif category == 'fast':
            color = (60, 76, 231)   # Red
        else:
            color = (128, 128, 128) # Gray for unknown
        
        # Format text with category indicator
        text = f"{keypoint_name.replace('_', ' ').title()}: {speed:.1f} ({category[0].upper()})"
        cv2.putText(image, text, (x_offset, y_current), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        y_current += y_offset
    
def calculate_angle(point_a, point_b, point_c, confidence_threshold=0.3, confidences=None):
    if confidences is not None:
        if (confidences[0] < confidence_threshold or #[0] represents point_a because of weird naming stuf
            confidences[1] < confidence_threshold or 
            confidences[2] < confidence_threshold):
            return None
        
    a = np.array(point_a)  #Converts the original point coordinates of point_a, point_b, and point_c to numpy arrays which can be used in pose estimation math calcs
    b = np.array(point_b) 
    c = np.array(point_c)    

    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c
     # b is the vertex of the angle 
    try:
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # Calculate cosine of the angle
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors
        angle_degrees = np.degrees(angle)  # Convert to degrees 
        return angle_degrees  # Return the angle in degrees
    except:
        return None  # Return None if angle calculation fails
    
def calculate_body_angles(keypoints, confidences, confidence_threshold=0.3):
    angles = {} #Dictionary to store angles 

    def get_angle(indx_A, indx_B, indx_C, angle_name): #Creates the angle points and the name of the angle which will be calculated
        if (indx_A < len(keypoints) and indx_B < len(keypoints) and indx_C < len(keypoints)): # this makes it so that the angle is only calculated if the keypoints are within the range of the keypoints
            confidence_levels = [confidences[indx_A], confidences[indx_B], confidences[indx_C]] # gets the confidence levels of the keypoints
            angle = calculate_angle(keypoints[indx_A], keypoints[indx_B], keypoints[indx_C], confidence_threshold, confidence_levels) # usees the above functio in order to calcualte the angle
            if angle is not None: # if the angle is succesfully calculated then this happens
                angles[angle_name] = angle # adds angle to the dictionary
        return angles.get(angle_name, None)  # Returns the angle or None if not calculated
    

    #0:nose
    #1:left_eye
    #2:right_eye
    #3:left_ear
    #4:right_ear
    #5:left_shoulder
    #6:right_shoulder
    #7:left_elbow
    #8:right_elbow 
    #9:left_wrist
    #10:right_wrist
    #11:left_hip
    #12:right_hip
    #13:left_knee
    #14:right_knee
    #15:left_ankle
    #16:right_ankle
    #17:neck
    #18:chest
    #19:mid_spine
    #20:lower_spine
    #21:tail_bone
    #22:left_toes
    #23:right_toes

    #Angles for various body parts:
   # Left arm angles
    get_angle(5, 7, 9, 'left_elbow')      # shoulder-elbow-wrist
    get_angle(17, 5, 7, 'left_shoulder_elbow')   # neck-shoulder-elbow
    get_angle(7, 5, 11, 'left_armpit')     # elbow-shoulder-hip
    get_angle(9, 5, 11, 'left_armpit_wrist')  # shoulder-hip-wrist
    get_angle(17, 5, 9, 'left_shoulder_wrist')  # neck-shoulder-wrist
    get_angle(9,5,3, 'left_wrist_ear')  # wrist-shoulder-ear
    get_angle(7, 5, 3, 'left_elbow_ear')  # elbow-shoulder-ear
    get_angle(7,5,6, 'left_elbow_right_shoulder')  # elbow-shoulder-right_shoulder
    get_angle(9,5,6, 'left_wrist_right_shoulder')  # wrist-shoulder-right_shoulder
    get_angle(7,5,18, 'left_elbow_chest')  # elbow-shoulder-chest
    get_angle(9,5,18, 'left_wrist_chest')  # wrist-shoulder-chest
    get_angle(7,5,19, 'left_elbow_mid_spine')  # elbow-shoulder-mid_spine
    get_angle(9,5,19, 'left_wrist_mid_spine')  # wrist-shoulder-mid_spine
    get_angle(7,5,20, 'left_elbow_lower_spine')  # elbow-shoulder-lower_spine
    get_angle(9,5,20, 'left_wrist_lower_spine')  # wrist-shoulder-lower_spine
    get_angle(7,5,21, 'left_elbow_tail_bone')  # elbow-shoulder-tail_bone
    get_angle(9,5,21, 'left_wrist_tail_bone')  # wrist-shoulder-tail_bone
    
    # Right arm angles  
    get_angle(6, 8, 10, 'right_elbow')     # shoulder-elbow-wrist
    get_angle(17, 6, 8, 'right_shoulder_elbow')  # neck-shoulder-elbow
    get_angle(8, 6, 12, 'right_armpit')    # elbow-shoulder-hip
    get_angle(10, 6, 12, 'right_armpit_wrist')  # shoulder-hip-wrist
    get_angle(17, 6, 10, 'right_shoulder_wrist')  # neck-shoulder-wrist
    get_angle(10,6,4, 'right_wrist_ear')  # wrist-shoulder-ear
    get_angle(8, 6, 4, 'right_elbow_ear')  # elbow-shoulder-ear
    get_angle(8,6,5, 'right_elbow_left_shoulder')  # elbow-shoulder-left_shoulder
    get_angle(10,6,5, 'right_wrist_left_shoulder')  # wrist-shoulder-left_shoulder
    get_angle(8,6,18, 'right_elbow_chest')  # elbow-shoulder-chest
    get_angle(10,6,18, 'right_wrist_chest')  # wrist-shoulder-chest
    get_angle(8,6,19, 'right_elbow_mid_spine')  # elbow-shoulder-mid_spine
    get_angle(10,6,19, 'right_wrist_mid_spine')  # wrist-shoulder-mid_spine
    get_angle(8,6,20, 'right_elbow_lower_spine')  # elbow-shoulder-lower_spine
    get_angle(10,6,20, 'right_wrist_lower_spine')  # wrist-shoulder-lower_spine
    get_angle(8,6,21, 'right_elbow_tail_bone')  # elbow-shoulder-tail_bone
    get_angle(10,6,21, 'right_wrist_tail_bone')  # wrist-shoulder-tail_bone
    
    #0:nose
    #1:left_eye
    #2:right_eye
    #3:left_ear
    #4:right_ear
    #5:left_shoulder
    #6:right_shoulder
    #7:left_elbow
    #8:right_elbow 
    #9:left_wrist
    #10:right_wrist
    #11:left_hip
    #12:right_hip
    #13:left_knee
    #14:right_knee
    #15:left_ankle
    #16:right_ankle
    #17:neck
    #18:chest
    #19:mid_spine
    #20:lower_spine
    #21:tail_bone
    #22:left_toes
    #23:right_toes

    # LEG ANGLES
    # Left leg angles
    get_angle(11, 13, 15, 'left_knee')     # hip-knee-ankle
    get_angle(11, 13, 22, 'left_knee_toes')  # hip-knee-toes
    get_angle(5, 11, 13, 'left_hip')       # shoulder-hip-knee
    get_angle(5,11,15, 'left_hip_ankle')  # shoulder-hip-ankle
    get_angle(5,11,22, 'left_hip_toes')  # shoulder-hip-toes
    get_angle(13, 15, 22, 'left_ankle')    # knee-ankle-toes
    
    # Right leg angles
    get_angle(12, 14, 16, 'right_knee')    # hip-knee-ankle
    get_angle(12, 14, 23, 'right_knee_toes')  # hip-knee-toes
    get_angle(6, 12, 14, 'right_hip')      # shoulder-hip-knee
    get_angle(6,12,16, 'right_hip_ankle')  # shoulder-hip-ankle
    get_angle(6,12,23, 'right_hip_toes')  # shoulder-hip-toes
    get_angle(14, 16, 23, 'right_ankle')   # knee-ankle-toes
    
    # TORSO/SPINE ANGLES
    get_angle(17, 18, 19, 'chest')   # neck-chest-mid_spine
    get_angle(18, 19, 20, 'mid_spine_chest')     # chest-mid_spine-lower_spine
    get_angle(17,19,20,'mid_spine' ) # neck-mid_spine-lower_spine
    get_angle(19, 20, 21, 'lower_spine')   # mid_spine-lower_spine-tail_bone
    get_angle(17,20,21, 'tail_bone')  # neck-lower_spine-tail_bone
    get_angle(17, 19, 21, 'spine_alignment')     # neck-mid_spine-tail_bone
    get_angle(0,17,19, 'neck_alignment')  # nose-neck-mid_spine
    get_angle(0,17,20, 'neck_lower_spine')  # nose-neck-lower_spine
    get_angle(0,17,21, 'neck_tail_bone')  # nose-neck-tail_bone
    
    # BODY ALIGNMENT ANGLES
    get_angle(5, 17, 6, 'shoulder_alignment')    # left_shoulder-neck-right_shoulder
    get_angle(5, 18, 6, 'shoulder_chest_alignment')  # left_shoulder-chest-right_shoulder
    get_angle(5, 19, 6, 'shoulder_mid_spine_alignment')  # left_shoulder-mid_spine-right_shoulder
    get_angle(5, 20, 6, 'shoulder_lower_spine_alignment')  # left_shoulder-lower_spine-right_shoulder
    get_angle(5, 21, 6, 'shoulder_tail_bone_alignment')  # left_shoulder-tail_bone-right_shoulder
    get_angle(7, 17, 8, 'elbow_alignment')       # left_elbow-neck-right_elbow
    get_angle(7, 18, 8, 'elbow_chest_alignment')  # left_elbow-chest-right_elbow
    get_angle(7, 19, 8, 'elbow_mid_spine_alignment')  # left_elbow-mid_spine-right_elbow
    get_angle(7, 20, 8, 'elbow_lower_spine_alignment')  # left_elbow-lower_spine-right_elbow
    get_angle(7, 21, 8, 'elbow_tail_bone_alignment')  # left_elbow-tail_bone-right_elbow
    get_angle(9, 17, 10, 'wrist_alignment')       # left_wrist-neck-right_wrist
    get_angle(9, 18, 10, 'wrist_chest_alignment')  # left_wrist-chest-right_wrist
    get_angle(9, 19, 10, 'wrist_mid_spine_alignment')  # left_wrist-mid_spine-right_wrist
    get_angle(9, 20, 10, 'wrist_lower_spine_alignment')  # left_wrist-lower_spine-right_wrist
    get_angle(9, 21, 10, 'wrist_tail_bone_alignment')  # left_wrist-tail_bone-right_wrist
    get_angle(11, 18, 12, 'hip_alignment')       # left_hip-chest-right_hip
    get_angle(11, 19, 12, 'hip_mid_spine_alignment')  # left_hip-mid_spine-right_hip
    get_angle(11, 20, 12, 'hip_lower_spine_alignment')  # left_hip-lower_spine-right_hip
    get_angle(11, 21, 12, 'hip_tail_bone_alignment')  # left_hip-tail_bone-right_hip

    return angles

def angle_box(image, angles, position_offset=(10,10)): #image is the frame we draw the box on, angles are the previously made dicrionary, and position_offset is the offset for the text box
    x_offset, y_start, = position_offset # this makes it so the beginning of the text box is 10 pixels from the left and begins 10 pixels from the top 
    y_offset= 8# how much the text box moves down each time for each new line
    angle_categories = {  #Creates a dictionary of the categories of angles
         'Arm Angles': [ #lists the angles within each category using []
            'left_elbow', 'left_shoulder_elbow', 'left_armpit', 'left_armpit_wrist',
            'left_shoulder_wrist', 'left_wrist_ear', 'left_elbow_ear', 'left_elbow_right_shoulder',
            'left_wrist_right_shoulder', 'left_elbow_chest', 'left_wrist_chest', 'left_elbow_mid_spine',
            'left_wrist_mid_spine', 'left_elbow_lower_spine', 'left_wrist_lower_spine',
            'left_elbow_tail_bone', 'left_wrist_tail_bone',
            'right_elbow', 'right_shoulder_elbow', 'right_armpit', 'right_armpit_wrist',
            'right_shoulder_wrist', 'right_wrist_ear', 'right_elbow_ear', 'right_elbow_left_shoulder',
            'right_wrist_left_shoulder', 'right_elbow_chest', 'right_wrist_chest', 'right_elbow_mid_spine',
            'right_wrist_mid_spine', 'right_elbow_lower_spine', 'right_wrist_lower_spine',
            'right_elbow_tail_bone', 'right_wrist_tail_bone'
       ],                           
         
        'Leg Angles': [
            'left_knee', 'left_knee_toes', 'left_hip', 'left_hip_ankle', 'left_hip_toes', 'left_ankle',
            'right_knee', 'right_knee_toes', 'right_hip', 'right_hip_ankle', 'right_hip_toes', 'right_ankle'
        ],
        'Spine Angles': [
            'chest', 'mid_spine_chest', 'mid_spine', 'lower_spine', 'tail_bone',
            'spine_alignment', 'neck_alignment', 'neck_lower_spine', 'neck_tail_bone'
        ],
        'Body Alignment Angles': [
            'shoulder_alignment', 'shoulder_chest_alignment', 'shoulder_mid_spine_alignment',
            'shoulder_lower_spine_alignment', 'shoulder_tail_bone_alignment',
            'elbow_alignment', 'elbow_chest_alignment', 'elbow_mid_spine_alignment',
            'elbow_lower_spine_alignment', 'elbow_tail_bone_alignment',
            'wrist_alignment', 'wrist_chest_alignment', 'wrist_mid_spine_alignment',
            'wrist_lower_spine_alignment', 'wrist_tail_bone_alignment',
            'hip_alignment', 'hip_mid_spine_alignment', 'hip_lower_spine_alignment',
            'hip_tail_bone_alignment'
        ]
        }
    colors = {
        'Arm Angles': (233, 193, 133),  # blue 
        'Leg Angles': (63, 208, 244),  # Yellow 
        'Spine Angles': (60, 76, 231),  # Red
        'Body Alignment Angles': (137, 165, 23)  # Green
    }
    y_current = y_start # Start position for text box

    for category, angle_names in angle_categories.items():  #creates a repeating for function for each category, .item() makes it so it gets the name and the value of the item within the categories dictionary 
        cv2.putText(image, category, (x_offset, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.2, colors[category], 1) # uses cv2 to put text on the video/ images
        y_current += y_offset  # Move down for next category using the offset variable 

        for angle_name in angle_names:  # this is a for loop that goes through each angle name within the category
            if angle_name in angles:  # checks if the angle name is in the angles dictionary
                angle_value = angles[angle_name]  # gets the value of the angle name from the angles dictionary
                text = f"{angle_name.replace('_', ' ').title()}: {angle_value:.1f}°"  # Formats the text to be displayed
                cv2.putText(image, text, (x_offset + 20, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.2, colors[category], 1)
                y_current += y_offset  # Move down for next angle text
        y_current += 10  # Add extra space after each category


def select_best_person(keypoints_array, confidences_array, confidence_threshold=0.3):
    """
    Select the best person to track based on proximity to camera and visibility.
         
    Args:
        keypoints_array: Array of shape (num_people, num_keypoints, 2)
        confidences_array: Array of shape (num_people, num_keypoints)
        confidence_threshold: Minimum confidence for a keypoint to be considered visible
         
    Returns:
        tuple: (best_keypoints, best_confidences, best_person_idx) or (None, None, -1) if no good person found
    """
    if len(keypoints_array) == 0:
        return None, None, -1
         
    best_person_idx = -1
    best_score = -1
         
    for person_idx, (kpts, confs) in enumerate(zip(keypoints_array, confidences_array)):
        # Count visible keypoints
        visible_keypoints = np.sum(confs > confidence_threshold)
                 
        # Skip if too few visible keypoints (need at least 8 for basic tracking)
        if visible_keypoints < 8:
            continue
                 
        # Calculate person's "size" in the image (proxy for distance from camera)
        # Use all visible keypoints for full body bounding box
        visible_points = []
                 
        for idx, conf in enumerate(confs):
            if conf > confidence_threshold:
                visible_points.append(kpts[idx])
                 
        if len(visible_points) < 2:
            # Not enough visible points, skip this person
            continue
                 
        visible_points = np.array(visible_points)
                 
        # Calculate bounding box of all visible keypoints (full body)
        body_width = np.max(visible_points[:, 0]) - np.min(visible_points[:, 0])
        body_height = np.max(visible_points[:, 1]) - np.min(visible_points[:, 1])
        body_area = body_width * body_height
                 
        # Calculate composite score: visibility (70%) + size/proximity (30%)
        visibility_score = visible_keypoints / len(confs)  # Normalized to 0-1
        size_score = min(body_area / 100000, 1.0)  # Normalized, cap at 1.0 (increased threshold for full body)
                 
        composite_score = (0.7 * visibility_score) + (0.3 * size_score)
                 
        if composite_score > best_score:
            best_score = composite_score
            best_person_idx = person_idx
         
    if best_person_idx == -1:
        return None, None, -1
         
    return keypoints_array[best_person_idx], confidences_array[best_person_idx], best_person_idx


def draw_skeleton_on_image(image, keypoints, confidences, velocities=None, body_measurements=None, fitness_scorer=None, rpe_calculator=None, velocity_tracker=None, show_velocity_vectors=True, confidence_threshold=0.3):
    """
    Draw skeleton connections on image with steadiness detection support and RPE display
    ENHANCED: Properly handles ready position detection and tracking states
    
    CRITICAL CHANGES:
    - Checks if tracking has started before calling fitness scorer evaluation
    - Displays ready position status prominently
    - Only shows performance metrics when tracking is active
    """
    
    # Skeleton connections (your exact specifications)
    skeleton_connections = {
        'green': [  # Face and head connections
            (5, 3), (3, 1), (1, 0), (0, 2), (2, 4), (4, 6),
        ],
        'blue': [   # Arms and torso connections
            (9, 7), (7, 5), (5, 6), (6, 8), (8, 10), (5, 11), (6, 12),
        ],
        'red': [    # Spine connections
            (17, 18), (17, 19), (18, 19), (19, 20), (20, 21),
        ],
        'yellow': [ # Legs connections
            (11, 13), (12, 14), (13, 15), (14, 16), (15, 22), (16, 23), (11, 12),
        ]
    }
    
    # Colors (BGR format for OpenCV)
    colors = {
        'green': (137, 165, 23),
        'blue': (233, 193, 133),
        'red': (60, 76, 231),
        'yellow': (63, 208, 244)
    }
    
    # Map each keypoint to its color based on which connections it belongs to
    keypoint_colors = {}
    for color_name, connections in skeleton_connections.items():
        color = colors[color_name]
        for start_idx, end_idx in connections:
            keypoint_colors[start_idx] = color
            keypoint_colors[end_idx] = color
    
    h, w = image.shape[:2]
    
    # Ensure keypoints are in the right format
    if keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if confidences.shape[0] == 1:
        confidences = confidences[0]
    
    # Convert normalized coordinates to pixel coordinates if needed
    if keypoints.max() <= 1.0:
        keypoints = keypoints * [w, h]
    
    # Draw skeleton connections with thicker lines
    for color_name, connections in skeleton_connections.items():
        color = colors[color_name]
        
        for start_idx, end_idx in connections:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue
            
            if (confidences[start_idx] > confidence_threshold and 
                confidences[end_idx] > confidence_threshold):
                
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                
                cv2.line(image, start_point, end_point, color, 4)
    
    # Draw keypoints with matching colors
    for i, (keypoint, conf) in enumerate(zip(keypoints, confidences)):
        if conf > confidence_threshold:
            point = tuple(map(int, keypoint))
            keypoint_color = keypoint_colors.get(i, (0, 255, 255))
            
            cv2.circle(image, point, 6, keypoint_color, -1)
            cv2.putText(image, str(i), (point[0] + 8, point[1] - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Calculate angles for display
    angles = calculate_body_angles(keypoints, confidences, confidence_threshold)
    
    # CRITICAL SECTION: Handle ready position detection and fitness scoring
    if fitness_scorer is not None and angles:
        # STEP 1: Check ready position and coordinate tracking start
        ready_status_info = fitness_scorer.check_ready_and_start_tracking(
            angles, keypoints, confidences, fitness_scorer.frame_count + fitness_scorer.pre_ready_frames + 1, velocities
        )
        
        # STEP 2: Only evaluate fitness if tracking has started
        if ready_status_info['tracking_started']:
            # Get body scale for adaptive radius calculation
            body_scale = None
            if body_measurements and 'reference_scale' in body_measurements:
                body_scale = body_measurements['reference_scale']
            
            # Perform fitness evaluation (this will only run when tracking is active)
            evaluation = fitness_scorer.evaluate_frame(
                angles, 
                velocities or {}, 
                keypoints=keypoints,
                body_scale=body_scale,
                confidences=confidences
            )
            
            # Draw performance metrics (only when tracking)
            image = draw_score_display_with_tracking_status(image, fitness_scorer)
            image = draw_rep_counter_display(image, fitness_scorer, position=(900, 150))
            image = draw_steadiness_info_box(image, fitness_scorer, position=(900, 600))
            
            # Draw ready position status (will show "tracking active")
            if evaluation and 'rep_info' in evaluation:
                rep_info = evaluation['rep_info']
                image = draw_waiting_for_ready_display(image, rep_info, position=(200, 600))

            # RPE Calculator integration (only when tracking)
            if velocity_tracker is not None:
                rep_counter = fitness_scorer.rep_counter
                try:
                    image = draw_integrated_rpe_display(image, velocity_tracker, rep_counter, position=(900, 300))
                except Exception as e:
                    cv2.putText(image, f"RPE: Calculating... ({str(e)[:20]})", (900, 320), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(image, "RPE: Needs VelocityTracker", (900, 320), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw feedback messages (only when tracking)
            if evaluation['feedback']:
                image = draw_feedback_messages(image, evaluation['feedback'], position=(900, 500))
            
            # Draw steadiness indicators for monitored joints (only when tracking)
            image = draw_steadiness_indicators(image, fitness_scorer, keypoints, confidences, body_scale, confidence_threshold)
            
        else:
            # TRACKING NOT STARTED - Display ready position guidance
            image = draw_waiting_for_ready_display(image, ready_status_info, position=(200, 600))
            
            # Show basic score display (grayed out)
            image = draw_score_display_with_tracking_status(image, fitness_scorer)
            
            # Show empty rep counter
            image = draw_rep_counter_display(image, fitness_scorer, position=(900, 150))
    
    # Draw angle measurements on the image
    if angles:
        # Create semi-transparent overlay for angle text
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (175, min(len(angles) * 25 + 100, h)), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        angle_box(image, angles)
    
    # ENHANCED: Handle velocity display based on tracking state
    if velocities:
        # Check if velocities contain tracking status information
        tracking_started = velocities.get('_tracking_started', True)  # Default to True for compatibility
        
        if tracking_started:
            # Normal velocity display when tracking
            overlay = image.copy()
            cv2.rectangle(overlay, (180, 0), (420, 400), (0, 0, 0), -1)
            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            velocity_info_box(image, velocities, body_measurements)
        else:
            # Show waiting message instead of velocity data
            preparation_frames = velocities.get('_preparation_frames', 0)
            overlay = image.copy()
            cv2.rectangle(overlay, (180, 0), (420, 100), (0, 0, 0), -1)
            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            cv2.putText(image, "Velocity Analysis", (190, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, "Waiting for ready position...", (190, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(image, f"Preparation frames: {preparation_frames}", (190, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return image, angles




def draw_steadiness_indicators(image, fitness_scorer, keypoints, confidences, body_scale=None, confidence_threshold=0.3):
    """
    Draw visual indicators for joints being monitored for steadiness
    
    Args:
        image: OpenCV image
        fitness_scorer: FitnessScorer instance with steadiness rules
        keypoints: Array of keypoint coordinates
        confidences: Array of confidence scores
        body_scale: Current body scale for adaptive radius calculation
        confidence_threshold: Minimum confidence for drawing indicators
    
    Returns:
        image: Image with steadiness indicators drawn
    """
    # Keypoint name mapping
    keypoint_mapping = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
        'neck': 17, 'chest': 18, 'mid_spine': 19, 'lower_spine': 20, 'tail_bone': 21,
        'left_toes': 22, 'right_toes': 23
    }
    
    # Get steadiness requirements
    steadiness_rules = fitness_scorer.get_joint_steadiness_requirements()
    
    for joint_name, rule in steadiness_rules.items():
        # Skip if joint not found in mapping
        if joint_name not in keypoint_mapping:
            continue
            
        joint_idx = keypoint_mapping[joint_name]
        if joint_idx >= len(keypoints) or joint_idx >= len(confidences):
            continue
            
        # Skip if confidence too low
        if confidences[joint_idx] < confidence_threshold:
            continue
        
        # Get current joint position
        current_position = keypoints[joint_idx]
        current_point = tuple(map(int, current_position))
        
        # Get center position if available
        if joint_name in fitness_scorer.steadiness_centers:
            center = fitness_scorer.steadiness_centers[joint_name]
            if center is not None:
                center_point = tuple(map(int, center))
                
                # Calculate effective radius
                if rule["adaptive_radius"] and body_scale is not None:
                    effective_radius = int(rule["radius"] * body_scale)
                else:
                    effective_radius = int(rule["radius"])
                
                # Calculate current distance from center
                distance = np.linalg.norm(current_position - center)
                
                # Determine colors based on violation status
                if distance <= effective_radius:
                    # Within bounds - green circle
                    circle_color = (0, 255, 0)  # Green
                    center_color = (0, 200, 0)  # Darker green
                    text_color = (0, 255, 0)
                else:
                    # Outside bounds - red circle
                    circle_color = (0, 0, 255)  # Red
                    center_color = (0, 0, 200)  # Darker red
                    text_color = (0, 0, 255)
                
                # Draw steadiness circle (allowed radius)
                cv2.circle(image, center_point, effective_radius, circle_color, 2)
                
                # Draw center point
                cv2.circle(image, center_point, 4, center_color, -1)
                
                # Draw line from center to current position
                cv2.line(image, center_point, current_point, circle_color, 1)
                
                # Draw joint label and distance info
                label_pos = (current_point[0] + 10, current_point[1] - 10)
                joint_display_name = joint_name.replace('_', ' ').title()
                distance_text = f"{joint_display_name}: {distance:.1f}/{effective_radius}"
                
                cv2.putText(image, distance_text, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                
                # Draw violation count if there are current violations
                if joint_name in fitness_scorer.steadiness_violations:
                    violations = fitness_scorer.steadiness_violations[joint_name]
                    if violations > 0:
                        violation_pos = (current_point[0] + 10, current_point[1] + 10)
                        violation_text = f"Violations: {violations}"
                        cv2.putText(image, violation_text, violation_pos, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    return image


def draw_steadiness_info_box(image, fitness_scorer, position=(450, 10)):
    """
    Draw information box showing steadiness monitoring status
    
    Args:
        image: OpenCV image
        fitness_scorer: FitnessScorer instance
        position: (x, y) position for info box
    
    Returns:
        image: Image with steadiness info box
    """
    x_offset, y_start = position
    y_offset = 15
    y_current = y_start
    
    # Get steadiness debug info
    debug_info = fitness_scorer.get_steadiness_debug_info()
    
    if not debug_info:
        return image
    
    # Create semi-transparent background
    box_height = len(debug_info) * y_offset + 40
    overlay = image.copy()
    cv2.rectangle(overlay, (x_offset-5, y_start-20), (x_offset+300, y_start + box_height), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Title
    cv2.putText(image, "Steadiness Monitor:", (x_offset, y_current), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_current += y_offset + 5
    
    # Joint information
    for joint_name, info in debug_info.items():
        joint_display = joint_name.replace('_', ' ').title()
        
        # Status color based on violations
        if info['violation_count'] > 0:
            status_color = (0, 0, 255)  # Red
            status = f"VIOLATING ({info['violation_count']})"
        elif info['center_position'] is not None:
            status_color = (0, 255, 0)  # Green
            status = "MONITORING"
        else:
            status_color = (255, 255, 0)  # Yellow
            status = "CALIBRATING"
        
        # Joint status line
        status_text = f"{joint_display}: {status}"
        cv2.putText(image, status_text, (x_offset, y_current), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
        y_current += y_offset
        
        # Rule details (smaller text)
        rule = info['rule']
        radius_type = "adaptive" if rule['adaptive_radius'] else "fixed"
        rule_text = f"  R:{rule['radius']} ({radius_type}) P:{rule['deduction']}"
        cv2.putText(image, rule_text, (x_offset + 10, y_current), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        y_current += y_offset - 3
    
    return image

def draw_rep_counter_display(image, fitness_scorer, position=(900, 200)):
    """
    Draw rep counter information on the image
    
    Args:
        image: OpenCV image
        fitness_scorer: FitnessScorer instance with rep_counter
        position: (x, y) position for display
    """
    rep_info = fitness_scorer.get_rep_info()
    x, y = position
    
    # Create semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x-5, y-25), (x+300, y+120), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
    
    # Title
    cv2.putText(image, "REP COUNTER", (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Rep count (large text)
    rep_text = f"Reps: {rep_info['total_reps']}"
    cv2.putText(image, rep_text, (x, y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Half reps
    half_rep_text = f"Half Reps: {rep_info['half_reps']}"
    cv2.putText(image, half_rep_text, (x, y + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Current phase with color coding
    phase = rep_info['current_phase']
    phase_colors = {
        'neutral': (255, 255, 255),      # White
        'eccentric': (0, 165, 255),      # Orange
        'concentric': (255, 0, 255)      # Magenta
    }
    phase_color = phase_colors.get(phase, (128, 128, 128))
    
    phase_text = f"Phase: {phase.title()}"
    cv2.putText(image, phase_text, (x, y + 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
    
    return image



def test_on_image():
    """Test the model on a single image"""
    print("Testing on image...")
    
    # Load model
    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        return
    
    model = YOLO(MODEL_PATH)
    
    
    # Create output directory
    output_dir = OUTPUT_DIR / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on image
    if Path(TEST_IMAGE).exists():
        image = cv2.imread(TEST_IMAGE)
        results = model(image)
        
        all_angles= []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                print(f"Detected {len(keypoints)} person(s)")
                
                # Draw skeleton for each person
                for i, (kpts, confs) in enumerate(zip(keypoints, confidences)):
                    visible_kpts = np.sum(confs > 0.3)
                    print(f"Person {i+1}: Visible keypoints: {visible_kpts}/24")
                    
                    image, angles = draw_skeleton_on_image(image, kpts, confs)
                    all_angles.append(angles)
        
        # Save result with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"result_image_{timestamp}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"Result saved as: {output_path}")
        
        cv2.imshow("AI Fitness Trainer - Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Test image not found: {TEST_IMAGE}")

def countdown_with_preview(cap, countdown_seconds=5, show_preview=True):
    """
    Display a countdown with camera preview before starting form recording
    
    Args:
        cap: OpenCV VideoCapture object
        countdown_seconds: Number of seconds to countdown (default: 5)
        show_preview: Whether to show camera preview during countdown
    
    Returns:
        bool: True if countdown completed, False if user pressed 'q' to quit
    """
    print(f"Starting {countdown_seconds} second countdown...")
    print("Position yourself in front of the camera")
    print("Press 'q' to quit, or wait for countdown to complete")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            return False
        
        # Calculate remaining time
        elapsed = time.time() - start_time
        remaining = max(0, countdown_seconds - elapsed)
        
        if remaining <= 0:
            # Countdown finished
            if show_preview:
                # Show "GO!" message for 1 second
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
                frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
                
                cv2.putText(frame, "GO!", 
                           (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
                
                cv2.imshow("AI Fitness Trainer - Countdown", frame)
                cv2.waitKey(1000)  # Show "GO!" for 1 second
            
            print("Countdown complete! Starting form analysis...")
            return True
        
        if show_preview:
            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
            # Display countdown number
            countdown_text = str(int(remaining) + 1)
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 8, 12)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 255), 12)
            
            # Display instructions
            instructions = [
                "Position yourself for exercise",
                "Make sure you're fully visible", 
                f"Starting in {int(remaining) + 1}...",
                "Press 'q' to cancel"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = 50 + (i * 40)
                cv2.putText(frame, instruction, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("AI Fitness Trainer - Countdown", frame)
        
        # Check for quit key
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Countdown cancelled by user")
            return False

def test_on_webcam():
    """
    Test the model on webcam with countdown, adaptive velocity tracking and fitness scoring
    ENHANCED: Proper ready position detection integration - SINGLE PERSON TRACKING
    
    CRITICAL CHANGES:
    1. VelocityTracker and FitnessScorer coordinate through ready position detection
    2. No tracking starts until user assumes proper ready position
    3. Clear visual feedback about ready position status
    4. Steadiness centers are set from ready position, not random movement
    """
    print("Testing on webcam with READY POSITION DETECTION and fitness scoring...")
    print("The system will wait for you to assume ready position before starting tracking")
    
    # Load model
    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    
    # CRITICAL: Initialize velocity tracker (includes ready position detector and fitness scorer)
    velocity_tracker = VelocityTracker(smoothing_window=5)
    
    # Create output directory
    output_dir = OUTPUT_DIR / "webcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Set webcam resolution
    WEBCAM_WIDTH = 1920
    WEBCAM_HEIGHT = 1080
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Resolution: {actual_width}x{actual_height} at {actual_fps}fps")
    
    # Add countdown
    countdown_completed = countdown_with_preview(cap, countdown_seconds=5, show_preview=True)
    
    if not countdown_completed:
        print("Countdown cancelled. Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("READY POSITION DETECTION PHASE")
    print("="*50)
    print("🎯 Move to ready position: arms extended down at sides")
    print("📊 Tracking will start automatically when ready position is detected")
    print("⚡ Stay steady in ready position until tracking begins")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reset")
    print("Press 'c' to restart countdown")
    
    saved_count = 0
    person_tracking_stats = {
        'total_frames': 0, 
        'person_detected': 0, 
        'multiple_people_detected': 0,
        'frames_before_ready': 0,
        'frames_after_ready': 0,
        'ready_detection_frame': None
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        person_tracking_stats['total_frames'] += 1
        
        # Run inference
        results = model(frame)
        
        # Process results with single person selection
        person_tracked = False
        tracking_started = False
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                num_people_detected = len(keypoints)
                
                if num_people_detected > 0:
                    person_tracking_stats['person_detected'] += 1
                    
                    if num_people_detected > 1:
                        person_tracking_stats['multiple_people_detected'] += 1
                
                # SELECT BEST PERSON TO TRACK
                best_kpts, best_confs, best_person_idx = select_best_person(
                    keypoints, confidences, confidence_threshold=0.3
                )
                
                if best_kpts is not None:
                    person_tracked = True
                    
                    # CRITICAL STEP: Calculate velocities with ready position integration
                    # This handles the ready position detection internally
                    velocities = velocity_tracker.calculate_velocities(
                        best_kpts, current_time, best_confs
                    )
                    
                    # Check if tracking has started (from velocity tracker status)
                    tracking_started = velocities.get('_tracking_started', False)
                    
                    # Update statistics based on tracking state
                    if tracking_started:
                        person_tracking_stats['frames_after_ready'] += 1
                        if person_tracking_stats['ready_detection_frame'] is None:
                            person_tracking_stats['ready_detection_frame'] = person_tracking_stats['total_frames']
                    else:
                        person_tracking_stats['frames_before_ready'] += 1
                    
                    # Get body measurements (always calculated)
                    body_measurements = velocity_tracker.body_calculator.update_body_proportions(
                        best_kpts, best_confs
                    )
                    
                    # CRITICAL: Get RPE analysis (handles ready position internally)
                    rpe_result = velocity_tracker.get_integrated_rpe_analysis(
                        rep_counter=velocity_tracker.fitness_scorer.rep_counter
                    )
                    
                    # CRITICAL: Draw skeleton with ready position integration
                    # This function now handles ready position detection and displays
                    frame, angles = draw_skeleton_on_image(
                        frame, best_kpts, best_confs, velocities, body_measurements, 
                        velocity_tracker.fitness_scorer, 
                        rpe_calculator=velocity_tracker.rpe_calculator,
                        velocity_tracker=velocity_tracker
                    )
                    
                    # Add tracking info to the display
                    tracking_color = (0, 255, 0) if tracking_started else (255, 255, 0)
                    tracking_status = "TRACKING ACTIVE" if tracking_started else "WAITING FOR READY"
                    
                    tracking_info = f"Person {best_person_idx + 1}/{num_people_detected} - {tracking_status}"
                    cv2.putText(frame, tracking_info, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
                    
                    if num_people_detected > 1:
                        multi_person_warning = f"⚠️ {num_people_detected} people detected - tracking best one"
                        cv2.putText(frame, multi_person_warning, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Show preparation progress
                    if not tracking_started:
                        prep_frames = velocities.get('_preparation_frames', 0)
                        prep_text = f"Preparation frames: {prep_frames} (Move to ready position)"
                        cv2.putText(frame, prep_text, (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add status info when no person is being tracked
        if not person_tracked:
            no_person_text = "No suitable person detected - move closer or improve lighting"
            cv2.putText(frame, no_person_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show frame
        window_title = "AI Fitness Trainer - Ready Position Detection"
        if tracking_started:
            window_title = "AI Fitness Trainer - Fitness Tracking Active"
        
        cv2.imshow(window_title, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_tag = "tracking" if tracking_started else "ready_detection"
            output_path = output_dir / f"frame_{status_tag}_{timestamp}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
            print(f"Frame saved as: {output_path}")
        elif key == ord('r'):
            # Reset everything - user must get ready again
            velocity_tracker.reset_tracking()
            person_tracking_stats['frames_before_ready'] = 0
            person_tracking_stats['frames_after_ready'] = 0
            person_tracking_stats['ready_detection_frame'] = None
            print("🔄 System reset - assume ready position to restart tracking")
        elif key == ord('c'):
            # Restart countdown
            print("Restarting countdown...")
            cv2.destroyAllWindows()
            countdown_completed = countdown_with_preview(cap, countdown_seconds=5, show_preview=True)
            if not countdown_completed:
                print("Countdown cancelled. Exiting...")
                break
            cv2.destroyAllWindows()
            print("Countdown complete! Resuming ready position detection...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print detailed session summary
    print("\n" + "="*50)
    print("READY POSITION DETECTION SESSION SUMMARY")
    print("="*50)
    print(f"Frames saved: {saved_count}")
    print(f"Total frames processed: {person_tracking_stats['total_frames']}")
    print(f"Frames with person detected: {person_tracking_stats['person_detected']} ({person_tracking_stats['person_detected']/person_tracking_stats['total_frames']*100:.1f}%)")
    print(f"Frames with multiple people: {person_tracking_stats['multiple_people_detected']} ({person_tracking_stats['multiple_people_detected']/person_tracking_stats['total_frames']*100:.1f}%)")
    print(f"Frames before ready position: {person_tracking_stats['frames_before_ready']}")
    print(f"Frames after ready position: {person_tracking_stats['frames_after_ready']}")
    
    if person_tracking_stats['ready_detection_frame']:
        print(f"Ready position detected at frame: {person_tracking_stats['ready_detection_frame']}")
        detection_time = person_tracking_stats['ready_detection_frame'] / actual_fps
        print(f"Time to ready detection: {detection_time:.1f} seconds")
    else:
        print("Ready position was never detected during session")
    
    # Get final tracking status
    tracking_status = velocity_tracker.get_tracking_status()
    print(f"\nFINAL TRACKING STATUS:")
    print(f"  Tracking started: {tracking_status['tracking_started']}")
    print(f"  Preparation frames: {tracking_status['preparation_frames']}")
    print(f"  Tracking frames: {tracking_status['tracking_frames']}")
    print(f"  Velocity history length: {tracking_status['velocity_history_length']}")
    
    # Get fitness summary (only meaningful if tracking started)
    if tracking_status['tracking_started']:
        print(velocity_tracker.fitness_scorer.get_summary_report())
    else:
        print("\nNo fitness evaluation performed - ready position was not detected")


def test_on_video():
    """
    Test the model on a video with ready position detection integration
    ENHANCED: Proper ready position detection for video processing
    """
    print("Testing on video with READY POSITION DETECTION and fitness scoring...")
    
    # Load model
    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    
    if not Path(TEST_VIDEO).exists():
        print(f"Test video not found: {TEST_VIDEO}")
        return
    
    # Initialize velocity tracker with ready position detection
    velocity_tracker = VelocityTracker(smoothing_window=3)
    
    # Create output directory
    output_dir = OUTPUT_DIR / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(TEST_VIDEO)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    is_vertical = height > width
    print(f"Video resolution: {width}x{height} at {fps}fps")
    print(f"Orientation: {'Vertical/Portrait' if is_vertical else 'Horizontal/Landscape'}")
    print("🎯 Will detect ready position before starting tracking")
    
    # Setup video writer
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orientation_tag = "vertical" if is_vertical else "horizontal"
    output_path = output_dir / f"ready_detection_video_{orientation_tag}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Tracking statistics
    video_stats = {
        'total_frames': 0, 
        'person_detected': 0, 
        'multiple_people_detected': 0,
        'frames_before_ready': 0,
        'frames_after_ready': 0,
        'ready_detection_frame': None,
        'person_switches': 0,
        'last_tracked_person': -1
    }
    
    frame_count = 0
    paused = False
    
    print(f"\n{'='*50}")
    print("PROCESSING VIDEO WITH READY POSITION DETECTION")
    print(f"{'='*50}")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            video_stats['total_frames'] += 1
        
            # Run inference
            results = model(frame)
            
            # Process results with single person selection
            person_tracked = False
            tracking_started = False
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy()
                    
                    num_people_detected = len(keypoints)
                    
                    if num_people_detected > 0:
                        video_stats['person_detected'] += 1
                        
                        if num_people_detected > 1:
                            video_stats['multiple_people_detected'] += 1
                    
                    # SELECT BEST PERSON TO TRACK
                    best_kpts, best_confs, best_person_idx = select_best_person(
                        keypoints, confidences, confidence_threshold=0.3
                    )
                    
                    if best_kpts is not None:
                        person_tracked = True
                        
                        # Track person switches
                        if (video_stats['last_tracked_person'] != -1 and 
                            video_stats['last_tracked_person'] != best_person_idx):
                            video_stats['person_switches'] += 1
                        video_stats['last_tracked_person'] = best_person_idx
                        
                        # Calculate velocities with ready position detection
                        velocities = velocity_tracker.calculate_velocities(
                            best_kpts, current_time, best_confs
                        )
                        
                        # Check tracking status
                        tracking_started = velocities.get('_tracking_started', False)
                        
                        # Update statistics
                        if tracking_started:
                            video_stats['frames_after_ready'] += 1
                            if video_stats['ready_detection_frame'] is None:
                                video_stats['ready_detection_frame'] = frame_count
                        else:
                            video_stats['frames_before_ready'] += 1
                        
                        # Get body measurements and RPE
                        body_measurements = velocity_tracker.body_calculator.update_body_proportions(
                            best_kpts, best_confs
                        )
                        
                        rpe_result = velocity_tracker.get_integrated_rpe_analysis(
                            rep_counter=velocity_tracker.fitness_scorer.rep_counter
                        )
                        
                        # Draw skeleton with ready position integration
                        frame, angles = draw_skeleton_on_image(
                            frame, best_kpts, best_confs, velocities, body_measurements, 
                            velocity_tracker.fitness_scorer, 
                            rpe_calculator=velocity_tracker.rpe_calculator,
                            velocity_tracker=velocity_tracker
                        )
                        
                        # Add tracking info overlay
                        tracking_color = (0, 255, 0) if tracking_started else (255, 255, 0)
                        tracking_status_text = "TRACKING" if tracking_started else "READY DETECTION"
                        
                        tracking_info = f"Person {best_person_idx + 1}/{num_people_detected} - {tracking_status_text}"
                        cv2.putText(frame, tracking_info, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
                        
                        if num_people_detected > 1:
                            multi_warning = f"Multiple people detected - auto-selecting best"
                            cv2.putText(frame, multi_warning, (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add status when no person tracked
            if not person_tracked:
                no_person_text = "No suitable person detected"
                cv2.putText(frame, no_person_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Write frame to output
            out.write(frame)
        
        # Display processing
        display_frame = frame
        if is_vertical and width > 600:
            display_width = 600
            display_height = int(display_width * (height / width))
            display_frame = cv2.resize(frame, (display_width, display_height))
        
        # Add frame info with tracking status
        tracking_status_text = "TRACKING ACTIVE" if tracking_started else "READY DETECTION"
        frame_info = f"Frame: {frame_count} | {tracking_status_text} | Person: {person_tracked}"
        cv2.putText(display_frame, frame_info, (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if paused:
            pause_text = "PAUSED - Press 'p' to resume"
            cv2.putText(display_frame, pause_text, (10, display_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("AI Fitness Trainer - Video Analysis with Ready Detection", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Processing stopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} at frame {frame_count}")
        elif key == ord('r'):
            # Reset tracking state
            velocity_tracker.reset_tracking()
            video_stats['frames_before_ready'] = 0
            video_stats['frames_after_ready'] = 0
            video_stats['ready_detection_frame'] = None
            print(f"🔄 Tracking reset at frame {frame_count} - ready detection will restart")
        
        # Progress update
        if not paused and frame_count % 30 == 0:
            progress_percent = (cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100
            ready_status = "Tracking" if tracking_started else "Waiting for ready"
            print(f"Progress: {progress_percent:.1f}% | Frame {frame_count} | Status: {ready_status}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*50}")
    print("VIDEO ANALYSIS WITH READY DETECTION COMPLETE")
    print(f"{'='*50}")
    print(f"✅ Video saved as: {output_path}")
    print(f"📊 VIDEO PROCESSING STATISTICS:")
    print(f"   • Total frames: {video_stats['total_frames']}")
    print(f"   • Frames with person: {video_stats['person_detected']} ({video_stats['person_detected']/video_stats['total_frames']*100:.1f}%)")
    print(f"   • Frames with multiple people: {video_stats['multiple_people_detected']} ({video_stats['multiple_people_detected']/video_stats['total_frames']*100:.1f}%)")
    print(f"   • Person switches: {video_stats['person_switches']}")
    print(f"   • Frames before ready: {video_stats['frames_before_ready']}")
    print(f"   • Frames after ready: {video_stats['frames_after_ready']}")
    
    if video_stats['ready_detection_frame']:
        print(f"   • Ready position detected at frame: {video_stats['ready_detection_frame']}")
        detection_time = video_stats['ready_detection_frame'] / fps
        print(f"   • Time to ready detection: {detection_time:.1f} seconds")
    else:
        print("   • Ready position was never detected during video")
    
    # Get final tracking status

    print(f"\n📈 FINAL TRACKING STATUS:")
    print(f"   • Tracking started: {tracking_status['tracking_started']}")
    print(f"   • Preparation frames: {tracking_status['preparation_frames']}")
    print(f"   • Tracking frames: {tracking_status['tracking_frames']}")
    print(f"   • Velocity history length: {tracking_status['velocity_history_length']}")
    
    # Get fitness summary (only if tracking started)
    if tracking_status['tracking_started']:
        print(velocity_tracker.fitness_scorer.get_summary_report())
    else:
        print("\n⚠️ No fitness evaluation performed - ready position was not detected in video")



if __name__ == "__main__":
    print("AI Fitness Trainer - Quick Test")
    print("=" * 40)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script to point to your trained model")
        print("Example: runs/hybrid_fitness_trainer/coco_custom_pose_v3/weights/best.pt")
        exit(1)
    
    # Create main runs directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nChoose test mode:")
    print("1. Test on image")
    print("2. Test on webcam (press 's' to save frames)") 
    print("3. Test on video")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == '1':
        test_on_image()
    elif choice == '2':
        test_on_webcam()
    elif choice == '3':
        test_on_video()
    else:
        print("Invalid choice. Testing on webcam...")
        test_on_webcam()