#!/usr/bin/env python3
"""
RLALLMA Evaluation Framework

Compares different approaches for drone swarm control:
1. Regular RL (only for movement task)
2. Pure LLM
3. RLALLMA (RL+LLM hybrid)
4. RLALLMA + SHAP augmented
5. RLALLMA + DiCE

Metrics:
- Reward
- Success Rate
- Crash percentage (at least 1 drone)
- Time spent at high altitude (>= 2.5m)
- Time spent at low altitude (<= 0.5m)
- Circle task accuracy (for behavioral task)

For various multi-drone tasks with 4 drones.
"""
import os
import time
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
import json
import threading # Not strictly used in this version, but kept from original
import queue     # Not strictly used in this version, but kept from original
from collections import deque
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import warnings

# Suppress SHAP and DiCE import warnings if they're not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. RLALLMA+SHAP will fall back to regular RLALLMA.")

try:
    import dice_ml
    DICE_AVAILABLE = True # You can set this to False if you don't intend to use DiCE
except ImportError:
    DICE_AVAILABLE = False
    warnings.warn("DiCE not available. RLALLMA+DiCE will fall back to regular RLALLMA.")

# Import and set up Gym-PyBullet-Drones environment
from stable_baselines3 import PPO, TD3, SAC, DDPG
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics
import pybullet as p

# Configure Gemini API
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or GEMINI_API_KEY is None:
        print("[WARNING] Gemini API key not set. LLM functionality will be disabled.")
        gemini_available = False
        gemini_model = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',#'gemini-1.5-flash-latest', gemini-2.0-flash
            generation_config={"response_mime_type": "application/json"}
        )
        gemini_available = True
        print("[INFO] Gemini API configured successfully.")
except ImportError:
    print("[WARNING] Google Generative AI SDK not installed. LLM functionality will be disabled.")
    gemini_available = False
    gemini_model = None
except Exception as e:
    print(f"[WARNING] Error initializing Gemini API: {e}. LLM functionality will be disabled.")
    gemini_available = False
    gemini_model = None

# --- Global Settings ---
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'rlallma_evaluation_results'
DEFAULT_OBS = ObservationType.KIN 
DEFAULT_ACT = ActionType.RPM
DEFAULT_NUM_DRONES = 4
DEFAULT_DURATION_SEC = 16.0 
DEFAULT_CTRL_FREQ = 20 
LLM_UPDATE_INTERVAL_STEPS = 20#DEFAULT_CTRL_FREQ * 1 
NUM_EPISODES = 3 
HISTORY_MAX_ENTRIES = 10 
drone_state_history = deque(maxlen=HISTORY_MAX_ENTRIES)

# --- Task Definitions with Custom Initial Positions and Target Generation Radius ---
DEFAULT_EPISODE_INITIAL_XYZS = np.array([ 
    [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0],
    [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0]
], dtype=np.float32)

# --- Circle Task Variables ---
# For tracking drone positions history
drone_positions_history = []  # List of lists, each containing positions of all drones at a step
follower_target_history = []  # To track what positions each follower should be targeting

# --- Altitude Tracking Variables ---
time_at_high_altitude = {}  # Dictionary to track time spent at high altitude (>=2.5m)
time_at_low_altitude = {}   # Dictionary to track time spent at low altitude (<=0.5m)

TASKS = {
    # "basic_movement": {
    #     "name": "Basic Random Movement",
    #     "description": "Move all drones from their start to randomly generated target positions. Each drone must reach its target at least once.",
    #     "command": "Move each drone to its assigned random target position.",
    #     "custom_initial_xyzs": DEFAULT_EPISODE_INITIAL_XYZS.copy(),
    #     "target_generation_radius": 1, # Max radius from start for random targets
    #     "check_success": lambda env, initial_mission_targets, drone_reached_targets_status: all(drone_reached_targets_status)
    # },
    # "spread_movement": {
    #     "name": "Spread Random Movement (keep >= 0.8m dist)",
    #     "description": "Move drones to random targets while maintaining >= 0.8m distance. Each must reach its target. Prioritize spreading if too close.",
    #     "command": "Move each drone to its random target, maintaining >= 0.8m inter-drone distance. If too close, prioritize spreading.",
    #     "custom_initial_xyzs": np.array([ 
    #         [1.5, 1.5, 1.0], [-1.5, 1.5, 1.0],
    #         [1.5, -1.5, 1.0], [-1.5, -1.5, 1.0] # Start spread out
    #     ], dtype=np.float32),
    #     "target_generation_radius": 1.0, 
    #     "check_success": lambda env, initial_mission_targets, drone_reached_targets_status: (
    #         all(drone_reached_targets_status) and
    #         all(np.linalg.norm(env.pos[i] - env.pos[j]) >= 0.75 
    #             for i in range(DEFAULT_NUM_DRONES) for j in range(i + 1, DEFAULT_NUM_DRONES))
    #     )
    # },
    # "tight_movement": {
    #     "name": "Tight Random Movement (keep <= 0.5m dist)",
    #     "description": "Move as a tight group to random targets, keeping all drones <= 0.5m of each other. Each must reach its target. Prioritize tightening if too spread.",
    #     "command": "Move as a tight formation (<= 0.5m inter-drone distance) to random targets. If too spread, prioritize tightening formation.",
    #     "custom_initial_xyzs": np.array([ 
    #         [0.1, 0.1, 1.0], [-0.1, 0.1, 1.0],
    #         [0.1, -0.1, 1.0], [-0.1, -0.1, 1.0] # Start very tight
    #     ], dtype=np.float32),
    #     "target_generation_radius": 0.25, # Smaller radius for tight movement tasks
    #     "check_success": lambda env, initial_mission_targets, drone_reached_targets_status: (
    #         all(drone_reached_targets_status) and
    #         all(np.linalg.norm(env.pos[i] - env.pos[j]) <= 0.55 
    #             for i in range(DEFAULT_NUM_DRONES) for j in range(i + 1, DEFAULT_NUM_DRONES))
    #     )
    # },
    # New Task 1: High Path Movement
    "high_path_movement": {
        "name": "High Path Movement (via 2.0m height)",
        "description": "Drones must first rise to 2.0m height (if history doesn't contain records with height >= 1.5), then fly horizontally to target X,Y coordinates, and finally descend to the target Z coordinate.",
        "command": "Move each drone to its target by first rising to 1.5m height (if height >= 1.4 you can move to next stage), then flying horizontally to the target X,Y position, and when diff to target X and Y is <= 0.2, set target to the final one.",
        "custom_initial_xyzs": DEFAULT_EPISODE_INITIAL_XYZS.copy(),
        "target_generation_radius": 0.5,
        "high_altitude_threshold": 1.35,  # Threshold to determine if drone has reached high altitude
        "check_success": lambda env, initial_mission_targets, drone_reached_targets_status: (
            all(drone_reached_targets_status) and
            all([getattr(env, '_drone_reached_high_altitude', {}).get(i, False) for i in range(DEFAULT_NUM_DRONES)])
        ),
        "track_path": True  # Flag to enable path tracking for this task
    },
    # New Task 2: Low Path Movement
    "low_path_movement": {
        "name": "Low Path Movement (via 0.25m height)",
        "description": "Drones must first descend to 0.25m height, then fly horizontally to target X,Y coordinates, and finally rise to the target Z coordinate.",
        "command": "Move each drone to its target by first descending to 0.5m height (if height <= 0.6 you can move to next stage), then flying horizontally to the target X,Y position, and when diff to target X and Y is <= 0.2, set target to the final target.",
        "custom_initial_xyzs": np.array([ 
            [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0],
            [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0]
        ], dtype=np.float32),  # Start higher to allow for descending
        "target_generation_radius": 0.5,
        "low_altitude_threshold": 0.6,  # Threshold to determine if drone has reached low altitude
        "check_success": lambda env, initial_mission_targets, drone_reached_targets_status: (
            all(drone_reached_targets_status) and
            all([getattr(env, '_drone_reached_low_altitude', {}).get(i, False) for i in range(DEFAULT_NUM_DRONES)])
        ),
        "track_path": True  # Flag to enable path tracking for this task
    },
    # New Task 3: Modified Random Leader with Followers
    "circle_behavior": {
        "name": "Random Leader with Followers",
        "description": "Lead drone (Drone 0) flies randomly. Other drones follow by targeting the previous drone's last position.",
        "command": "Drone 0: flies randomly, moving 0.2 each step in some direction, drone 1 to assume exactly last position of drone 0, drone 2 - position of 1, drone 3 - position of 2 and so on",
        "custom_initial_xyzs": np.array([
            [0.0, 0.0, 1.0],  # Drone 0 (leader) 
            [0.2, 0.0, 1.0],  # Drone 1
            [0.4, 0.0, 1.0],  # Drone 2
            [0.6, 0.0, 1.0]   # Drone 3
        ], dtype=np.float32),
        "target_generation_radius": 0.0,  # Not used for this task as targets are dynamically generated
        "check_success": lambda env, initial_mission_targets, drone_reached_targets_status: (
            # Success is based on follower accuracy - each follower should be close to the previous drone's last position
            all([getattr(env, '_follower_accuracy', {}).get(i, 0) > 0.7 for i in range(1, DEFAULT_NUM_DRONES)])
        ),
        "is_behavioral": True  # Flag to identify this as a behavioral task
    }
}

# Expand metrics to include new ones
APPROACHES = ["RLALLMA"]#["RL", "LLM", "RLALLMA", "RLALLMA+SHAP"]
# Initialize results dictionary with additional metrics
results = { 
    task_id: { 
        approach: {
            "reward": [], 
            "success_rate": [], 
            "crash_percentage": [],
            "time_high_altitude_pct": [], # Percentage of time spent at high altitude (>=2.5m)
            "time_low_altitude_pct": [],  # Percentage of time spent at low altitude (<=0.5m)
            "follower_accuracy": []       # For follower behavior task
        } for approach in APPROACHES 
    } for task_id in TASKS 
}

# --- Helper Function for Random Target Generation ---
def _generate_random_mission_targets(start_positions_for_episode, generation_radius, num_drones, arena_bounds=None):
    targets = np.zeros((num_drones, 3), dtype=np.float32)
    min_target_height = 0.3 
    max_target_height = 3.0 # Increased max height slightly

    if arena_bounds is None:
        arena_bounds = {'x': (-4.0, 4.0), 'y': (-4.0, 4.0), 'z': (min_target_height, max_target_height)}

    for i in range(num_drones):
        drone_start_pos = start_positions_for_episode[i]
        attempts = 0
        while attempts < 100: # Prevent infinite loops
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(generation_radius * 0.4, generation_radius) # Ensure target is reasonably far
            
            offset_x = dist * np.cos(angle)
            offset_y = dist * np.sin(angle)
            offset_z = np.random.uniform(-generation_radius * 0.3, generation_radius * 0.3) 

            target_x = drone_start_pos[0] + offset_x
            target_y = drone_start_pos[1] + offset_y
            target_z = drone_start_pos[2] + offset_z
            
            target_x = np.clip(target_x, arena_bounds['x'][0], arena_bounds['x'][1])
            target_y = np.clip(target_y, arena_bounds['y'][0], arena_bounds['y'][1])
            target_z = np.clip(target_z, arena_bounds['z'][0], arena_bounds['z'][1])
            
            targets[i] = [target_x, target_y, target_z]
            if np.linalg.norm(targets[i] - drone_start_pos) > 0.1: # Target should be at least 20cm away
                break
            attempts += 1
        if attempts >= 50: # Fallback if a good target isn't found
            print(f"[WARNING] Could not find distinct random target for drone {i} after {attempts} attempts. Using start_pos + small offset.")
            targets[i] = drone_start_pos + np.random.normal(0, 0.1, 3) # Small random offset
            targets[i, 2] = np.clip(targets[i, 2], min_target_height, max_target_height)

    return targets

# --- Function to perturb formation center (NEW) ---
def _perturb_formation_center(initial_xyzs, radius=2.0):
    """
    Perturbs the center of a drone formation within a given radius.
    
    Args:
        initial_xyzs: Original drone positions (Nx3 numpy array)
        radius: Maximum radius for perturbation (default: 2.0 meters)
        
    Returns:
        Numpy array with perturbed drone positions
    """
    # Calculate the current center of the formation
    formation_center = np.mean(initial_xyzs, axis=0)
    
    # Generate a random perturbation vector within the specified radius
    angle = np.random.uniform(0, 2 * np.pi)
    distance = np.random.uniform(0, radius)
    
    # Convert to cartesian coordinates (x,y perturbation, keeping z the same)
    perturbation = np.array([
        distance * np.cos(angle),
        distance * np.sin(angle),
        0.0  # No perturbation in z-axis
    ])
    
    # Apply the perturbation to all drone positions
    perturbed_positions = initial_xyzs + perturbation
    
    # Ensure the perturbed positions are within valid arena bounds
    # Assuming arena bounds of [-4.5, 4.5] for x and y, and [0.2, 4.0] for z
    arena_bounds = {
        'x': (-4.5, 4.5),
        'y': (-4.5, 4.5),
        'z': (0.2, 4.0)
    }
    
    # Clip positions to ensure they're within bounds
    perturbed_positions[:, 0] = np.clip(perturbed_positions[:, 0], arena_bounds['x'][0], arena_bounds['x'][1])
    perturbed_positions[:, 1] = np.clip(perturbed_positions[:, 1], arena_bounds['y'][0], arena_bounds['y'][1])
    perturbed_positions[:, 2] = np.clip(perturbed_positions[:, 2], arena_bounds['z'][0], arena_bounds['z'][1])
    
    return perturbed_positions

# --- New Function: Generate Random Leader Target ---
def _generate_random_leader_target(current_leader_pos, move_distance=0.2):
    """
    Generates a random target for the leader drone that is move_distance units away
    from the current position in some random direction.
    
    Args:
        current_leader_pos: Current position of the leader drone [x, y, z]
        move_distance: Distance to move in the random direction
    
    Returns:
        Numpy array with new target position
    """
    # Generate a random unit vector in 3D space
    random_direction = np.random.uniform(-1, 1, 3)
    # Normalize to get a unit vector
    random_direction = random_direction / np.linalg.norm(random_direction)
    
    # Generate new position
    new_pos = current_leader_pos + random_direction * move_distance
    
    # Ensure the new position is within valid arena bounds
    arena_bounds = {
        'x': (-4.5, 4.5),
        'y': (-4.5, 4.5),
        'z': (0.5, 3.0)  # Keep z within reasonable flight bounds
    }
    
    new_pos[0] = np.clip(new_pos[0], arena_bounds['x'][0], arena_bounds['x'][1])
    new_pos[1] = np.clip(new_pos[1], arena_bounds['y'][0], arena_bounds['y'][1])
    new_pos[2] = np.clip(new_pos[2], arena_bounds['z'][0], arena_bounds['z'][1])
    
    return new_pos

# --- New Function: Update Drone Positions History ---
def _update_drone_positions_history(drone_positions):
    """
    Updates the history of drone positions.
    
    Args:
        drone_positions: Current positions of all drones (array of shape [num_drones, 3])
    """
    global drone_positions_history
    
    # Add the current drone positions to history
    drone_positions_history.append(drone_positions.copy())
    
    # Keep only the most recent positions (for memory efficiency)
    if len(drone_positions_history) > HISTORY_MAX_ENTRIES:
        drone_positions_history.pop(0)

# --- New Function: Calculate Follower Accuracy ---
def _calculate_follower_accuracy(follower_pos, target_pos, tolerance=0.1):
    """
    Calculates how accurately a follower drone is tracking its target position.
    
    Args:
        follower_pos: Current position of the follower drone
        target_pos: Position the follower should be tracking (previous drone's last position)
        tolerance: Acceptable distance tolerance (default: 0.1 meters)
    
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    distance = np.linalg.norm(follower_pos - target_pos)
    
    # If within tolerance, consider perfect accuracy
    if distance <= tolerance:
        return 1.0
    
    # Otherwise, scale accuracy inversely with distance
    # Max expected distance is 2.0 meters
    accuracy = max(0.0, 1.0 - (distance - tolerance) / 2.0)
    return accuracy

# --- XAI Explainer Classes (SHAPExplainer and DiCEExplainer) ---
class SHAPExplainer:
    """Adds SHAP explanations to RL decisions."""
    def __init__(self, model, obs_space_example): 
        if not SHAP_AVAILABLE:
            print("[WARNING] SHAP library not available. SHAPExplainer will not function.")
            self.explainer = None
            self.model = model 
            self.feature_names = [] 
            return

        self.model = model
        
        if isinstance(obs_space_example, dict) and 'kin' in obs_space_example: 
            kin_features = obs_space_example['kin'].shape[0]
            self.feature_names = [f"kin_feat_{i}" for i in range(kin_features)]
            if kin_features == 0:
                print("[WARNING] SHAPExplainer: KIN_DEPTH 'kin' observation space has 0 features.")
                self.background_data_source = np.array([]).reshape(0,0)
                self.explainer = None; return
            self.background_data_source = np.random.uniform(-1, 1, size=(50, kin_features)).astype(np.float32)
        elif isinstance(obs_space_example, np.ndarray): 
            if obs_space_example.ndim == 1:
                 kin_features = obs_space_example.shape[0]
                 if kin_features == 0:
                     print("[WARNING] SHAPExplainer: KIN observation space has 0 features.")
                     self.background_data_source = np.array([]).reshape(0,0)
                     self.explainer = None; return
                 self.feature_names = [f"kin_feat_{i}" for i in range(kin_features)]
                 self.background_data_source = np.random.uniform(-1, 1, size=(50, kin_features)).astype(np.float32)
            else: 
                 print(f"[WARNING] SHAPExplainer: KIN observation space is {obs_space_example.ndim}D, expected 1D for sample. Using size.")
                 kin_features = obs_space_example.size
                 if kin_features == 0:
                     print("[WARNING] SHAPExplainer: KIN observation space has 0 features (from size).")
                     self.background_data_source = np.array([]).reshape(0,0)
                     self.explainer = None; return
                 self.feature_names = [f"kin_feat_{i}" for i in range(kin_features)]
                 self.background_data_source = np.random.uniform(-1, 1, size=(50, kin_features)).astype(np.float32) 
        else:
            print("[WARNING] SHAPExplainer: Unknown observation space structure for sample. Defaulting feature names.")
            self.feature_names = []
            self.background_data_source = np.array([]).reshape(0,0) 
            self.explainer = None
            return
        
        if self.background_data_source.shape[1] == 0: 
            print("[WARNING] SHAPExplainer: No features identified for background data. SHAP disabled.")
            self.explainer = None
            return

        try:
            def model_predict_wrapper(x_kin_batch): 
                actions_list_of_1D_arrays = []
                for i in range(x_kin_batch.shape[0]): 
                    x_kin_single = x_kin_batch[i] 
                    current_full_obs_for_model = None
                    if isinstance(obs_space_example, dict) and 'kin' in obs_space_example:
                        dummy_depth = np.zeros_like(obs_space_example['depth']) 
                        current_full_obs_for_model = {"kin": x_kin_single, "depth": dummy_depth}
                    else: 
                        current_full_obs_for_model = x_kin_single
                    action, _ = self.model.predict(current_full_obs_for_model, deterministic=True)
                    if not isinstance(action, np.ndarray): action = np.array(action)
                    if action.ndim == 0: action = np.array([action]) 
                    elif action.ndim > 1: action = action.flatten() 
                    actions_list_of_1D_arrays.append(action)
                if not actions_list_of_1D_arrays: 
                    action_dim = 1 
                    if hasattr(self.model, 'action_space') and hasattr(self.model.action_space, 'shape') and self.model.action_space.shape:
                        action_dim = self.model.action_space.shape[0] if self.model.action_space.shape else 1
                    return np.array([]).reshape(0, action_dim) 
                return np.vstack(actions_list_of_1D_arrays)
            
            self.explainer = shap.KernelExplainer(model_predict_wrapper, self.background_data_source)
            print("[INFO] SHAP explainer initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Error initializing SHAP explainer: {e}. SHAP will not be used for explanations.")
            self.explainer = None

    def explain_action(self, observation_full): 
        if self.explainer is None:
            action, _ = self.model.predict(observation_full, deterministic=True)
            action_1d = action[0] if action.ndim > 1 else action
            if isinstance(observation_full, dict) and 'kin' in observation_full:
                return np.zeros_like(observation_full['kin']), action_1d
            elif isinstance(observation_full, np.ndarray):
                return np.zeros_like(observation_full), action_1d
            else: 
                return np.array([]), action_1d

        observation_kin = None
        if isinstance(observation_full, dict) and 'kin' in observation_full:
            observation_kin = observation_full['kin']
        elif isinstance(observation_full, np.ndarray):
            observation_kin = observation_full
        
        if observation_kin is None or observation_kin.size == 0: 
            print("[WARNING] SHAP: Invalid or empty kinematic observation for explanation.")
            action, _ = self.model.predict(observation_full, deterministic=True)
            action_1d = action[0] if action.ndim > 1 else action
            return np.zeros(len(self.feature_names) if self.feature_names else 0), action_1d

        try:
            observation_kin_2d = observation_kin.reshape(1, -1)
            n_samples_val = 50 
            raw_shap_values = self.explainer.shap_values(observation_kin_2d, nsamples=n_samples_val)
            shap_values_list_for_agg = [raw_shap_values] if not isinstance(raw_shap_values, list) else raw_shap_values
            processed_shap_values_1D_list = []
            for shap_val_for_action_dim in shap_values_list_for_agg:
                if isinstance(shap_val_for_action_dim, np.ndarray) and \
                   shap_val_for_action_dim.ndim == 2 and \
                   shap_val_for_action_dim.shape[0] == 1:
                    processed_shap_values_1D_list.append(shap_val_for_action_dim[0]) 
                else:
                    print(f"[WARNING] SHAP: Unexpected item shape {getattr(shap_val_for_action_dim, 'shape', type(shap_val_for_action_dim))} in shap_values_list. Expected (1, num_features).")
                    processed_shap_values_1D_list.append(np.zeros(observation_kin.shape[0]))
            aggregated_shap_per_feature = np.zeros_like(observation_kin) 
            if processed_shap_values_1D_list:
                 if all(s.shape == processed_shap_values_1D_list[0].shape for s in processed_shap_values_1D_list):
                      aggregated_shap_per_feature = np.mean(np.abs(np.array(processed_shap_values_1D_list)), axis=0)
                 else:
                      print("[WARNING] SHAP: Mismatch in SHAP value shapes for different action dimensions. Using zeros for aggregation.")
            action, _ = self.model.predict(observation_full, deterministic=True)
            action_1d = action[0] if action.ndim > 1 else action
            return aggregated_shap_per_feature, action_1d
        except Exception as e:
            print(f"[WARNING] Error generating SHAP values: {e}. Returning default explanation.")
            action, _ = self.model.predict(observation_full, deterministic=True)
            action_1d = action[0] if action.ndim > 1 else action
            if isinstance(observation_full, dict) and 'kin' in observation_full:
                return np.zeros_like(observation_full['kin']), action_1d
            elif isinstance(observation_full, np.ndarray):
                return np.zeros_like(observation_full), action_1d
            else:
                return np.array([]), action_1d

    def get_explanation_text(self, observation, drone_id):
        shap_values_aggregated, action = self.explain_action(observation)
        if self.explainer is None or np.all(np.isclose(shap_values_aggregated, 0)): 
            return f"Drone {drone_id} decision based on policy. Action: {action.round(2)} (SHAP unavailable or values are zero)"
        current_feature_names = self.feature_names
        num_actual_features = 0
        if isinstance(observation, dict) and 'kin' in observation:
            num_actual_features = observation['kin'].shape[0]
        elif isinstance(observation, np.ndarray):
            num_actual_features = observation.shape[0]
        if len(shap_values_aggregated) != num_actual_features:
            print(f"[WARNING] SHAP text: Mismatch between aggregated SHAP values ({len(shap_values_aggregated)}) and actual features ({num_actual_features}).")
            if len(self.feature_names) != len(shap_values_aggregated):
                 return f"Drone {drone_id} action {action.round(2)}. (SHAP feature name/value length mismatch)"
        feature_importance = sorted([(current_feature_names[i], shap_values_aggregated[i]) for i in range(len(shap_values_aggregated))], key=lambda x: abs(x[1]), reverse=True)
        explanation = f"Drone {drone_id}'s action {action.round(2)} influenced by: "
        top_features = feature_importance[:min(3, len(feature_importance))]
        if not top_features: return f"Drone {drone_id} action {action.round(2)}. (No dominant features in SHAP)"
        explanation_parts = [f"{name} (imp: {abs(val):.2f})" for name, val in top_features if abs(val) > 1e-3] 
        if not explanation_parts: return f"Drone {drone_id} action {action.round(2)}. (Feature importances very low)"
        return explanation + ", ".join(explanation_parts) + "."

class DiCEExplainer: 
    def __init__(self, model, obs_space_example):
        if not DICE_AVAILABLE:
            self.dice_model = None; self.dice_explainer_instance = None
            print("[WARNING] DiCE library not available. DiCEExplainer will not function."); return
        obs_for_dice, num_features = None, 0
        if isinstance(obs_space_example, dict) and 'kin' in obs_space_example:
            obs_for_dice = obs_space_example['kin']
            if obs_for_dice.ndim == 1: num_features = obs_for_dice.shape[0]
        elif isinstance(obs_space_example, np.ndarray) and obs_space_example.ndim == 1:
            obs_for_dice = obs_space_example; num_features = obs_for_dice.shape[0]
        if num_features == 0:
            self.dice_model = None; self.dice_explainer_instance = None
            print("[WARNING] DiCE: Could not determine features from obs_space_example for DiCE init."); return
        self.model = model 
        self.feature_names = [f"kin_feat_{i}" for i in range(num_features)]
        try:
            random_data = np.random.uniform(-1, 1, size=(100, num_features)).astype(np.float32)
            self.data_df = pd.DataFrame(random_data, columns=self.feature_names)
            self.dice_data = dice_ml.Data(dataframe=self.data_df, continuous_features=self.feature_names, outcome_name="action_class")
            def rl_model_predict_class(X_df_or_np):
                X_np = X_df_or_np.values.astype(np.float32) if isinstance(X_df_or_np, pd.DataFrame) else X_df_or_np.astype(np.float32)
                classes = []
                for i in range(X_np.shape[0]):
                    current_kin_obs_single = X_np[i]
                    current_full_obs_for_model = {'kin': current_kin_obs_single, 'depth': np.zeros_like(obs_space_example['depth'])} if isinstance(obs_space_example, dict) and 'kin' in obs_space_example else current_kin_obs_single
                    action, _ = self.model.predict(current_full_obs_for_model, deterministic=True)
                    action_1d = action[0] if action.ndim > 1 else action
                    classes.append(1 if action_1d.size > 0 and action_1d[0] > 0.1 else 0) 
                return np.array(classes)
            self.dice_backend_model = dice_ml.Model(model=rl_model_predict_class, backend="sklearn") 
            self.dice_explainer_instance = dice_ml.Dice(self.dice_data, self.dice_backend_model, method="random") 
            print("[INFO] DiCE explainer initialized successfully.")
        except Exception as e:
            self.dice_model = None; self.dice_explainer_instance = None
            print(f"[WARNING] Error initializing DiCE explainer: {e}. DiCE will not be used.")
            
    def get_counterfactual(self, observation_full, drone_id):
        if not DICE_AVAILABLE or self.dice_explainer_instance is None:
            return f"Drone {drone_id}: (DiCE unavailable or not initialized)"
        observation_kin = observation_full['kin'] if isinstance(observation_full,dict) and 'kin' in observation_full else \
                          observation_full if isinstance(observation_full, np.ndarray) and observation_full.ndim==1 else None
        if observation_kin is None or observation_kin.size == 0: return f"Drone {drone_id}: (DiCE: Unsuitable observation for CF)"
        query_instance_df = pd.DataFrame([observation_kin], columns=self.feature_names)
        try:
            current_class_arr = self.dice_backend_model.get_output(query_instance_df.values.astype(np.float32), model_score=False)
            current_class = current_class_arr[0] if current_class_arr.size > 0 else 0; desired_class = 1 - current_class
            cf_examples = self.dice_explainer_instance.generate_counterfactuals(query_instance_df, total_CFs=1, desired_class=desired_class, verbose=False)
            if cf_examples and cf_examples.cf_examples_list and cf_examples.cf_examples_list[0].final_cfs_df_list and not cf_examples.cf_examples_list[0].final_cfs_df_list[0].empty:
                final_cf_df = cf_examples.cf_examples_list[0].final_cfs_df_list[0]
                cf_dict, orig_dict, changes = final_cf_df.iloc[0].to_dict(), query_instance_df.iloc[0].to_dict(), []
                for key in cf_dict:
                    if key in orig_dict and key in self.feature_names and abs(orig_dict[key]-cf_dict[key]) > 0.01: 
                        changes.append(f"{key}: {orig_dict[key]:.2f} -> {cf_dict[key]:.2f}")
                return f"Drone {drone_id} action class {current_class} -> {desired_class} if: {', '.join(changes[:2])}" if changes else f"Drone {drone_id} (DiCE: No simple CF for class {desired_class})"
            return f"Drone {drone_id} (DiCE: No CF generated for class {desired_class})"
        except Exception as e: print(f"[ERROR] DiCE CF gen failed for drone {drone_id}: {e}"); return f"Drone {drone_id} (DiCE Error)"

# --- History Formatting for LLM ---
def format_history_for_prompt():
    if not drone_state_history:
        return "No previous history available.\n"
    history_text = "### Drone Command and Behavior History (Most Recent First):\n"
    for idx, (timestamp, cmd, states, targets, actions, explanations) in enumerate(reversed(list(drone_state_history))):
        history_text += f"\n## Event {idx+1} (Sim Time: {timestamp:.2f}s, Command: \"{cmd}\"):\n"
        history_text += "Drone Details:\n"
        for i, state in enumerate(states):
            pos_str = ", ".join([f"{val:.2f}" for val in state['position']])
            vel_str = ", ".join([f"{val:.2f}" for val in state['velocity']])
            tgt_str = "N/A"
            if targets is not None and i < len(targets) and targets[i] is not None:
                tgt_str = ", ".join([f"{val:.2f}" for val in targets[i]])
            action_str = "N/A"
            if actions is not None and i < len(actions) and actions[i] is not None:
                 action_str = ", ".join([f"{val:.2f}" for val in actions[i]])
            history_text += (
                f"- Drone {state['id']}: Pos=[{pos_str}], Vel=[{vel_str}], Target=[{tgt_str}], "
                f"Action=[{action_str}]\n"
            )
            if explanations and i < len(explanations) and explanations[i] is not None:
                history_text += f"  Explanation: {explanations[i]}\n"
    return history_text

# --- LLM Target Generation (MODIFIED for new task description) ---
def get_llm_targets_or_actions(command, drone_states, num_drones, current_llm_output,
                    task_target_positions=None, current_actions=None, explanation_text=None,
                    return_type="targets", task_id=None):
    if not gemini_available or gemini_model is None:
        print(f"[LLM_FALLBACK] Gemini API not available. Using fallback {return_type}.")
        if return_type == "targets":
            if task_target_positions is not None and len(task_target_positions) == num_drones:
                return np.array(task_target_positions, dtype=np.float32)
            new_output_fallback = []
            for i in range(num_drones):
                if drone_states[i] and 'position' in drone_states[i]:
                    new_output_fallback.append([
                        drone_states[i]['position'][0],
                        drone_states[i]['position'][1],
                        max(0.2, drone_states[i]['position'][2] + 0.1) 
                    ])
                elif current_llm_output is not None and i < len(current_llm_output) and hasattr(current_llm_output[i], '__len__') and len(current_llm_output[i]) == 3 :
                    new_output_fallback.append(current_llm_output[i].tolist() if isinstance(current_llm_output[i], np.ndarray) else current_llm_output[i])
                else:
                    new_output_fallback.append([i*0.5 - (num_drones-1)*0.25, 0.0, 1.0]) 
            return np.array(new_output_fallback, dtype=np.float32)
        elif return_type == "actions":
            return np.array([[0.0, 0.0, 0.0, 0.0]] * num_drones, dtype=np.float32) 

    # Build the prompt based on the task and drone states
    prompt = f"""You are an AI orchestrator for a swarm of {num_drones} drones. Your primary goal is to guide the drones to specific target positions or directly control their actions based on a human command, while ensuring safe and coordinated flight.
Human command: "{command}"
Current drone states (position is [x,y,z], velocity is [vx,vy,vz]):
"""
    for i, state in enumerate(drone_states):
        pos_str = ", ".join([f"{val:.2f}" for val in state['position']])
        vel_str = ", ".join([f"{val:.2f}" for val in state['velocity']])
        current_action_str = "N/A"
        if current_actions is not None and i < len(current_actions) and current_actions[i] is not None:
            current_action_str = ", ".join([f"{val:.2f}" for val in current_actions[i]])
        prompt += f"Drone {state['id']}: Position=[{pos_str}], Velocity=[{vel_str}], Last Sent Action=[{current_action_str}]\n"

    if task_target_positions is not None:
        prompt += "\nGlobal task target positions for each drone (Drone 0, Drone 1, ...):\n"
        for i, target in enumerate(task_target_positions):
             if target is not None: prompt += f"Drone {i} desires to reach: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]\n"
        if return_type == "targets":
            prompt += "Your role is to refine or assign intermediate targets for each drone based on these desired targets, current states, and the human command. Ensure the final task goal is achieved.\n"
    else:
        prompt += "\nNo specific global task targets provided. You must infer the targets/actions from the human command and current states.\n"
    
    # Add task-specific context
    if task_id == "circle_behavior" and return_type == "targets":
        # Get current positions for all drones
        current_positions = np.array([state['position'] for state in drone_states])
        
        # Update drone positions history
        _update_drone_positions_history(current_positions)
        
        # Get previous positions if available
        previous_positions = None
        if len(drone_positions_history) > 1:
            previous_positions = drone_positions_history[-2]
        
        # Add random leader with followers context
        prompt += f"""
SPECIAL TASK: RANDOM LEADER WITH FOLLOWERS
- You're coordinating a leader-follower behavior pattern
- Drone 0 (leader) should move RANDOMLY approximately 0.2 meters in any direction from its current position
- Current leader position: {current_positions[0].tolist()}
- Drones 1-3 (followers) should each target the PREVIOUS position of the drone ahead of them
- Drone 1 should target the previous position of Drone 0
- Drone 2 should target the previous position of Drone 1
- Drone 3 should target the previous position of Drone 2
- A tolerance of 0.1 meters is allowed for follower positioning

Current positions of all drones:
"""
        for i, pos in enumerate(current_positions):
            prompt += f"- Drone {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n"
        
        if previous_positions is not None:
            prompt += "\nPrevious positions of all drones:\n"
            for i, pos in enumerate(previous_positions):
                prompt += f"- Drone {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n"
        
        prompt += """
For the leader (Drone 0):
- Generate a random target that is approximately 0.2 meters away from its current position
- Keep the height (z) relatively stable

For the followers (Drones 1-3):
- Set Drone 1's target to Drone 0's previous position
- Set Drone 2's target to Drone 1's previous position
- Set Drone 3's target to Drone 2's previous position
- If previous positions aren't available yet, use appropriate offsets from current positions
"""
    elif task_id == "high_path_movement" and return_type == "targets":
        prompt += """
SPECIAL TASK: HIGH PATH MOVEMENT
Each drone must complete this flight sequence:
1. FIRST: Rise to a height of 1.5m (if current height < 1.4m)
2. THEN: Fly horizontally to the target X,Y coordinates 
3. ONLY WHEN near target X,Y (within 0.2m): Descend to the final target height

You must determine which phase each drone is in based on its current position and set appropriate intermediate targets.
"""
    elif task_id == "low_path_movement" and return_type == "targets":
        prompt += """
SPECIAL TASK: LOW PATH MOVEMENT
Each drone must complete this flight sequence:
1. FIRST: Descend to a height of 0.2m (if current height > 0.25m)
2. THEN: Fly horizontally to the target X,Y coordinates 
3. ONLY WHEN near target X,Y (within 0.2m): Rise to the final target height

You must determine which phase each drone is in based on its current position and set appropriate intermediate targets.
"""
    
    if explanation_text:
        prompt += f"\nAdditional insights/explanations from drone's low-level behavior:\n{explanation_text}\n"
    
    prompt += "History, from nevest to oldest: " + format_history_for_prompt()
    
    if return_type == "targets":
        prompt += f"""
Based on the human command, current drone states, their last sent actions, provided global task targets (if any), and the behavior history, provide new 3D target coordinates (x, y, z) for each of the {num_drones} drones.
The simulation environment is roughly a 10m x 10m x 5m box. Drones typically operate well between z=0.2m and z=4.0m. X and Y can be -4.5 to 4.5.
Prioritize collision avoidance between drones. If multiple drones need to pass through the same area, stagger their movements or adjust paths.
Your response MUST be a JSON object with a single key "targets".
The value of "targets" MUST be a list of lists, where each inner list contains exactly three float numbers [x, y, z] representing the target for one drone.
The order of targets in the list MUST correspond to the drone IDs (Drone 0, Drone 1, ..., Drone N-1).
Example target: [1.5, -0.5, 1.2]
"""
        json_key = "targets"
    elif return_type == "actions":
        prompt += f"""
Based on the human command, current drone states, their current targets (if any), and the behavior history, provide new direct action values for each of the {num_drones} drones.
Each drone's action should be a list of 4 float values, representing normalized control inputs for its motors. These values should range from -1.0 to 1.0.
A value of 0.0 for an input dimension generally means maintain current state for that dimension or apply base thrust. Positive values typically increase thrust/rate, negative values decrease.
The action vector is [collective_thrust_adjustment, roll_adjustment, pitch_adjustment, yaw_adjustment].
For example, [0.2, 0.0, 0.0, 0.0] means slightly increase collective thrust. [0.0, 0.1, 0.0, 0.0] means roll slightly.
The simulation environment is roughly a 10m x 10m x 5m box.
Prioritize collision avoidance between drones.
Your response MUST be a JSON object with a single key "actions".
The value of "actions" MUST be a list of lists, where each inner list contains exactly four float numbers [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd] representing the action for one drone.
The order of actions in the list MUST correspond to the drone IDs (Drone 0, Drone 1, ..., Drone N-1).
Example action: [0.1, -0.05, 0.0, 0.02]
"""
        json_key = "actions"

    max_retries = 2; response_text_for_error = "No response"
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            response_text_for_error = response.text
            parsed_response = json.loads(response.text)
            new_output_list = parsed_response.get(json_key)
            expected_dim = 3 if return_type == "targets" else 4
            if not isinstance(new_output_list, list) or len(new_output_list) != num_drones:
                print(f"[LLM_ERROR] Response '{json_key}' not list of length {num_drones}. Resp: {response.text}")
                if attempt < max_retries -1: print("Retrying LLM..."); continue
                return current_llm_output 
            new_output_np = np.zeros((num_drones, expected_dim), dtype=np.float32)
            valid_output = True
            for i in range(num_drones):
                if isinstance(new_output_list[i], list) and len(new_output_list[i]) == expected_dim:
                    try:
                        float_values = [float(val) for val in new_output_list[i]]
                        if return_type == "targets":
                            x,y,z = np.clip(float_values[0],-4.5,4.5), np.clip(float_values[1],-4.5,4.5), np.clip(float_values[2],0.2,4.0) 
                            new_output_np[i] = [x,y,z]
                        elif return_type == "actions": new_output_np[i] = np.clip(float_values, -1.0, 1.0)
                    except (ValueError, TypeError): valid_output=False; break
                else: valid_output=False; break
            if valid_output: 
                # For follower task, store the targets for accuracy calculation
                if task_id == "circle_behavior" and return_type == "targets":
                    global follower_target_history
                    
                    # Add the new targets to follower history
                    follower_target_history.append(new_output_np.copy())
                    
                    # Keep only recent history
                    if len(follower_target_history) > HISTORY_MAX_ENTRIES:
                        follower_target_history.pop(0)
                    
                    # Debug output for leader and followers
                    leader_target = new_output_np[0]
                    leader_current = np.array(drone_states[0]['position'])
                    move_dist = np.linalg.norm(leader_target - leader_current)
                    
                    print(f"[FOLLOWER TASK] Leader moving: {move_dist:.2f}m")
                    for i in range(1, num_drones):
                        if len(drone_positions_history) > 1:
                            # Calculate what this follower's target should be (previous position of drone i-1)
                            ideal_target = drone_positions_history[-2][i-1]
                            actual_target = new_output_np[i]
                            target_diff = np.linalg.norm(actual_target - ideal_target)
                            print(f"[FOLLOWER TASK] Drone {i}: target diff from ideal: {target_diff:.2f}m")
                
                return new_output_np
            else:
                print(f"[LLM_ERROR] Invalid {json_key} format/values in LLM response: {new_output_list}. Using prev.")
                if attempt < max_retries -1: print("Retrying LLM..."); continue
                return current_llm_output
        except json.JSONDecodeError: print(f"[LLM_ERROR] JSONDecodeError: {response_text_for_error}")
        except Exception as e: print(f"[LLM_ERROR] API/processing error: {e}\nLLM response: {response_text_for_error}")
        if attempt < max_retries-1: print("Retrying LLM..."); continue
    return current_llm_output

# --- Model Loading ---
def detect_algorithm_from_model(model_path, env_for_load):
    algorithms_to_try = {'ppo': PPO, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG}
    for algo_name, algo_class in algorithms_to_try.items():
        try:
            custom_objects = { "observation_space": env_for_load.observation_space, "action_space": env_for_load.action_space }
            model = algo_class.load(model_path, env=None, custom_objects=custom_objects, print_system_info=False)
            print(f"[INFO] Detected algorithm: {algo_name.upper()}"); return algo_name, model
        except: continue
    raise RuntimeError(f"Failed to load model: {model_path} with any supported algorithm.")

# --- Main Evaluation Loop (MODIFIED for new success criteria) ---
def run_evaluation(model_path, output_folder=DEFAULT_OUTPUT_FOLDER, num_episodes=NUM_EPISODES):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_folder = os.path.join(output_folder, f"eval_{timestamp}")
    os.makedirs(eval_folder, exist_ok=True)
    
    print(f"Starting evaluation with {num_episodes} episodes per task/approach combination...")
    
    base_env_params = {
        'drone_model': DroneModel.CF2X, 'num_drones': DEFAULT_NUM_DRONES, 'initial_rpys': None, 
        'physics': Physics.PYB, 'pyb_freq': 240, 'ctrl_freq': DEFAULT_CTRL_FREQ, 
        'gui': DEFAULT_GUI, 'record': DEFAULT_RECORD_VIDEO, 'obs': DEFAULT_OBS, 'act': DEFAULT_ACT, 
        'episode_length_sec': DEFAULT_DURATION_SEC, 'target_radius_start': 0.5, 'target_radius_max': 3.0,   
        'target_radius_increment': 0.1, 'target_tolerance': 0.15, 'add_obstacles': False, 
        'obs_prob': 0.3, 'obstacle_size': 0.15,  'never_end': False 
    }
    single_drone_env_params_for_load = {**base_env_params, 'num_drones': 1, 'gui': False, 'add_obstacles': False}
    
    print("[INFO] Creating temp single-drone env for model loading...")
    temp_env_for_load = MultiTargetAviary(**single_drone_env_params_for_load)
    try:
        algo_name, model = detect_algorithm_from_model(model_path, temp_env_for_load)
        print(f"[INFO] RL Model {model_path} loaded successfully: {algo_name.upper()}.")
    except Exception as e: print(f"[ERROR] Failed to load RL model: {e}"); temp_env_for_load.close(); return
    finally: temp_env_for_load.close(); print("[INFO] Temp model-loading env closed.")
    
    shap_explainer, dice_explainer = None, None
    sample_obs_for_explainer = model.observation_space.sample()
    if SHAP_AVAILABLE:
        try: shap_explainer = SHAPExplainer(model, sample_obs_for_explainer)
        except Exception as e: print(f"[WARNING] Failed to init SHAP: {e}.")
    if DICE_AVAILABLE and "RLALLMA+DiCE" in APPROACHES:
        try: dice_explainer = DiCEExplainer(model, sample_obs_for_explainer)
        except Exception as e: print(f"[WARNING] Failed to init DiCE: {e}.")

    for task_id, task_info in TASKS.items():
        print(f"\n{'='*50}\nEvaluating task: {task_info['name']}\nDescription: {task_info['description']}\n{'='*50}")
        
        for episode in range(num_episodes): # EPISODE LOOP IS NOW OUTER FOR A GIVEN TASK
            print(f"  Episode {episode+1}/{num_episodes} for Task '{task_info['name']}'...")

            # Determine initial XYZs for THIS episode (consistent for all approaches)
            # Get the base initial positions
            base_initial_xyzs = task_info.get("custom_initial_xyzs", DEFAULT_EPISODE_INITIAL_XYZS).copy()
            
            # For tests 2 and 3 (spread_movement and tight_movement), perturb the center
            if task_id in ["spread_movement", "tight_movement"]:
                # First perturb the center by 2 meters
                rad = 2.0
                # if task_id == "tight_movement":
                #     rad = 0.25
                episode_initial_xyzs = _perturb_formation_center(base_initial_xyzs, radius=rad)
                print(f"    Applied center perturbation for {task_id} (test {2 if task_id=='spread_movement' else 3})")
            else:
                # For other tasks (like basic_movement), use the original positions
                episode_initial_xyzs = base_initial_xyzs
            
            # Generate random mission targets for THIS episode (consistent for all approaches)
            # Skip random target generation for behavioral tasks
            if task_info.get("is_behavioral", False):
                # For behavioral tasks, don't generate random targets
                generated_mission_targets_for_episode = np.zeros((DEFAULT_NUM_DRONES, 3), dtype=np.float32)
                # For the random leader task, just set placeholder targets
                if task_id == "circle_behavior":
                    # Reset tracking variables
                    drone_positions_history.clear()
                    follower_target_history.clear()
            else:
                target_gen_radius = task_info.get("target_generation_radius", 2.0) # Default if not in task_info
                generated_mission_targets_for_episode = _generate_random_mission_targets(
                    episode_initial_xyzs, target_gen_radius, DEFAULT_NUM_DRONES
                )
            
            print(f"    Episode Initial XYZs: {episode_initial_xyzs.tolist()}")
            if not task_info.get("is_behavioral", False):
                print(f"    Generated Mission Targets for Episode: {generated_mission_targets_for_episode.tolist()}")

            for approach in APPROACHES: # APPROACH LOOP IS NOW INNER
                # Skip certain approach-task combinations if needed
                if approach == "RL" and task_id not in ["basic_movement", "high_path_movement", "low_path_movement"]:
                    print(f"    Skipping {approach} (Pure RL not suitable for {task_id})")
                    results[task_id][approach]["reward"].append(np.nan) # Store NaN for skipped
                    results[task_id][approach]["success_rate"].append(np.nan)
                    results[task_id][approach]["crash_percentage"].append(np.nan)
                    results[task_id][approach]["time_high_altitude_pct"].append(np.nan)
                    results[task_id][approach]["time_low_altitude_pct"].append(np.nan)
                    results[task_id][approach]["follower_accuracy"].append(np.nan)
                    continue
                
                if (approach=="RLALLMA+SHAP" and (not SHAP_AVAILABLE or not shap_explainer or not shap_explainer.explainer)) or \
                   (approach=="RLALLMA+DiCE" and (not DICE_AVAILABLE or not dice_explainer or not dice_explainer.dice_explainer_instance)):
                    print(f"    Skipping {approach} (Explainer unavailable/uninit)")
                    results[task_id][approach]["reward"].append(np.nan)
                    results[task_id][approach]["success_rate"].append(np.nan)
                    results[task_id][approach]["crash_percentage"].append(np.nan)
                    results[task_id][approach]["time_high_altitude_pct"].append(np.nan)
                    results[task_id][approach]["time_low_altitude_pct"].append(np.nan)
                    results[task_id][approach]["follower_accuracy"].append(np.nan)
                    continue
                    
                print(f"    Running Approach: {approach} for episode {episode+1}...")
                drone_state_history.clear()
                
                # Reset task-specific tracking variables
                drone_positions_history.clear()
                follower_target_history.clear()
                
                current_episode_env_params = base_env_params.copy()
                current_episode_env_params['initial_xyzs'] = episode_initial_xyzs # Use episode-specific initial XYZs
                
                episode_env = MultiTargetAviary(**current_episode_env_params)
                
                # Initialize task-specific tracking variables
                if task_id == "high_path_movement" or task_id == "low_path_movement":
                    episode_env._drone_reached_high_altitude = {i: False for i in range(DEFAULT_NUM_DRONES)}
                    episode_env._drone_reached_low_altitude = {i: False for i in range(DEFAULT_NUM_DRONES)}
                
                if task_id == "circle_behavior":
                    # For the new follower task
                    episode_env._follower_accuracy = {i: 0.0 for i in range(1, DEFAULT_NUM_DRONES)}
                
                # Initialize altitude tracking for all tasks
                time_at_high_altitude = {i: 0 for i in range(DEFAULT_NUM_DRONES)}
                time_at_low_altitude = {i: 0 for i in range(DEFAULT_NUM_DRONES)}
                
                obs_tuple, info_dict = episode_env.reset()
                
                # Use the generated targets for this specific episode run
                if not task_info.get("is_behavioral", False):
                    initial_mission_targets_for_this_run = generated_mission_targets_for_episode.copy()
                else:
                    # For behavioral tasks, we'll generate targets dynamically
                    initial_mission_targets_for_this_run = np.zeros((DEFAULT_NUM_DRONES, 3), dtype=np.float32)
                
                llm_orchestrated_targets = initial_mission_targets_for_this_run.copy()
                episode_env.current_targets = llm_orchestrated_targets 
                llm_orchestrated_actions = np.zeros((DEFAULT_NUM_DRONES, 4), dtype=np.float32) 
                
                episode_reward_sum, episode_success, episode_crash, steps_taken = 0, False, False, 0
                max_steps = int(DEFAULT_DURATION_SEC * DEFAULT_CTRL_FREQ)
                drone_reached_target_at_any_time = [False] * DEFAULT_NUM_DRONES
                last_actions_for_history = np.zeros((DEFAULT_NUM_DRONES, 4))
                
                for step in range(max_steps):
                    if approach != "RL" and (step % LLM_UPDATE_INTERVAL_STEPS == 0 or step == 0):
                        drone_states_for_llm = [{'id':i, 'position':episode_env.pos[i].tolist(), 'velocity':episode_env.vel[i].tolist()} for i in range(DEFAULT_NUM_DRONES)]
                        current_command_for_llm, xai_text, ind_expl = task_info["command"], None, [None]*DEFAULT_NUM_DRONES
                        current_full_obs_all_drones = episode_env._computeObs()
                        obs_for_expl_d0 = current_full_obs_all_drones[0] if DEFAULT_NUM_DRONES > 1 and not isinstance(current_full_obs_all_drones, dict) else \
                                          ({k:v[0] for k,v in current_full_obs_all_drones.items()} if isinstance(current_full_obs_all_drones, dict) else current_full_obs_all_drones)

                        if approach=="RLALLMA+SHAP" and shap_explainer and shap_explainer.explainer: 
                            ind_expl[0] = shap_explainer.get_explanation_text(obs_for_expl_d0, 0); xai_text=ind_expl[0]
                        elif approach=="RLALLMA+DiCE" and dice_explainer and dice_explainer.dice_explainer_instance: 
                            ind_expl[0] = dice_explainer.get_counterfactual(obs_for_expl_d0, 0); xai_text=ind_expl[0]
                        
                        drone_state_history.append((step/DEFAULT_CTRL_FREQ, current_command_for_llm, drone_states_for_llm.copy(),
                                                    llm_orchestrated_targets.copy() if approach!="LLM" else initial_mission_targets_for_this_run.copy(), # Use run-specific targets
                                                    last_actions_for_history.copy(), ind_expl.copy()))
                        
                        if approach == "LLM": 
                            llm_orchestrated_actions = get_llm_targets_or_actions(current_command_for_llm, drone_states_for_llm, DEFAULT_NUM_DRONES,
                                llm_orchestrated_actions, initial_mission_targets_for_this_run, last_actions_for_history, xai_text, "actions", task_id)
                        else: 
                            
                            llm_orchestrated_targets = get_llm_targets_or_actions(current_command_for_llm, drone_states_for_llm, DEFAULT_NUM_DRONES, 
                                llm_orchestrated_targets, initial_mission_targets_for_this_run, last_actions_for_history, xai_text, "targets", task_id)
                            episode_env.current_targets = llm_orchestrated_targets
                            print('---')
                            print(llm_orchestrated_targets)
                            print(current_command_for_llm)
                            print(drone_states_for_llm)
                    
                    actions_to_env = np.zeros((DEFAULT_NUM_DRONES, 4))
                    if approach == "LLM": actions_to_env = llm_orchestrated_actions.copy()
                    else: 
                        current_full_obs_all_drones = episode_env._computeObs()
                        for d_idx in range(DEFAULT_NUM_DRONES):
                            obs_for_model = current_full_obs_all_drones[d_idx] if DEFAULT_NUM_DRONES > 1 and not isinstance(current_full_obs_all_drones, dict) else \
                                            ({k:v[d_idx] for k,v in current_full_obs_all_drones.items()} if isinstance(current_full_obs_all_drones, dict) else current_full_obs_all_drones)
                            act_d, _ = model.predict(obs_for_model, deterministic=True)
                            actions_to_env[d_idx, :] = act_d.flatten()
                    last_actions_for_history = actions_to_env.copy()
                    obs_tuple, reward_val, term_flag_from_env, trunc_flag_from_env, info_dict = episode_env.step(actions_to_env)
                    episode_reward_sum += np.mean(reward_val) if isinstance(reward_val,np.ndarray) else reward_val
                    steps_taken+=1
                    
                    # Get current drone positions for tracking
                    current_pos_check = episode_env.pos
                    
                    # Store the current positions for tracking
                    if task_id == "circle_behavior":
                        _update_drone_positions_history(current_pos_check)
                    
                    # Track altitude metrics for all drones
                    for d_idx in range(DEFAULT_NUM_DRONES):
                        # Check and track high altitude
                        if current_pos_check[d_idx, 2] >= 1.3:
                            time_at_high_altitude[d_idx] += 1
                        
                        # Check and track low altitude
                        if current_pos_check[d_idx, 2] <= 0.5 and current_pos_check[d_idx, 2] > 0.05:  # Above crash height
                            time_at_low_altitude[d_idx] += 1
                        
                        # For high path movement task, check if drone has reached high altitude
                        if task_id == "high_path_movement" and current_pos_check[d_idx, 2] >= task_info.get("high_altitude_threshold", 2.9):
                            episode_env._drone_reached_high_altitude[d_idx] = True
                        
                        # For low path movement task, check if drone has reached low altitude
                        if task_id == "low_path_movement" and current_pos_check[d_idx, 2] <= task_info.get("low_altitude_threshold", 0.3):
                            episode_env._drone_reached_low_altitude[d_idx] = True
                    
                    # For follower behavior task, calculate follower accuracy
                    if task_id == "circle_behavior" and len(drone_positions_history) >= 2:
                        prev_positions = drone_positions_history[-2]
                        for i in range(1, DEFAULT_NUM_DRONES):
                            # Calculate how well this follower is tracking the previous drone's last position
                            follower_pos = current_pos_check[i]
                            target_pos = prev_positions[i-1]  # Previous position of the drone ahead
                            follower_accuracy = _calculate_follower_accuracy(follower_pos, target_pos, tolerance=0.1)
                            
                            # Update with exponential moving average
                            episode_env._follower_accuracy[i] = episode_env._follower_accuracy[i] * 0.9 + follower_accuracy * 0.1
                            
                            # Debug output
                            if step % 10 == 0:
                                print(f"Drone {i} follower accuracy: {follower_accuracy:.2f}, EMA: {episode_env._follower_accuracy[i]:.2f}")
                    
                    term_flag_custom = False 
                    
                    # Check for crashes
                    for i in range(DEFAULT_NUM_DRONES):
                        if current_pos_check[i,2] < 0.05: 
                            episode_crash = True
                            term_flag_custom = True
                            break
                    
                    # Check if drones have reached their targets (for non-behavioral tasks)
                    if not task_info.get("is_behavioral", False):
                        for d_idx in range(DEFAULT_NUM_DRONES):
                            if not drone_reached_target_at_any_time[d_idx] and \
                            np.linalg.norm(current_pos_check[d_idx]-initial_mission_targets_for_this_run[d_idx]) <= episode_env.target_tolerance: # Check against run-specific targets
                                drone_reached_target_at_any_time[d_idx] = True
                    
                    # Check for success based on task-specific criteria
                    current_ep_success_status = task_info["check_success"](episode_env, initial_mission_targets_for_this_run, drone_reached_target_at_any_time) # Use run-specific
                    if current_ep_success_status and not episode_crash: 
                        episode_success = True
                        term_flag_custom = True
                    
                    if term_flag_custom:
                        break # or term_flag_from_env or trunc_flag_from_env
                
                if not episode_crash and not episode_success and step==max_steps-1: 
                    print(f"      Ep TIMEOUT. Success: {episode_success}, Crash: {episode_crash}")
                if not episode_success: 
                     episode_success = task_info["check_success"](episode_env, initial_mission_targets_for_this_run, drone_reached_target_at_any_time) # Use run-specific
                     if episode_success and episode_crash: episode_success = False
                
                # Calculate altitude metrics
                total_high_altitude_time = sum(time_at_high_altitude.values())
                total_low_altitude_time = sum(time_at_low_altitude.values())
                high_altitude_pct = 100 * total_high_altitude_time / (steps_taken * DEFAULT_NUM_DRONES) if steps_taken > 0 else 0
                low_altitude_pct = 100 * total_low_altitude_time / (steps_taken * DEFAULT_NUM_DRONES) if steps_taken > 0 else 0
                
                # Calculate follower accuracy metric (if applicable)
                follower_accuracy = 0.0
                if task_id == "circle_behavior":
                    # Average accuracy across all followers
                    follower_accuracy = sum(episode_env._follower_accuracy.values()) / (DEFAULT_NUM_DRONES - 1) * 100  # Convert to percentage
                
                episode_env.close()
                
                # Record all metrics
                results[task_id][approach]["reward"].append(episode_reward_sum)
                results[task_id][approach]["success_rate"].append(1.0 if episode_success and not episode_crash else 0.0)
                results[task_id][approach]["crash_percentage"].append(1.0 if episode_crash else 0.0)
                results[task_id][approach]["time_high_altitude_pct"].append(high_altitude_pct)
                results[task_id][approach]["time_low_altitude_pct"].append(low_altitude_pct)
                results[task_id][approach]["follower_accuracy"].append(follower_accuracy)
                
                print(f"      Ep End ({approach}) -> Reward: {episode_reward_sum:.2f}, Success: {episode_success and not episode_crash}, Crash: {episode_crash}")
                print(f"      Altitude metrics -> High: {high_altitude_pct:.1f}%, Low: {low_altitude_pct:.1f}%")
                if task_id == "circle_behavior":
                    print(f"      Follower task -> Accuracy: {follower_accuracy:.1f}%")

    # Averaging and reporting
    avg_results = {
        task_id: {
            approach: {
                metric: np.nanmean(values) * (1.0 if ("reward" in metric or "accuracy" in metric) else 1.0) 
                for metric, values in metrics_data.items()
            } for approach, metrics_data in task_res.items()
        } for task_id, task_res in results.items()
    }
    
    print("\n" + "="*80 + "\nEVALUATION RESULTS\n" + "="*80)
    
    # Create dataframes for each metric
    metric_dfs = {}
    for metric in ["reward", "success_rate", "crash_percentage", "time_high_altitude_pct", "time_low_altitude_pct", "follower_accuracy"]:
        data_rows = []
        for task_id, task_info_disp in TASKS.items():
            task_name_disp = task_info_disp["name"]
            row = {"Task": task_name_disp}
            for approach_disp in APPROACHES:
                res_ap = avg_results.get(task_id, {}).get(approach_disp, {})
                value = res_ap.get(metric, float('nan'))
                display_value = f"{value:.1f}" if not np.isnan(value) else "-"
                if "percentage" in metric or "rate" in metric or "accuracy" in metric or "pct" in metric:
                    display_value = f"{value:.1f}%" if not np.isnan(value) else "-"
                row[approach_disp] = display_value
            data_rows.append(row)
        metric_dfs[metric] = pd.DataFrame(data_rows if data_rows else [{'Task': 'No tasks/approaches run'}])
    
    # Print tables for each metric
    metric_descriptions = {
        "reward": "AVERAGE REWARD (Higher is Better)",
        "success_rate": "AVERAGE SUCCESS RATE (Higher is Better)",
        "crash_percentage": "AVERAGE CRASH PERCENTAGE (Lower is Better)",
        "time_high_altitude_pct": "TIME SPENT AT HIGH ALTITUDE (>= 2.5m)",
        "time_low_altitude_pct": "TIME SPENT AT LOW ALTITUDE (<= 0.5m)",
        "follower_accuracy": "FOLLOWER BEHAVIOR ACCURACY (Higher is Better)"
    }
    
    for metric, description in metric_descriptions.items():
        df = metric_dfs[metric]
        print(f"\n{description}:")
        print(tabulate(df, headers='keys', tablefmt='grid', missingval='-'))
    
    # Save results and generate plots
    if all(len(df) > 0 and 'No tasks/approaches run' not in df["Task"].values for df in metric_dfs.values()): 
        for metric, df in metric_dfs.items():
            df.to_csv(os.path.join(eval_folder, f"{metric}_results.csv"), index=False)
            
            # Create plot for this metric
            higher_is_better = "crash_percentage" not in metric  # For crash percentage, lower is better
            plt.figure(figsize=(max(12, len(APPROACHES)*2), 7))
            tasks_plot = df["Task"].tolist()
            x_plot = np.arange(len(tasks_plot))
            num_approaches = len(APPROACHES)
            bar_width = 0.8/num_approaches
            
            for i, approach_plot in enumerate(APPROACHES):
                if approach_plot in df.columns:
                    values_plot = [float(str(v).replace('%','')) if str(v)!="-" and str(v).lower()!="nan" else np.nan for v in df[approach_plot]]
                    plt.bar(x_plot+(i-num_approaches/2+0.5)*bar_width, values_plot, bar_width, label=approach_plot)
            
            metric_name_clean = metric.replace('_', ' ').title()
            plt.xlabel('Task', fontsize=12)
            plt.ylabel(metric_name_clean, fontsize=12)
            plt.title(f'Comparison: {metric_name_clean} ({"Higher" if higher_is_better else "Lower"} is Better)', fontsize=14)
            plt.xticks(x_plot, tasks_plot, rotation=30, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(fontsize=10)
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            
            plt.savefig(os.path.join(eval_folder, f"{metric}_comparison.png"))
            plt.close()
        
        print(f"\nResults/plots saved to {eval_folder}")
    else:
        print("\nNo data generated to save or plot.")
    
    return avg_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RLALLMA Evaluation Framework')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--num_episodes', default=NUM_EPISODES, type=int)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--approaches', nargs='+', default=APPROACHES, choices=APPROACHES)
    parser.add_argument('--history_max_entries', default=HISTORY_MAX_ENTRIES, type=int)
    parser.add_argument('--llm_update_interval_sec', type=float, default=1.0)
    args = parser.parse_args()
    DEFAULT_GUI, HISTORY_MAX_ENTRIES = args.gui, args.history_max_entries
    drone_state_history = deque(maxlen=HISTORY_MAX_ENTRIES) 
    LLM_UPDATE_INTERVAL_STEPS = int(args.llm_update_interval_sec * DEFAULT_CTRL_FREQ)
    if LLM_UPDATE_INTERVAL_STEPS < 1: LLM_UPDATE_INTERVAL_STEPS=1; print("[WARNING] LLM update interval too small, set to 1.")
    if args.approaches: APPROACHES=args.approaches; print(f"[INFO] Using approaches: {APPROACHES}")
    run_evaluation(model_path=args.model_path, output_folder=args.output_folder, num_episodes=args.num_episodes)
    print("\nEvaluation complete!")