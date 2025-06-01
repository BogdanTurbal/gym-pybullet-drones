import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import torch
import os

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class VisionMultiTargetAviary(MultiTargetAviary):
    """
    Vision-enhanced adaptive difficulty multi-drone RL environment.
    
    Features:
    - Supports both single and multi-agent setups
    - Uses a Dict observation space with separate kinematic and depth map data
    - Maintains the adaptive difficulty target system
    - Compatible with Stable Baselines 3
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,  # Will be overridden
        act: ActionType = ActionType.RPM,
        # Vision parameters
        vision_attributes: bool = True,
        img_width: int = 64,
        img_height: int = 48,
        include_depth: bool = True,
        include_segmentation: bool = False,
        normalize_depth: bool = True,
        # Other parameters (from MultiTargetAviary)
        episode_length_sec: float = 3.0,
        target_radius_start: float = 0.1,
        target_radius_max: float = 1.0,
        target_radius_increment: float = 0.1,
        target_tolerance: float = 0.01,
        success_threshold: float = 0.9,
        evaluation_window: int = 100,
        collision_distance: float = 0.1,
        lambda_distance: float = 10.0,
        lambda_angle: float = 1.0,
        lambda_1: float = 0.0,    
        lambda_2: float = 0.0,     
        lambda_3: float = 0.0,    
        lambda_4: float = 0.0,    
        lambda_5: float = 0.0,    
        crash_penalty: float = 200.0,
        bounds_penalty: float = 200.0,
        individual_target_reward: float = 200.0,
    ):
        # Store vision-specific parameters
        self.include_depth = include_depth
        self.include_segmentation = include_segmentation
        self.img_width = img_width
        self.img_height = img_height
        self.normalize_depth = normalize_depth
        
        # Vision attributes must be True to get images
        
        
        # Ensure vision attributes are enabled
        if vision_attributes and not obs == ObservationType.RGB:
            # We'll use KIN but with depth map addition
            print(f"[VisionMultiTargetAviary] Using KIN observations with depth maps")
        
        # Initialize the parent class
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,  # Original observation type (overridden in our methods)
            act=act,
            episode_length_sec=episode_length_sec,
            target_radius_start=target_radius_start,
            target_radius_max=target_radius_max,
            target_radius_increment=target_radius_increment,
            target_tolerance=target_tolerance,
            success_threshold=success_threshold,
            evaluation_window=evaluation_window,
            collision_distance=collision_distance,
            lambda_distance=lambda_distance,
            lambda_angle=lambda_angle,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            lambda_5=lambda_5,
            crash_penalty=crash_penalty,
            bounds_penalty=bounds_penalty,
            individual_target_reward=individual_target_reward,
        )
        self.VISION_ATTR = vision_attributes
        
        # Override IMG_RES to use our custom dimensions
        if vision_attributes:
            self.IMG_RES = np.array([img_width, img_height])
            
        # Print setup information
        print(f"[VisionMultiTargetAviary] Initialized with {num_drones} drone(s)")
        print(f"[VisionMultiTargetAviary] Vision enabled: depth={include_depth}, seg={include_segmentation}")
        print(f"[VisionMultiTargetAviary] Image resolution: {img_width}x{img_height}")

    def _observationSpace(self):
        """Create a Dict observation space with separate kinematic and vision components."""
        print('-' * 40)
        print(self.VISION_ATTR, self.include_depth)
        if not self.VISION_ATTR or not self.include_depth:
            # If vision is disabled, use parent's observation space
            return super()._observationSpace()
        
        # Get kinematic observation space from parent
        kin_obs_space = super()._observationSpace()
        
        # For multi-drone case, create a Dict for each drone
        spaces_dict = {}
        
        for i in range(self.NUM_DRONES):
            drone_spaces = {}
            
            # Add kinematic space
            if isinstance(kin_obs_space, spaces.Dict):
                # If parent already uses Dict space, extract drone's part
                drone_spaces["kinematics"] = kin_obs_space[str(i)] if str(i) in kin_obs_space.spaces else kin_obs_space
            else:
                # Otherwise, extract from Box space
                drone_spaces["kinematics"] = spaces.Box(
                    low=kin_obs_space.low[i],
                    high=kin_obs_space.high[i],
                    dtype=np.float32
                )
            
            # Add depth vision space
            if self.include_depth:
                drone_spaces["depth"] = spaces.Box(
                    low=0.01,
                    high=1000.0,
                    shape=(self.img_height, self.img_width),
                    dtype=np.float32
                )
            
            # Add segmentation vision space
            if self.include_segmentation:
                drone_spaces["segmentation"] = spaces.Box(
                    low=0,
                    high=100,
                    shape=(self.img_height, self.img_width),
                    dtype=np.int32
                )
            
            # Add RGB space if needed
            if hasattr(self, 'rgb') and self.rgb is not None:
                drone_spaces["rgb"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.img_height, self.img_width, 4),
                    dtype=np.uint8
                )
            
            # Add this drone's spaces to the dict
            spaces_dict[str(i)] = spaces.Dict(drone_spaces)
        
        return spaces.Dict(spaces_dict)

    def _computeObs(self):
        """Compute observations as a dictionary with separate kinematic and depth components."""
        if not self.VISION_ATTR or not self.include_depth:
            # If vision is disabled, use parent's observation method
            print('FUCK ' * 40)
            return super()._computeObs()
        
        # Get kinematic observations from parent
        kin_obs = super()._computeObs()
        
        # Ensure we have vision data for all drones
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
            for i in range(self.NUM_DRONES):
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(
                    i,
                    segmentation=self.include_segmentation
                )
                
                # Record images if enabled
                if self.RECORD:
                    if self.include_depth:
                        self._exportImage(
                            img_type=ImageType.DEP,
                            img_input=self.dep[i],
                            path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                            frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                        )
        
        # Create the dictionary observation
        obs_dict = {}
        
        for i in range(self.NUM_DRONES):
            drone_obs = {}
            
            # Add kinematic observations
            if isinstance(kin_obs, dict):
                # If parent already uses Dict, extract drone's part
                drone_obs["kinematics"] = kin_obs[str(i)] if str(i) in kin_obs else kin_obs
            else:
                # Otherwise, extract from ndarray
                drone_obs["kinematics"] = kin_obs[i]
            
            # Add depth vision
            if self.include_depth:
                depth_map = self.dep[i].copy()
                
                # Optionally normalize depth
                if self.normalize_depth:
                    depth_map = self._normalizeDepthMap(depth_map)
                
                drone_obs["depth"] = depth_map
            
            # Add segmentation if enabled
            if self.include_segmentation:
                drone_obs["segmentation"] = self.seg[i]
            
            # Add RGB if available
            if hasattr(self, 'rgb') and self.rgb is not None:
                drone_obs["rgb"] = self.rgb[i]
            
            # Add this drone's observations to the dict
            obs_dict[str(i)] = drone_obs
        
        return obs_dict

    def _normalizeDepthMap(self, depth_map):
        """Normalize depth map to improve learning signal."""
        # Clip to reasonable range (e.g., 0.01 to 10 meters)
        clipped = np.clip(depth_map, 0.01, 10.0)
        
        # Apply normalization (1.0 is close, 0.0 is far)
        # This improves the learning signal by emphasizing nearby objects
        normalized = 1.0 - (clipped - 0.01) / (10.0 - 0.01)
        
        return normalized

    def reset(self, seed=None, options=None):
        """Reset environment and ensure depth maps are initialized."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Ensure depth maps are updated immediately after reset
        if self.VISION_ATTR and self.include_depth:
            for i in range(self.NUM_DRONES):
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(
                    i,
                    segmentation=self.include_segmentation
                )
                
            # Recompute observations with depth maps
            obs = self._computeObs()
        
        return obs, info

    def step(self, action):
        """Execute one simulation step and return Dict observations."""
        # Call parent step method
        _, reward, terminated, truncated, info = super().step(action)
        
        # Compute observations (which will be Dict format)
        obs = self._computeObs()
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        # Use the parent's render method
        super().render()


# Usage example
if __name__ == "__main__":
    # Test the environment
    env = VisionMultiTargetAviary(
        num_drones=1,
        gui=False,
        record=False,
        obs=ObservationType.RGB,
        act=ActionType.RPM,
        vision_attributes=True, 
        include_depth=True,
        include_segmentation=False,
        img_width=64,
        img_height=48,
        episode_length_sec=5.0,
        ctrl_freq=24
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test a reset
    obs, info = env.reset()
    print(f"Observation structure: {obs}")
    
    # Run a few random steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if i % 10 == 0:
            # Print structure of first drone's observation
            if isinstance(obs, dict) and '0' in obs:
                print(f"Step {i}: Kinematics shape={obs['0']['kinematics'].shape}, " + 
                      f"Depth shape={obs['0']['depth'].shape}, Reward={reward}")
        
        if done:
            print(f"Episode ended after {i+1} steps")
            break
    
    env.close()