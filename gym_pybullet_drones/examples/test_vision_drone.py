#!/usr/bin/env python3
"""
test_vision_drone.py - Script to test the VisionMultiTargetAviary environment

This script demonstrates how to use the VisionMultiTargetAviary environment
with dictionary observations and visualize the depth maps.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.envs.VisionMultiTargetAviary import VisionMultiTargetAviary

# Configure matplotlib for interactive plotting
plt.ion()

def visualize_depth_map(depth_map, ax=None):
    """Visualize a depth map"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Depth map is already 2D in the Dict observation
    depth_2d = depth_map
    
    # Clip depth values for better visualization
    depth_2d_clipped = np.clip(depth_2d, 0.01, 10.0)
    
    # Normalize to [0, 1] range for visualization
    depth_normalized = (depth_2d_clipped - 0.01) / (10.0 - 0.01)
    
    # Display the depth map
    im = ax.imshow(depth_normalized, cmap='viridis')
    ax.set_title('Depth Map (lighter = closer)')
    
    return im

def test_environment(render=True, num_episodes=3, num_steps_per_episode=100):
    """Test the VisionMultiTargetAviary environment"""
    # Create environment
    env = VisionMultiTargetAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=render,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        vision_attributes=True,
        img_width=64,
        img_height=48,
        include_depth=True,
        include_segmentation=False,
        episode_length_sec=5.0,
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Set up visualization
    if render:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plt.tight_layout()
    
    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1}")
        print(f"Target position: {info['current_targets'][0]}")
        print(f"Observation structure:")
        print(f"  - Dict with {len(obs)} drone(s)")
        print(obs)
        print(f"  - Drone 0 has keys: {list(obs['0'].keys())}")
        print(f"  - Kinematics shape: {obs['0']['kinematics'].shape}")
        print(f"  - Depth map shape: {obs['0']['depth'].shape}")
        
        # Extract observations for the first drone
        drone_obs = obs['0']
        kin_obs = drone_obs['kinematics']
        depth_map = drone_obs['depth']
        
        # Print position and target info
        drone_pos = kin_obs[:3]  # First 3 values are position
        
        # Target information - depends on your implementation
        # If target is part of kinematics, it might be at a specific index
        # For this example, let's assume it's in the info dict
        target_pos = info['current_targets'][0]
        
        print(f"Drone position: {drone_pos}")
        print(f"Target position: {target_pos}")
        
        if render:
            # Initialize plots
            pos_line, = ax1.plot([0], [0], 'ro', markersize=10, label='Drone')
            target_line, = ax1.plot([0], [0], 'go', markersize=10, label='Target')
            ax1.set_xlim(-3, 3)
            ax1.set_ylim(-3, 3)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Drone and Target Positions (Top View)')
            ax1.legend()
            ax1.grid(True)
            
            depth_im = visualize_depth_map(depth_map, ax2)
            plt.pause(0.01)
        
        cumulative_reward = 0
        
        # Run episode
        for step in range(num_steps_per_episode):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            
            # Extract observations for the first drone
            drone_obs = obs['0']
            kin_obs = drone_obs['kinematics']
            depth_map = drone_obs['depth']
            
            # Update visualization
            if render:
                # Get drone position
                drone_pos = kin_obs[:3]
                target_pos = info['current_targets'][0]
                
                # Update position plot
                pos_line.set_data([drone_pos[0]], [drone_pos[1]])
                target_line.set_data([target_pos[0]], [target_pos[1]])
                
                # Update depth map
                depth_im.set_array(depth_map)
                
                plt.pause(0.01)
            
            # Print progress
            if step % 10 == 0:
                min_dist = info.get('min_distance_to_target', 0)
                print(f"Step {step:3d} | Distance: {min_dist:.3f} | Reward: {reward:.1f}")
            
            if render:
                time.sleep(0.01)  # Slow down for visualization
            
            if done:
                success = info.get('episode_success', False)
                print(f"Episode ended at step {step+1}: {'SUCCESS' if success else 'FAILURE'}")
                print(f"Cumulative reward: {cumulative_reward:.2f}")
                break
    
    env.close()
    
    if render:
        # Keep plot open at the end
        plt.ioff()
        plt.show()


def inspect_observation_space():
    """Create environment and inspect the observation space structure"""
    env = VisionMultiTargetAviary(
        num_drones=1,
        vision_attributes=True,
        include_depth=True,
        img_width=64,
        img_height=48,
    )
    
    print("\n=== Observation Space Structure ===")
    obs_space = env.observation_space
    
    # Check if it's a Dict
    if isinstance(obs_space, gym.spaces.Dict):
        print(f"Top level: Dict with {len(obs_space.spaces)} items")
        
        # Inspect first drone
        if '0' in obs_space.spaces:
            drone_space = obs_space.spaces['0']
            
            print("\nFirst drone space:")
            if isinstance(drone_space, gym.spaces.Dict):
                print(f"  Dict with keys: {list(drone_space.spaces.keys())}")
                
                # Print details of each key
                for key, space in drone_space.spaces.items():
                    print(f"  - {key}: {type(space).__name__} with shape {space.shape}")
            else:
                print(f"  {type(drone_space).__name__} with shape {drone_space.shape}")
    else:
        print(f"Not a Dict: {type(obs_space).__name__} with shape {obs_space.shape}")
    
    # Get a sample observation
    obs, _ = env.reset()
    
    print("\n=== Sample Observation Structure ===")
    if isinstance(obs, dict):
        print(f"Top level: Dict with {len(obs)} items")
        
        # Inspect first drone observation
        if '0' in obs:
            drone_obs = obs['0']
            
            print("\nFirst drone observation:")
            if isinstance(drone_obs, dict):
                print(f"  Dict with keys: {list(drone_obs.keys())}")
                
                # Print details of each key
                for key, value in drone_obs.items():
                    if hasattr(value, 'shape'):
                        print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  - {key}: {type(value)}")
            else:
                print(f"  {type(drone_obs).__name__}")
    else:
        print(f"Not a Dict: {type(obs).__name__}")
    
    env.close()


def main():
    print("Testing VisionMultiTargetAviary environment with Dict observations...")
    
    # First inspect the observation space structure
    inspect_observation_space()
    
    # Then run the test
    test_environment(render=False, num_episodes=3, num_steps_per_episode=200)


if __name__ == "__main__":
    main()