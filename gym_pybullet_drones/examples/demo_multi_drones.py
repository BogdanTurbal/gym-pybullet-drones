#!/usr/bin/env python3
"""
demo_multi_drones.py: Load pre-trained multi-drone swarm model and run demonstration
"""
import os
import time
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Default settings
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'demo_results'
DEFAULT_OBS           = ObservationType('kin')   # 'kin' or 'rgb'
DEFAULT_ACT           = ActionType('one_d_rpm')  # 'rpm','pid','vel','one_d_rpm','one_d_pid'
DEFAULT_DRONES        = 4
DEFAULT_DURATION_SEC  = 5.0  # seconds per target phase


def create_target_sequence(num_drones=4, scale=0.5):
    """Create a challenging but achievable target sequence"""
    
    if num_drones == 4:
        # 4-drone formations
        targets = np.array([
            # Phase 0: Square formation
            [[ scale,  scale, 0.5], [-scale,  scale, 0.5], 
             [-scale, -scale, 0.5], [ scale, -scale, 0.5]],
            
            # Phase 1: Rotate clockwise
            [[-scale,  scale, 0.5], [-scale, -scale, 0.5], 
             [ scale, -scale, 0.5], [ scale,  scale, 0.5]],
            
            # Phase 2: Diamond formation (higher altitude)
            [[ 0.0,  scale*1.2, 2.0], [-scale*1.2,  0.0, 2.0], 
             [ 0.0, -scale*1.2, 2.0], [ scale*1.2,  0.0, 2.0]],
            
            # Phase 3: Tight formation at center
            [[ 0.3,  0.3, 1.8], [-0.3,  0.3, 1.8], 
             [-0.3, -0.3, 1.8], [ 0.3, -0.3, 1.8]],
             
            # Phase 4: Line formation
            [[ 0.0,  scale, 0.5], [ 0.0,  scale/3, 0.5], 
             [ 0.0, -scale/3, 0.5], [ 0.0, -scale, 0.5]]
        ])
    else:
        # For other numbers of drones, create circular formations
        targets = []
        n_phases = 5
        for phase in range(n_phases):
            phase_targets = []
            radius = scale * (1.0 + 0.2 * phase)  # Varying radius
            height = 0.5 + 0.3 * phase  # Varying height
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones + phase * np.pi / 4  # Rotate each phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                phase_targets.append([x, y, height])
            targets.append(phase_targets)
        targets = np.array(targets)
    
    return targets.astype(np.float32)


def run_demonstration(model_path, output_folder, gui, record_video, plot, num_episodes=1):
    """
    Load pre-trained model and run demonstration
    
    Args:
        model_path: Path to the saved PPO model (.zip file)
        output_folder: Folder to save demonstration results
        gui: Whether to show PyBullet GUI
        record_video: Whether to record video
        plot: Whether to generate plots
        num_episodes: Number of demonstration episodes to run
    """
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"[INFO] Loading model from: {model_path}")
    
    # Prepare output directory
    os.makedirs(output_folder, exist_ok=True)

    # Get control frequency
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    # Create target sequence
    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=1.2)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] Target sequence shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Total episode length: {len(target_sequence) * steps_per_target} steps")
    print(f"[INFO] Control frequency: {freq} Hz")

    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Create evaluation environment
    eval_env = MultiTargetAviary(
        num_drones=DEFAULT_DRONES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        target_sequence=target_sequence,
        steps_per_target=steps_per_target,
        gui=False,
        record=False
    )

    # Print environment info
    print('[INFO] Action space:', eval_env.action_space)
    print('[INFO] Observation space:', eval_env.observation_space)

    # Run evaluation
    print(f"[INFO] Running evaluation with {num_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=num_episodes, deterministic=True
    )
    print(f"[RESULTS] Evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run demonstration episodes
    for episode in range(num_episodes):
        print(f"\n[INFO] Running demonstration episode {episode + 1}/{num_episodes}...")
        
        # Create demonstration environment
        demo_env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=gui,
            record=record_video
        )
        
        # Optional: Create logger for visualization
        logger = None
        if plot:
            logger = Logger(
                logging_freq_hz=freq, 
                num_drones=DEFAULT_DRONES, 
                output_folder=output_folder
            )
        
        obs, info = demo_env.reset(seed=42 + episode)
        start = time.time()
        episode_reward = 0
        demo_metrics = {
            'phase_rewards': [],
            'targets_reached_per_phase': [],
            'distances': [],
            'actions': [],
            'positions': []
        }
        
        max_steps = len(target_sequence) * steps_per_target
        print(f"[INFO] Running for {max_steps} steps...")
        
        for i in range(max_steps + 100):  # Add buffer steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(action)
            episode_reward += reward
            
            # Collect demo metrics
            if 'phase' in info:
                current_phase = info['phase']
                if len(demo_metrics['phase_rewards']) <= current_phase:
                    demo_metrics['phase_rewards'].extend([0] * (current_phase - len(demo_metrics['phase_rewards']) + 1))
                demo_metrics['phase_rewards'][current_phase] += reward
            
            if 'distance_to_targets' in info:
                demo_metrics['distances'].append(np.mean(info['distance_to_targets']))
            
            # Store actions and positions for analysis
            demo_metrics['actions'].append(action.copy() if hasattr(action, 'copy') else action)
            
            if gui:
                demo_env.render()
                sync(i, start, demo_env.CTRL_TIMESTEP)
            
            # Log progress every 100 steps
            if i % 100 == 0:
                phase = info.get('phase', -1)
                targets_reached = np.sum(info.get('targets_reached', []))
                dist = np.mean(info.get('distance_to_targets', [0]))
                print(f"Step {i:4d} | Phase {phase} | Targets: {targets_reached} | Distance: {dist:.3f} | Reward: {reward:.3f}")
            
            # Optional logging for plotting
            if plot and logger and DEFAULT_OBS == ObservationType.KIN:
                obs_arr = obs.squeeze() if obs.ndim > 1 else obs
                act_arr = action.squeeze() if action.ndim > 1 else action
                
                for d in range(DEFAULT_DRONES):
                    # Extract state information
                    if obs_arr.ndim > 1:
                        drone_obs = obs_arr[d]
                    else:
                        drone_obs = obs_arr
                    
                    # Create state vector for logging
                    state = np.hstack([
                        drone_obs[0:3],     # position
                        np.zeros(4),        # quaternion placeholder
                        drone_obs[3:15] if len(drone_obs) > 15 else np.pad(drone_obs[3:], (0, 12-len(drone_obs[3:]))),  # velocity, etc.
                        act_arr[d] if act_arr.ndim > 0 else [act_arr]
                    ])
                    logger.log(drone=d, timestamp=i/freq, state=state, control=np.zeros(12))
            
            if done:
                print(f"[INFO] Episode {episode + 1} finished at step {i}")
                break
        
        print(f"[RESULTS] Episode {episode + 1} total reward: {episode_reward:.2f}")
        
        # Print episode summary
        if demo_metrics['distances']:
            print(f"[METRICS] Final distance to targets: {demo_metrics['distances'][-1]:.3f}")
            print(f"[METRICS] Mean distance during episode: {np.mean(demo_metrics['distances']):.3f}")
            print(f"[METRICS] Min distance achieved: {np.min(demo_metrics['distances']):.3f}")
        
        # if demo_metrics['phase_rewards']:
        #     print(f"[METRICS] Rewards per phase: {[f'{r:.2f}' for r in demo_metrics['phase_rewards']}}")
        
        demo_env.close()
        
        # Generate plots for this episode
        if plot and demo_metrics['distances']:
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Demonstration Episode {episode + 1} Analysis')
                
                # Distance plot
                axes[0, 0].plot(demo_metrics['distances'])
                axes[0, 0].set_title('Distance to Targets Over Time')
                axes[0, 0].set_xlabel('Steps')
                axes[0, 0].set_ylabel('Mean Distance')
                axes[0, 0].grid(True)
                
                # Phase rewards plot
                if demo_metrics['phase_rewards']:
                    axes[0, 1].bar(range(len(demo_metrics['phase_rewards'])), demo_metrics['phase_rewards'])
                    axes[0, 1].set_title('Rewards per Phase')
                    axes[0, 1].set_xlabel('Phase')
                    axes[0, 1].set_ylabel('Total Reward')
                    axes[0, 1].grid(True)
                
                # Distance histogram
                axes[1, 0].hist(demo_metrics['distances'], bins=30, alpha=0.7)
                axes[1, 0].set_title('Distance Distribution')
                axes[1, 0].set_xlabel('Distance to Targets')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
                
                # Running average of distance
                window_size = min(50, len(demo_metrics['distances']) // 10)
                if window_size > 1:
                    running_avg = np.convolve(demo_metrics['distances'], 
                                            np.ones(window_size)/window_size, mode='valid')
                    axes[1, 1].plot(running_avg)
                    axes[1, 1].set_title(f'Running Average Distance (window={window_size})')
                    axes[1, 1].set_xlabel('Steps')
                    axes[1, 1].set_ylabel('Average Distance')
                    axes[1, 1].grid(True)
                
                plt.tight_layout()
                plot_path = os.path.join(output_folder, f'demo_episode_{episode + 1}_analysis.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"[INFO] Saved analysis plot: {plot_path}")
                plt.close()
                
            except ImportError:
                print("[WARNING] Matplotlib not available, skipping plots")
            except Exception as e:
                print(f"[WARNING] Failed to generate plots: {e}")
        
        # Generate trajectory plots using logger
        if plot and logger and DEFAULT_OBS == ObservationType.KIN:
            try:
                logger.plot()
                print(f"[INFO] Generated trajectory plots in: {output_folder}")
            except Exception as e:
                print(f"[WARNING] Failed to generate trajectory plots: {e}")

    print(f"\n[INFO] Demonstration completed!")
    print(f"[INFO] Results saved to: {output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstration of pre-trained multi-drone swarm model')
    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to the trained PPO model (.zip file)')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, 
                        help='Use PyBullet GUI during demonstration')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, 
                        help='Record video of demonstration')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='Results folder for demonstration outputs')
    parser.add_argument('--plot', default=True, type=str2bool, 
                        help='Generate analysis plots')
    parser.add_argument('--num_episodes', default=1, type=int,
                        help='Number of demonstration episodes to run')
    
    args = parser.parse_args()

    # Validate arguments
    if not args.model_path.endswith('.zip'):
        print("[WARNING] Model path should end with .zip extension")
    
    try:
        run_demonstration(
            args.model_path,
            args.output_folder,
            args.gui,
            args.record_video,
            args.plot,
            args.num_episodes
        )
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        exit(1)