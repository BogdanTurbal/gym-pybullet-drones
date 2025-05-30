#!/usr/bin/env python3
"""
demo_adaptive_difficulty.py: Load pre-trained adaptive difficulty model and run demonstration
Updated to work with adaptive difficulty single-drone system
Supports TD3, SAC, and PPO models with adaptive target radius
"""
import os
import time
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from collections import deque

from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

# Import your custom extractors (REQUIRED for model loading)
from multi_agent_extractors_td3 import (
    MultiAgentMatrixExtractor,
    MultiAgentSelfAttentionExtractor,
    MultiAgentMeanPoolExtractor,
    create_multiagent_model
)

# Import your updated environment
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Updated settings to match training script
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'demo_results'
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('rpm')
DEFAULT_DRONES        = 1
DEFAULT_DURATION_SEC  = 3.0


class AdaptiveDifficultyMetricsCollector:
    """Metrics collection for adaptive difficulty demonstration analysis"""
    
    def __init__(self, num_drones):
        self.num_drones = num_drones
        self.reset()
    
    def reset(self):
        """Reset all metrics for new episode"""
        self.episode_data = {
            'rewards': [],
            'distances': [],
            'actions': [],
            'positions': [],
            'targets': [],
            'target_radius_history': [],
            'success_rate_history': [],
            'episode_success': False,
            'step_info': []
        }
        self.total_reward = 0
        self.steps = 0
        self.final_target_radius = 0.0
        self.final_success_rate = 0.0
    
    def update(self, action, obs, reward, info, step):
        """Update metrics with current step data"""
        self.steps += 1
        self.total_reward += reward
        
        # Basic metrics
        self.episode_data['rewards'].append(reward)
        self.episode_data['actions'].append(action.copy() if hasattr(action, 'copy') else np.array(action))
        
        # Position and target tracking
        if obs is not None and len(obs) > 0:
            # Extract position from observation
            if obs.ndim > 1:
                position = obs[0, 0:3]  # First drone position
            else:
                position = obs[0:3]
            self.episode_data['positions'].append(position.copy())
        
        # Adaptive difficulty metrics
        if 'target_radius' in info:
            self.episode_data['target_radius_history'].append(info['target_radius'])
            self.final_target_radius = info['target_radius']
        
        if 'success_rate_last_100' in info:
            self.episode_data['success_rate_history'].append(info['success_rate_last_100'])
            self.final_success_rate = info['success_rate_last_100']
        
        if 'current_targets' in info:
            self.episode_data['targets'].append(info['current_targets'].copy())
        
        if 'min_distance_to_target' in info:
            self.episode_data['distances'].append(info['min_distance_to_target'])
        
        if 'episode_success' in info:
            self.episode_data['episode_success'] = info['episode_success']
        
        # Store step info for detailed analysis
        self.episode_data['step_info'].append({
            'step': step,
            'reward': reward,
            'distance': info.get('min_distance_to_target', 0),
            'target_radius': info.get('target_radius', 0),
            'success_rate': info.get('success_rate_last_100', 0),
            'episode_success': info.get('episode_success', False),
        })
    
    def get_summary(self):
        """Get comprehensive episode summary"""
        if not self.episode_data['rewards']:
            return {}
        
        summary = {
            'total_reward': self.total_reward,
            'total_steps': self.steps,
            'mean_reward_per_step': self.total_reward / self.steps if self.steps > 0 else 0,
            'episode_success': self.episode_data['episode_success'],
            'final_target_radius': self.final_target_radius,
            'final_success_rate': self.final_success_rate,
        }
        
        # Distance statistics
        if self.episode_data['distances']:
            distances = self.episode_data['distances']
            summary.update({
                'distance_stats': {
                    'final': distances[-1] if distances else 0,
                    'mean': np.mean(distances),
                    'min_achieved': np.min(distances),
                    'max': np.max(distances),
                    'std': np.std(distances),
                    'improvement': distances[0] - distances[-1] if len(distances) > 1 else 0
                }
            })
        
        # Adaptive difficulty progression
        if self.episode_data['target_radius_history']:
            radius_history = self.episode_data['target_radius_history']
            summary['adaptive_stats'] = {
                'initial_radius': radius_history[0] if radius_history else 0,
                'final_radius': radius_history[-1] if radius_history else 0,
                'radius_changes': len(set(radius_history)) - 1,
                'max_radius_achieved': max(radius_history) if radius_history else 0,
            }
        
        # Success rate progression
        if self.episode_data['success_rate_history']:
            success_history = self.episode_data['success_rate_history']
            summary['success_progression'] = {
                'initial_rate': success_history[0] if success_history else 0,
                'final_rate': success_history[-1] if success_history else 0,
                'max_rate_achieved': max(success_history) if success_history else 0,
                'mean_rate': np.mean(success_history) if success_history else 0,
            }
        
        return summary


def detect_algorithm_from_model(model_path):
    """Detect which algorithm was used for training from model file"""
    try:
        # Try to load as different algorithms to detect type
        algorithms = ['td3', 'sac', 'ppo']
        
        for algo in algorithms:
            try:
                if algo == 'td3':
                    model = TD3.load(model_path, print_system_info=False)
                elif algo == 'sac':
                    model = SAC.load(model_path, print_system_info=False)
                elif algo == 'ppo':
                    model = PPO.load(model_path, print_system_info=False)
                
                print(f"[INFO] Detected algorithm: {algo.upper()}")
                return algo, model
                
            except Exception:
                continue
        
        # If all fail, default to PPO
        print("[WARNING] Could not detect algorithm, defaulting to PPO")
        model = PPO.load(model_path)
        return 'ppo', model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model with any algorithm: {e}")


def create_enhanced_adaptive_plots(metrics_collector, output_folder, episode_num):
    """Create comprehensive analysis plots for adaptive difficulty system"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        episode_data = metrics_collector.episode_data
        summary = metrics_collector.get_summary()
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Reward over time
        ax1 = fig.add_subplot(gs[0, 0])
        if episode_data['rewards']:
            ax1.plot(episode_data['rewards'], alpha=0.7, label='Step Reward', color='blue')
            # Add running average
            window = min(50, len(episode_data['rewards']) // 10)
            if window > 1:
                running_avg = np.convolve(episode_data['rewards'], np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(episode_data['rewards'])), running_avg, 
                        color='red', linewidth=2, label=f'Running Avg ({window})')
            ax1.set_title('Reward Over Time')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True)
        
        # Plot 2: Distance to target over time
        ax2 = fig.add_subplot(gs[0, 1])
        if episode_data['distances']:
            distances = episode_data['distances']
            steps = range(len(distances))
            ax2.plot(steps, distances, label='Distance to Target', linewidth=2, color='green')
            
            # Add target tolerance line
            target_tolerance = 0.05  # From environment
            ax2.axhline(y=target_tolerance, color='red', linestyle='--', 
                       label=f'Target Tolerance ({target_tolerance})')
            
            ax2.set_title('Distance to Target Over Time')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Distance')
            ax2.legend()
            ax2.grid(True)
        
        # Plot 3: Target radius progression
        ax3 = fig.add_subplot(gs[0, 2])
        if episode_data['target_radius_history']:
            radius_history = episode_data['target_radius_history']
            ax3.plot(radius_history, linewidth=2, color='purple', label='Target Radius')
            ax3.set_title('Adaptive Target Radius')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Target Radius (m)')
            ax3.legend()
            ax3.grid(True)
        
        # Plot 4: Success rate progression
        ax4 = fig.add_subplot(gs[0, 3])
        if episode_data['success_rate_history']:
            success_history = episode_data['success_rate_history']
            ax4.plot(success_history, linewidth=2, color='orange', label='Success Rate')
            ax4.axhline(y=0.9, color='red', linestyle='--', label='Threshold (0.9)')
            ax4.set_title('Success Rate Progression')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Success Rate')
            ax4.set_ylim(0, 1)
            ax4.legend()
            ax4.grid(True)
        
        # Plot 5: 3D trajectory
        ax5 = fig.add_subplot(gs[1, 0], projection='3d')
        if episode_data['positions'] and episode_data['targets']:
            positions = np.array(episode_data['positions'])
            targets = np.array(episode_data['targets'])
            
            # Plot trajectory
            ax5.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    'b-', alpha=0.7, linewidth=2, label='Drone Path')
            
            # Plot start and end positions
            ax5.scatter(*positions[0], color='green', s=100, label='Start')
            ax5.scatter(*positions[-1], color='red', s=100, label='End')
            
            # Plot targets (they change, so plot first and last)
            if len(targets) > 0:
                ax5.scatter(*targets[0][0], color='orange', s=100, marker='*', label='Initial Target')
                if len(targets) > 1:
                    ax5.scatter(*targets[-1][0], color='purple', s=100, marker='*', label='Final Target')
            
            ax5.set_title('3D Trajectory')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_zlabel('Z')
            ax5.legend()
        
        # Plot 6: Distance distribution
        ax6 = fig.add_subplot(gs[1, 1])
        if episode_data['distances']:
            distances = episode_data['distances']
            ax6.hist(distances, bins=30, alpha=0.7, edgecolor='black', color='green')
            ax6.axvline(np.mean(distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(distances):.3f}')
            ax6.axvline(0.05, color='orange', linestyle='--', label='Target Tolerance')
            ax6.set_title('Distance Distribution')
            ax6.set_xlabel('Distance to Target')
            ax6.set_ylabel('Frequency')
            ax6.legend()
            ax6.grid(True)
        
        # Plot 7: Action analysis
        ax7 = fig.add_subplot(gs[1, 2])
        if episode_data['actions']:
            actions = np.array(episode_data['actions'])
            if actions.ndim > 1:
                # Plot action variance over time
                action_var = np.var(actions, axis=1)
                ax7.plot(action_var, alpha=0.7, color='brown')
                ax7.set_title('Action Variance Over Time')
                ax7.set_xlabel('Steps')
                ax7.set_ylabel('Action Variance')
                ax7.grid(True)
        
        # Plot 8: Reward distribution
        ax8 = fig.add_subplot(gs[1, 3])
        if episode_data['rewards']:
            rewards = episode_data['rewards']
            unique_rewards, counts = np.unique(rewards, return_counts=True)
            
            # Create bar plot for discrete rewards
            ax8.bar(unique_rewards, counts, alpha=0.7, edgecolor='black')
            ax8.set_title('Reward Distribution')
            ax8.set_xlabel('Reward Value')
            ax8.set_ylabel('Frequency')
            ax8.grid(True, axis='y')
            
            # Add labels on bars
            for reward, count in zip(unique_rewards, counts):
                ax8.text(reward, count + max(counts)*0.01, str(count), 
                        ha='center', va='bottom')
        
        # Plot 9: Performance summary (text)
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.axis('off')
        
        # Create summary text
        success_str = "SUCCESS" if summary.get('episode_success', False) else "FAILURE"
        summary_text = f"""
        ADAPTIVE DIFFICULTY DEMONSTRATION SUMMARY - Episode {episode_num}
        
        Episode Outcome: {success_str}
        • Total Reward: {summary.get('total_reward', 0):.2f}
        • Total Steps: {summary.get('total_steps', 0)}
        • Mean Reward/Step: {summary.get('mean_reward_per_step', 0):.4f}
        
        Distance Performance:
        • Final Distance: {summary.get('distance_stats', {}).get('final', 0):.3f}
        • Mean Distance: {summary.get('distance_stats', {}).get('mean', 0):.3f}
        • Best Distance: {summary.get('distance_stats', {}).get('min_achieved', 0):.3f}
        • Distance Improvement: {summary.get('distance_stats', {}).get('improvement', 0):.3f}
        
        Adaptive Difficulty:
        • Final Target Radius: {summary.get('final_target_radius', 0):.3f}m
        • Final Success Rate: {summary.get('final_success_rate', 0):.3f}
        • Max Radius Achieved: {summary.get('adaptive_stats', {}).get('max_radius_achieved', 0):.3f}m
        • Radius Changes: {summary.get('adaptive_stats', {}).get('radius_changes', 0)}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Plot 10: Performance gauges
        ax10 = fig.add_subplot(gs[2, 2:])
        
        # Calculate performance scores
        distance_score = max(0, 100 * (1 - summary.get('distance_stats', {}).get('mean', 1) / 1.0))  # Normalize by max distance
        success_score = 100 if summary.get('episode_success', False) else 0
        difficulty_score = summary.get('final_target_radius', 0) / 1.0 * 100  # Normalize by max radius
        overall_score = (distance_score + success_score + difficulty_score) / 3
        
        # Create gauge plot
        categories = ['Distance\nPerformance', 'Episode\nSuccess', 'Difficulty\nLevel', 'Overall\nScore']
        scores = [distance_score, success_score, difficulty_score, overall_score]
        colors = ['green' if s > 70 else 'orange' if s > 40 else 'red' for s in scores]
        
        bars = ax10.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
        ax10.set_ylim(0, 100)
        ax10.set_title('Performance Metrics')
        ax10.set_ylabel('Score (0-100)')
        ax10.grid(True, axis='y')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Adaptive Difficulty Drone Demonstration Analysis - Episode {episode_num}', 
                     fontsize=16, fontweight='bold')
        
        # Save plot
        plot_path = os.path.join(output_folder, f'demo_episode_{episode_num}_adaptive_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved adaptive difficulty analysis plot: {plot_path}")
        plt.close()
        
        return plot_path
        
    except ImportError:
        print("[WARNING] Matplotlib not available, skipping enhanced plots")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to generate enhanced plots: {e}")
        return None


def run_demonstration(model_path, output_folder, gui, record_video, plot, num_episodes=1, 
                     detailed_analysis=True, save_metrics=True, vec_normalize_path=None):
    """
    Load pre-trained adaptive difficulty model and run demonstration
    
    Args:
        model_path: Path to the saved model (.zip file)
        output_folder: Folder to save demonstration results
        gui: Whether to show PyBullet GUI
        record_video: Whether to record video
        plot: Whether to generate plots
        num_episodes: Number of demonstration episodes to run
        detailed_analysis: Whether to perform detailed analysis
        save_metrics: Whether to save metrics to files
        vec_normalize_path: Path to VecNormalize stats if model was trained with normalization
    """
    
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.zip'):
        print("[WARNING] Model path should typically end with .zip extension")
    
    print(f"[INFO] Loading adaptive difficulty model from: {model_path}")
    print(f"[INFO] Running {num_episodes} demonstration episode(s)")
    print(f"[INFO] Using action type: {DEFAULT_ACT}")
    print(f"[INFO] Multi-agent extractors imported and available")
    
    # Prepare output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    demo_folder = os.path.join(output_folder, f'adaptive_demo_{timestamp}')
    os.makedirs(demo_folder, exist_ok=True)

    # Get control frequency
    try:
        dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        freq = int(dummy_env.CTRL_FREQ)
        dummy_env.close()
    except Exception as e:
        print(f"[ERROR] Failed to create dummy environment: {e}")
        return

    print(f"[INFO] Episode length: {DEFAULT_DURATION_SEC:.1f} seconds")
    print(f"[INFO] Control frequency: {freq} Hz")
    print(f"[INFO] Max steps per episode: {int(DEFAULT_DURATION_SEC * freq)}")

    # Load the trained model with automatic algorithm detection
    try:
        algorithm, model = detect_algorithm_from_model(model_path)
        print(f"[INFO] Model loaded successfully as {algorithm.upper()}!")
        print(f"[INFO] Model policy: {type(model.policy).__name__}")
        
        # Check if it has custom features extractor
        if hasattr(model.policy, 'features_extractor'):
            print(f"[INFO] Features extractor: {type(model.policy.features_extractor).__name__}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[ERROR] Make sure multi_agent_extractors_td3.py is available and importable")
        raise RuntimeError(f"Failed to load model: {e}")

    # Adaptive difficulty hyperparameters (matching training script)
    adaptive_params = {
        'episode_length_sec': DEFAULT_DURATION_SEC,
        'target_radius_start': 0.1,
        'target_radius_max': 1.0,
        'target_radius_increment': 0.1,
        'target_tolerance': 0.05,
        'success_threshold': 0.9,
        'evaluation_window': 100,
        'crash_penalty': 200.0,
        'bounds_penalty': 200.0,
    }

    # Create evaluation environment for quick assessment
    try:
        print("[INFO] Creating adaptive difficulty evaluation environment...")
        eval_env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            gui=False,
            record=False,
            **adaptive_params
        )
        
        print('[INFO] Action space:', eval_env.action_space)
        print('[INFO] Observation space:', eval_env.observation_space)
        
        # Load VecNormalize stats if available
        vec_env = None
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            print(f"[INFO] Loading VecNormalize stats from: {vec_normalize_path}")
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv
                vec_env = DummyVecEnv([lambda: eval_env])
                vec_env = VecNormalize.load(vec_normalize_path, vec_env)
                vec_env.training = False  # Don't update stats during evaluation
                print("[INFO] VecNormalize stats loaded successfully")
            except Exception as e:
                print(f"[WARNING] Failed to load VecNormalize stats: {e}")
                vec_env = None
        
        # Use the wrapped environment for evaluation if available
        eval_env_for_eval = vec_env if vec_env else eval_env
        
        # Quick evaluation
        print(f"[INFO] Running quick evaluation...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env_for_eval, n_eval_episodes=3, deterministic=True
        )
        print(f"[EVAL] Quick assessment - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        if vec_env:
            vec_env.close()
        else:
            eval_env.close()
        
    except Exception as e:
        print(f"[WARNING] Failed to run quick evaluation: {e}")

    # Store all episode results
    all_episode_summaries = []
    all_episode_metrics = []

    # Run demonstration episodes
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"[INFO] Running adaptive difficulty demonstration episode {episode + 1}/{num_episodes}...")
        print(f"{'='*60}")
        
        try:
            # Create demonstration environment
            demo_env = MultiTargetAviary(
                num_drones=DEFAULT_DRONES,
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                gui=gui,
                record=record_video,
                **adaptive_params
            )
            
            # Apply normalization if available
            demo_vec_env = None
            if vec_normalize_path and os.path.exists(vec_normalize_path):
                try:
                    from stable_baselines3.common.vec_env import DummyVecEnv
                    demo_vec_env = DummyVecEnv([lambda: demo_env])
                    demo_vec_env = VecNormalize.load(vec_normalize_path, demo_vec_env)
                    demo_vec_env.training = False
                    print(f"[INFO] Applied VecNormalize to demo environment")
                except Exception as e:
                    print(f"[WARNING] Failed to apply VecNormalize to demo: {e}")
                    demo_vec_env = None
            
            # Use the appropriate environment
            env_for_demo = demo_vec_env if demo_vec_env else demo_env
            
            # Initialize metrics collector
            metrics_collector = AdaptiveDifficultyMetricsCollector(DEFAULT_DRONES)
            
            # Optional: Create logger for trajectory visualization
            logger = None
            if plot:
                try:
                    logger = Logger(
                        logging_freq_hz=freq, 
                        num_drones=DEFAULT_DRONES, 
                        output_folder=demo_folder
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to create logger: {e}")
            
            # Reset environment
            obs = env_for_demo.reset(seed=42 + episode)
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle newer gym versions
                info = obs[1] if len(obs) > 1 else {}
            else:
                info = {}
            
            start_time = time.time()
            
            max_steps = demo_env.max_episode_steps
            print(f"[INFO] Running for maximum {max_steps} steps...")
            
            if 'target_radius' in info:
                print(f"[INFO] Initial target radius: {info['target_radius']:.3f}m")
            if 'current_targets' in info:
                print(f"[INFO] Target position: {info['current_targets'][0]}")
            
            # Main demonstration loop
            for i in range(max_steps + 50):  # Add buffer steps
                try:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Step environment
                    step_result = env_for_demo.step(action)
                    if len(step_result) == 5:
                        obs, reward, done, truncated, info = step_result
                    else:
                        obs, reward, done, info = step_result
                        truncated = False
                    
                    # Handle vectorized environments
                    if isinstance(reward, (list, np.ndarray)):
                        reward = reward[0]
                    if isinstance(done, (list, np.ndarray)):
                        done = done[0]
                    if isinstance(truncated, (list, np.ndarray)):
                        truncated = truncated[0]
                    if isinstance(info, (list, np.ndarray)):
                        info = info[0] if len(info) > 0 else {}
                    
                    # Update metrics
                    metrics_collector.update(action, obs, reward, info, i)
                    
                    # Render if GUI enabled (only for non-vectorized env)
                    if gui and not demo_vec_env:
                        demo_env.render()
                        sync(i, start_time, demo_env.CTRL_TIMESTEP)
                    
                    # Log progress every 30 steps
                    if i % 30 == 0:
                        dist = info.get('min_distance_to_target', 0)
                        radius = info.get('target_radius', 0)
                        success_rate = info.get('success_rate_last_100', 0)
                        print(f"Step {i:3d} | Dist: {dist:.3f} | Radius: {radius:.3f} | "
                              f"Success Rate: {success_rate:.3f} | Reward: {reward:6.1f} | "
                              f"Total: {metrics_collector.total_reward:6.1f}")
                    
                    # Optional logging for trajectory plotting
                    if plot and logger and DEFAULT_OBS == ObservationType.KIN:
                        try:
                            # Handle vectorized observations
                            if demo_vec_env:
                                obs_arr = obs[0] if obs.ndim > 1 and obs.shape[0] == 1 else obs
                            else:
                                obs_arr = obs.squeeze() if obs.ndim > 1 else obs
                            
                            act_arr = action.squeeze() if action.ndim > 1 else action
                            
                            for d in range(DEFAULT_DRONES):
                                # Extract state information safely
                                if obs_arr.ndim > 1:
                                    drone_obs = obs_arr[d]
                                else:
                                    drone_obs = obs_arr
                                
                                # Create state vector for logging
                                pos = drone_obs[0:3] if len(drone_obs) >= 3 else np.pad(drone_obs, (0, 3-len(drone_obs)))
                                vel_etc = drone_obs[3:15] if len(drone_obs) > 15 else np.pad(drone_obs[3:], (0, 12-len(drone_obs[3:])))
                                
                                # Handle different action types
                                if act_arr.ndim > 1 and act_arr.shape[0] >= DEFAULT_DRONES * 4:
                                    # Action array is [drone0_action[4], drone1_action[4], ...]
                                    drone_action = act_arr[d*4:(d+1)*4]
                                elif act_arr.ndim == 1 and len(act_arr) == 4:
                                    # Single drone action
                                    drone_action = act_arr
                                else:
                                    drone_action = [0, 0, 0, 0]
                                
                                state = np.hstack([
                                    pos,                    # position
                                    np.zeros(4),           # quaternion placeholder
                                    vel_etc,               # velocity, etc.
                                    drone_action           # action (4 values)
                                ])
                                
                                # Ensure state is the right length (20 elements expected)
                                if len(state) < 20:
                                    state = np.pad(state, (0, 20-len(state)))
                                elif len(state) > 20:
                                    state = state[:20]
                                
                                logger.log(drone=d, timestamp=i/freq, state=state, control=np.zeros(12))
                        except Exception as e:
                            if i == 0:  # Only warn once
                                print(f"[WARNING] Logging failed: {e}")
                    
                    # Check for episode termination
                    if done or truncated:
                        outcome = "SUCCESS" if info.get('episode_success', False) else "FAILURE"
                        print(f"[INFO] Episode {episode + 1} finished at step {i}: {outcome}")
                        break
                        
                except Exception as e:
                    print(f"[ERROR] Error at step {i}: {e}")
                    break
            
            # Get episode summary
            episode_summary = metrics_collector.get_summary()
            all_episode_summaries.append(episode_summary)
            all_episode_metrics.append(metrics_collector.episode_data)
            
            # Print episode results
            print(f"\n[RESULTS] Episode {episode + 1} Summary:")
            success_str = "SUCCESS" if episode_summary.get('episode_success', False) else "FAILURE"
            print(f"  • Outcome: {success_str}")
            print(f"  • Total Reward: {episode_summary.get('total_reward', 0):.2f}")
            print(f"  • Total Steps: {episode_summary.get('total_steps', 0)}")
            print(f"  • Final Target Radius: {episode_summary.get('final_target_radius', 0):.3f}m")
            print(f"  • Success Rate: {episode_summary.get('final_success_rate', 0):.3f}")
            
            if 'distance_stats' in episode_summary:
                dist_stats = episode_summary['distance_stats']
                print(f"  • Final Distance: {dist_stats.get('final', 0):.3f}")
                print(f"  • Mean Distance: {dist_stats.get('mean', 0):.3f}")
                print(f"  • Best Distance: {dist_stats.get('min_achieved', 0):.3f}")
                print(f"  • Distance Improvement: {dist_stats.get('improvement', 0):.3f}")
            
            # Save metrics to file if requested
            if save_metrics:
                try:
                    import json
                    metrics_file = os.path.join(demo_folder, f'episode_{episode + 1}_adaptive_metrics.json')
                    
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_data = {}
                    for key, value in episode_summary.items():
                        if isinstance(value, np.ndarray):
                            serializable_data[key] = value.tolist()
                        elif isinstance(value, dict):
                            serializable_data[key] = {}
                            for k, v in value.items():
                                if isinstance(v, np.ndarray):
                                    serializable_data[key][k] = v.tolist()
                                else:
                                    serializable_data[key][k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
                        else:
                            serializable_data[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
                    
                    with open(metrics_file, 'w') as f:
                        json.dump(serializable_data, f, indent=2)
                    print(f"[INFO] Saved adaptive difficulty metrics to: {metrics_file}")
                except Exception as e:
                    print(f"[WARNING] Failed to save metrics: {e}")
            
            # Generate enhanced plots for this episode
            if plot and detailed_analysis:
                plot_path = create_enhanced_adaptive_plots(metrics_collector, demo_folder, episode + 1)
                if plot_path:
                    print(f"[INFO] Generated adaptive difficulty analysis plots")
            
            # Generate trajectory plots using logger
            if plot and logger and DEFAULT_OBS == ObservationType.KIN:
                try:
                    logger.plot()
                    print(f"[INFO] Generated trajectory plots in: {demo_folder}")
                except Exception as e:
                    print(f"[WARNING] Failed to generate trajectory plots: {e}")
            
            # Close environments
            if demo_vec_env:
                demo_vec_env.close()
            else:
                demo_env.close()
            
        except Exception as e:
            print(f"[ERROR] Episode {episode + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate overall summary if multiple episodes
    if num_episodes > 1 and all_episode_summaries:
        print(f"\n{'='*60}")
        print("[INFO] OVERALL ADAPTIVE DIFFICULTY DEMONSTRATION SUMMARY")
        print(f"{'='*60}")
        
        # Calculate aggregate statistics
        total_rewards = [ep.get('total_reward', 0) for ep in all_episode_summaries]
        success_outcomes = [ep.get('episode_success', False) for ep in all_episode_summaries]
        final_radii = [ep.get('final_target_radius', 0) for ep in all_episode_summaries]
        success_rates = [ep.get('final_success_rate', 0) for ep in all_episode_summaries]
        mean_distances = [ep.get('distance_stats', {}).get('mean', 0) for ep in all_episode_summaries]
        
        success_rate_overall = np.mean(success_outcomes)
        
        print(f"Episodes: {len(all_episode_summaries)}")
        print(f"Overall Success Rate: {success_rate_overall:.3f}")
        print(f"Reward - Mean: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Reward - Range: [{np.min(total_rewards):.2f}, {np.max(total_rewards):.2f}]")
        print(f"Distance - Mean: {np.mean(mean_distances):.3f} ± {np.std(mean_distances):.3f}")
        print(f"Final Target Radius - Mean: {np.mean(final_radii):.3f} ± {np.std(final_radii):.3f}")
        print(f"Max Target Radius Achieved: {np.max(final_radii):.3f}m")
        
        # Save overall summary
        if save_metrics:
            try:
                import json
                overall_summary = {
                    'num_episodes': len(all_episode_summaries),
                    'algorithm': algorithm,
                    'overall_success_rate': float(success_rate_overall),
                    'reward_stats': {
                        'mean': float(np.mean(total_rewards)),
                        'std': float(np.std(total_rewards)),
                        'min': float(np.min(total_rewards)),
                        'max': float(np.max(total_rewards)),
                        'all_rewards': [float(r) for r in total_rewards]
                    },
                    'distance_stats': {
                        'mean': float(np.mean(mean_distances)),
                        'std': float(np.std(mean_distances)),
                        'min': float(np.min(mean_distances)),
                        'max': float(np.max(mean_distances)),
                        'all_distances': [float(d) for d in mean_distances]
                    },
                    'adaptive_difficulty_stats': {
                        'final_radius_mean': float(np.mean(final_radii)),
                        'final_radius_std': float(np.std(final_radii)),
                        'max_radius_achieved': float(np.max(final_radii)),
                        'final_success_rate_mean': float(np.mean(success_rates)),
                    },
                    'individual_episodes': all_episode_summaries
                }
                
                summary_file = os.path.join(demo_folder, 'adaptive_overall_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(overall_summary, f, indent=2)
                print(f"[INFO] Saved overall adaptive difficulty summary to: {summary_file}")
            except Exception as e:
                print(f"[WARNING] Failed to save overall summary: {e}")

    print(f"\n[INFO] Adaptive difficulty demonstration completed!")
    print(f"[INFO] Results saved to: {demo_folder}")
    return demo_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced demonstration of pre-trained adaptive difficulty drone model')
    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to the trained model (.zip file)')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, 
                        help='Use PyBullet GUI during demonstration (default: True)')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, 
                        help='Record video of demonstration (default: False)')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='Results folder for demonstration outputs')
    parser.add_argument('--plot', default=True, type=str2bool, 
                        help='Generate analysis plots (default: True)')
    parser.add_argument('--num_episodes', default=1, type=int,
                        help='Number of demonstration episodes to run (default: 1)')
    parser.add_argument('--detailed_analysis', default=True, type=str2bool,
                        help='Perform detailed analysis with enhanced plots (default: True)')
    parser.add_argument('--save_metrics', default=True, type=str2bool,
                        help='Save metrics to JSON files (default: True)')
    parser.add_argument('--vec_normalize_path', default=None, type=str,
                        help='Path to VecNormalize stats file (vec_normalize.pkl)')
    
    args = parser.parse_args()

    # Validate arguments
    if not args.model_path:
        print("[ERROR] Model path is required!")
        exit(1)
    
    if args.num_episodes < 1:
        print("[ERROR] Number of episodes must be at least 1!")
        exit(1)
    
    print("="*60)
    print("Adaptive Difficulty Drone Demonstration - Enhanced")
    print("="*60)
    print(f"Action type: {DEFAULT_ACT}")
    print(f"Multi-agent extractors: Available")
    print(f"Supports: TD3, SAC, PPO models")
    print(f"Adaptive difficulty: 0.1m -> 1.0m target radius")
    print("="*60)
    
    try:
        demo_folder = run_demonstration(
            args.model_path,
            args.output_folder,
            args.gui,
            args.record_video,
            args.plot,
            args.num_episodes,
            args.detailed_analysis,
            args.save_metrics,
            args.vec_normalize_path
        )
        
        print(f"\n[SUCCESS] Adaptive difficulty demonstration completed successfully!")
        print(f"[SUCCESS] Check results in: {demo_folder}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Demonstration interrupted by user")
        exit(0)
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)