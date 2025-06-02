#!/usr/bin/env python3
"""
demo_adaptive_difficulty.py: Load pre-trained adaptive difficulty model and run demonstration
Updated to work with adaptive difficulty single-drone system with obstacle support
Supports TD3, SAC, and PPO models with adaptive target radius and obstacles
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
DEFAULT_OBS           = ObservationType.KIN_DEPTH
DEFAULT_ACT           = ActionType('rpm')
DEFAULT_DRONES        = 1
DEFAULT_DURATION_SEC  = 6.0


class AdaptiveDifficultyMetricsCollector:
    """Metrics collection for adaptive difficulty demonstration analysis with obstacle support"""
    
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
            'step_info': [],
            # New obstacle-related metrics
            'obstacle_distances': [],
            'num_obstacles_history': [],
            'obstacle_info_history': []
        }
        self.total_reward = 0
        self.steps = 0
        self.final_target_radius = 0.0
        self.final_success_rate = 0.0
        self.final_num_obstacles = 0
        self.min_obstacle_distance_achieved = np.inf
    
    def update(self, action, obs, reward, info, step):
        """Update metrics with current step data - fixed to handle KIN_DEPTH observations"""
        self.steps += 1
        self.total_reward += reward
        
        # Basic metrics
        self.episode_data['rewards'].append(reward)
        self.episode_data['actions'].append(action.copy() if hasattr(action, 'copy') else np.array(action))
        
        # Position and target tracking - FIXED FOR KIN_DEPTH
        if obs is not None:
            if isinstance(obs, dict) and 'kin' in obs:  # Handle KIN_DEPTH observations
                kin_obs = obs['kin']
                if len(kin_obs) > 0:
                    if hasattr(kin_obs, 'ndim') and kin_obs.ndim > 1:
                        position = kin_obs[0, 0:3]  # First drone position
                    else:
                        position = kin_obs[0:3]
                    self.episode_data['positions'].append(position.copy())
            elif hasattr(obs, 'ndim') and len(obs) > 0:  # Handle KIN observations (numpy arrays)
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
        
        # Obstacle-related metrics
        if 'num_obstacles' in info:
            self.episode_data['num_obstacles_history'].append(info['num_obstacles'])
            self.final_num_obstacles = info['num_obstacles']
        
        if 'min_obstacle_distance' in info and info['min_obstacle_distance'] != np.inf:
            self.episode_data['obstacle_distances'].append(info['min_obstacle_distance'])
            self.min_obstacle_distance_achieved = min(self.min_obstacle_distance_achieved, info['min_obstacle_distance'])
        
        if 'obstacles' in info:
            self.episode_data['obstacle_info_history'].append(info['obstacles'])
        
        # Store step info for detailed analysis
        step_info = {
            'step': step,
            'reward': reward,
            'distance': info.get('min_distance_to_target', 0),
            'target_radius': info.get('target_radius', 0),
            'success_rate': info.get('success_rate_last_100', 0),
            'episode_success': info.get('episode_success', False),
            # Obstacle info
            'num_obstacles': info.get('num_obstacles', 0),
            'min_obstacle_distance': info.get('min_obstacle_distance', np.inf),
        }
        self.episode_data['step_info'].append(step_info)
    
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
        
        # NEW: Obstacle statistics
        if self.episode_data['obstacle_distances']:
            obstacle_distances = self.episode_data['obstacle_distances']
            summary['obstacle_stats'] = {
                'final_num_obstacles': self.final_num_obstacles,
                'min_obstacle_distance_achieved': self.min_obstacle_distance_achieved,
                'mean_obstacle_distance': np.mean(obstacle_distances),
                'std_obstacle_distance': np.std(obstacle_distances),
                'collision_risk_steps': len([d for d in obstacle_distances if d < 0.15]),  # Steps with high collision risk
            }
        else:
            summary['obstacle_stats'] = {
                'final_num_obstacles': self.final_num_obstacles,
                'min_obstacle_distance_achieved': np.inf,
                'mean_obstacle_distance': np.inf,
                'std_obstacle_distance': 0,
                'collision_risk_steps': 0,
            }
        
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
    """Create comprehensive analysis plots for adaptive difficulty system with obstacles"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        episode_data = metrics_collector.episode_data
        summary = metrics_collector.get_summary()
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(24, 15))  # Larger figure to accommodate obstacle plots
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
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
        
        # Plot 4: NEW - Obstacle distance over time
        ax4 = fig.add_subplot(gs[0, 3])
        if episode_data['obstacle_distances']:
            obstacle_distances = episode_data['obstacle_distances']
            steps = range(len(obstacle_distances))
            ax4.plot(steps, obstacle_distances, label='Min Obstacle Distance', linewidth=2, color='red')
            
            # Add collision threshold line
            collision_threshold = 0.1  # From environment
            ax4.axhline(y=collision_threshold, color='red', linestyle='--', 
                       label=f'Collision Threshold ({collision_threshold})')
            
            ax4.set_title('Distance to Nearest Obstacle')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Distance')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No Obstacle Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distance to Nearest Obstacle')
        
        # Plot 5: Success rate progression
        ax5 = fig.add_subplot(gs[1, 0])
        if episode_data['success_rate_history']:
            success_history = episode_data['success_rate_history']
            ax5.plot(success_history, linewidth=2, color='orange', label='Success Rate')
            ax5.axhline(y=0.9, color='red', linestyle='--', label='Threshold (0.9)')
            ax5.set_title('Success Rate Progression')
            ax5.set_xlabel('Steps')
            ax5.set_ylabel('Success Rate')
            ax5.set_ylim(0, 1)
            ax5.legend()
            ax5.grid(True)
        
        # Plot 6: 3D trajectory with obstacles
        ax6 = fig.add_subplot(gs[1, 1], projection='3d')
        if episode_data['positions'] and episode_data['targets']:
            positions = np.array(episode_data['positions'])
            targets = np.array(episode_data['targets'])
            
            # Plot trajectory
            ax6.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    'b-', alpha=0.7, linewidth=2, label='Drone Path')
            
            # Plot start and end positions
            ax6.scatter(*positions[0], color='green', s=100, label='Start')
            ax6.scatter(*positions[-1], color='red', s=100, label='End')
            
            # Plot targets (they change, so plot first and last)
            if len(targets) > 0:
                ax6.scatter(*targets[0][0], color='orange', s=100, marker='*', label='Initial Target')
                if len(targets) > 1:
                    ax6.scatter(*targets[-1][0], color='purple', s=100, marker='*', label='Final Target')
            
            # NEW: Plot obstacles if available
            if episode_data['obstacle_info_history']:
                # Use the last obstacle configuration
                obstacles = episode_data['obstacle_info_history'][-1]
                for obs in obstacles:
                    pos = obs['position']
                    obs_type = obs.get('type', 'unknown')
                    color = 'brown' if obs_type == 'cube' else 'gray'
                    ax6.scatter(*pos, color=color, s=50, marker='s' if obs_type == 'cube' else 'o', alpha=0.7)
                
                # Add legend entry for obstacles
                ax6.scatter([], [], color='brown', s=50, marker='s', alpha=0.7, label='Obstacles')
            
            ax6.set_title('3D Trajectory with Obstacles')
            ax6.set_xlabel('X')
            ax6.set_ylabel('Y')
            ax6.set_zlabel('Z')
            ax6.legend()
        
        # Plot 7: Distance distribution
        ax7 = fig.add_subplot(gs[1, 2])
        if episode_data['distances']:
            distances = episode_data['distances']
            ax7.hist(distances, bins=30, alpha=0.7, edgecolor='black', color='green')
            ax7.axvline(np.mean(distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(distances):.3f}')
            ax7.axvline(0.05, color='orange', linestyle='--', label='Target Tolerance')
            ax7.set_title('Distance Distribution')
            ax7.set_xlabel('Distance to Target')
            ax7.set_ylabel('Frequency')
            ax7.legend()
            ax7.grid(True)
        
        # Plot 8: NEW - Obstacle distance distribution
        ax8 = fig.add_subplot(gs[1, 3])
        if episode_data['obstacle_distances']:
            obstacle_distances = episode_data['obstacle_distances']
            ax8.hist(obstacle_distances, bins=30, alpha=0.7, edgecolor='black', color='red')
            ax8.axvline(np.mean(obstacle_distances), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(obstacle_distances):.3f}')
            ax8.axvline(0.1, color='red', linestyle='--', label='Collision Risk')
            ax8.set_title('Obstacle Distance Distribution')
            ax8.set_xlabel('Distance to Nearest Obstacle')
            ax8.set_ylabel('Frequency')
            ax8.legend()
            ax8.grid(True)
        else:
            ax8.text(0.5, 0.5, 'No Obstacle Data', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Obstacle Distance Distribution')
        
        # Plot 9: Action analysis
        ax9 = fig.add_subplot(gs[2, 0])
        if episode_data['actions']:
            actions = np.array(episode_data['actions'])
            if actions.ndim > 1:
                # Plot action variance over time
                action_var = np.var(actions, axis=1)
                ax9.plot(action_var, alpha=0.7, color='brown')
                ax9.set_title('Action Variance Over Time')
                ax9.set_xlabel('Steps')
                ax9.set_ylabel('Action Variance')
                ax9.grid(True)
        
        # Plot 10: Reward distribution
        ax10 = fig.add_subplot(gs[2, 1])
        if episode_data['rewards']:
            rewards = episode_data['rewards']
            unique_rewards, counts = np.unique(rewards, return_counts=True)
            
            # Create bar plot for discrete rewards
            ax10.bar(unique_rewards, counts, alpha=0.7, edgecolor='black')
            ax10.set_title('Reward Distribution')
            ax10.set_xlabel('Reward Value')
            ax10.set_ylabel('Frequency')
            ax10.grid(True, axis='y')
            
            # Add labels on bars
            for reward, count in zip(unique_rewards, counts):
                ax10.text(reward, count + max(counts)*0.01, str(count), 
                        ha='center', va='bottom')
        
        # Plot 11: Performance summary (text)
        ax11 = fig.add_subplot(gs[2, 2:])
        ax11.axis('off')
        
        # Create summary text with obstacle info
        success_str = "SUCCESS" if summary.get('episode_success', False) else "FAILURE"
        obstacle_stats = summary.get('obstacle_stats', {})
        
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
        
        Obstacle Navigation:
        • Number of Obstacles: {obstacle_stats.get('final_num_obstacles', 0)}
        • Min Obstacle Distance: {obstacle_stats.get('min_obstacle_distance_achieved', np.inf):.3f}
        • Mean Obstacle Distance: {obstacle_stats.get('mean_obstacle_distance', np.inf):.3f}
        • Collision Risk Steps: {obstacle_stats.get('collision_risk_steps', 0)}
        """
        
        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Plot 12: Performance gauges with obstacle avoidance
        ax12 = fig.add_subplot(gs[3, :])
        
        # Calculate performance scores
        distance_score = max(0, 100 * (1 - summary.get('distance_stats', {}).get('mean', 1) / 1.0))
        success_score = 100 if summary.get('episode_success', False) else 0
        difficulty_score = summary.get('final_target_radius', 0) / 1.0 * 100
        
        # NEW: Obstacle avoidance score
        min_obs_dist = obstacle_stats.get('min_obstacle_distance_achieved', np.inf)
        if min_obs_dist == np.inf:
            obstacle_score = 100  # No obstacles or perfect avoidance
        else:
            # Score based on minimum distance maintained (0.2m = 100%, 0.05m = 0%)
            obstacle_score = max(0, min(100, (min_obs_dist - 0.05) / (0.2 - 0.05) * 100))
        
        overall_score = (distance_score + success_score + difficulty_score + obstacle_score) / 4
        
        # Create gauge plot
        categories = ['Distance\nPerformance', 'Episode\nSuccess', 'Difficulty\nLevel', 'Obstacle\nAvoidance', 'Overall\nScore']
        scores = [distance_score, success_score, difficulty_score, obstacle_score, overall_score]
        colors = ['green' if s > 70 else 'orange' if s > 40 else 'red' for s in scores]
        
        bars = ax12.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
        ax12.set_ylim(0, 100)
        ax12.set_title('Performance Metrics (Including Obstacle Avoidance)')
        ax12.set_ylabel('Score (0-100)')
        ax12.grid(True, axis='y')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Adaptive Difficulty Drone with Obstacles - Episode {episode_num}', 
                     fontsize=16, fontweight='bold')
        
        # Save plot
        plot_path = os.path.join(output_folder, f'demo_episode_{episode_num}_adaptive_obstacles_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved adaptive difficulty with obstacles analysis plot: {plot_path}")
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
    Load pre-trained adaptive difficulty model and run demonstration with obstacles
    
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
    print(f"[INFO] OBSTACLES ENABLED with probability: 1.0 (maximum density)")
    
    # Prepare output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    demo_folder = os.path.join(output_folder, f'adaptive_demo_obstacles_{timestamp}')
    os.makedirs(demo_folder, exist_ok=True)

    # Get control frequency
    try:
        dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        freq = int(dummy_env.CTRL_FREQ)
        dummy_env.close()
    except Exception as e:
        print(f"[ERROR] Failed to create dummy environment: {e}")
        return
    
    freq = 40

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

    # Adaptive difficulty hyperparameters with OBSTACLES ENABLED
    adaptive_params = {
        'episode_length_sec': DEFAULT_DURATION_SEC,
        'target_radius_start': 0.5,
        'target_radius_max': 3.0,
        'target_radius_increment': 0.1,
        'target_tolerance': 0.05,
        'success_threshold': 0.9,
        'evaluation_window': 100,
        'crash_penalty': 200.0,
        'bounds_penalty': 200.0,
        'ctrl_freq': 40,
        'pyb_freq': 120,
        # NEW: OBSTACLE PARAMETERS - SET TO MAXIMUM DIFFICULTY
        'add_obstacles': True,        # Enable obstacles
        'obs_prob': 1.0,             # Maximum obstacle density as requested
        'obstacle_size': 0.2         # Standard obstacle size
    }
    
    print(f"[INFO] Obstacle parameters: add_obstacles={adaptive_params['add_obstacles']}, " +
          f"obs_prob={adaptive_params['obs_prob']}, obstacle_size={adaptive_params['obstacle_size']}")

    # Create evaluation environment for quick assessment
    try:
        print("[INFO] Creating adaptive difficulty evaluation environment with obstacles...")
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
        print(f"[INFO] Running quick evaluation with obstacles...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env_for_eval, n_eval_episodes=3, deterministic=True
        )
        print(f"[EVAL] Quick assessment with obstacles - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
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
        print(f"[INFO] Running adaptive difficulty demonstration with OBSTACLES episode {episode + 1}/{num_episodes}...")
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
            reset_result = env_for_demo.reset(seed=42 + episode)
            
            # Handle different reset return formats
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
                info = reset_result[1] if len(reset_result) > 1 else {}
                # Ensure info is a dictionary
                if not isinstance(info, dict):
                    info = {}
            else:
                obs = reset_result
                info = {}
            
            # Set custom target for demo - challenging position with obstacles in the way
            if demo_vec_env:
                # For vectorized environments, we need to access the underlying environment
                if hasattr(demo_vec_env, 'venv') and hasattr(demo_vec_env.venv, 'envs'):
                    demo_vec_env.venv.envs[0].current_targets = demo_vec_env.venv.envs[0].start_positions + np.array([[-2.0, -2.0, -0.25]])
                elif hasattr(demo_vec_env, 'envs'):
                    demo_vec_env.envs[0].current_targets = demo_vec_env.envs[0].start_positions + np.array([[-2.0, -2.0, -0.25]])
            else:
                env_for_demo.current_targets = env_for_demo.start_positions + np.array([[-2.0, -2.0, -0.25]])
            
            # Get initial info if missing
            if not info or 'target_radius' not in info:
                try:
                    if demo_vec_env:
                        # Try to get info from underlying environment
                        if hasattr(demo_vec_env, 'venv') and hasattr(demo_vec_env.venv, 'envs'):
                            initial_info = demo_vec_env.venv.envs[0]._computeInfo()
                        elif hasattr(demo_vec_env, 'envs'):
                            initial_info = demo_vec_env.envs[0]._computeInfo()
                        else:
                            initial_info = {}
                    else:
                        initial_info = env_for_demo._computeInfo()
                    
                    if isinstance(initial_info, dict):
                        info.update(initial_info)
                        print(f"[INFO] Retrieved initial info from environment")
                    
                except Exception as e:
                    print(f"[WARNING] Could not get initial info from environment: {e}")
            
            start_time = time.time()
            
            max_steps = demo_env.max_episode_steps
            print(f"[INFO] Running for maximum {max_steps} steps...")
            
            if 'target_radius' in info:
                print(f"[INFO] Initial target radius: {info['target_radius']:.3f}m")
            if 'current_targets' in info:
                print(f"[INFO] Target position: {info['current_targets'][0]}")
            if 'num_obstacles' in info:
                print(f"[INFO] Number of obstacles: {info['num_obstacles']}")
            
            # Main demonstration loop
            for i in range(max_steps + 50):  # Add buffer steps
                try:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Step environment
                    step_result = env_for_demo.step(action)
                    
                    # Debug step result format
                    if i == 0:  # Only debug on first step
                        print(f"[DEBUG] Step result length: {len(step_result)}")
                        print(f"[DEBUG] Step result types: {[type(x) for x in step_result]}")
                    
                    if len(step_result) == 5:
                        obs, reward, done, truncated, info = step_result
                    else:
                        obs, reward, done, info = step_result
                        truncated = False
                    
                    # Debug info type
                    if i == 0:
                        print(f"[DEBUG] Info type before processing: {type(info)}")
                        if hasattr(info, 'shape'):
                            print(f"[DEBUG] Info shape: {info.shape}")
                    
                    # Handle vectorized environments
                    if isinstance(reward, (list, np.ndarray)):
                        reward = reward[0]
                    if isinstance(done, (list, np.ndarray)):
                        done = done[0]
                    if isinstance(truncated, (list, np.ndarray)):
                        truncated = truncated[0]
                    if isinstance(info, (list, np.ndarray)):
                        if len(info) > 0 and isinstance(info[0], dict):
                            info = info[0]
                        else:
                            info = {}
                    elif not isinstance(info, dict):
                        # If info is not a dict (e.g., numpy scalar), create empty dict
                        if i == 0:  # Only warn once per episode
                            print(f"[WARNING] Info is not a dict (type: {type(info)}), creating empty dict")
                        info = {}
                    
                    # Debug final info type
                    if i == 0:
                        print(f"[DEBUG] Info type after processing: {type(info)}")
                        print(f"[DEBUG] Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
                    
                    # If info is empty or missing critical information, try to get it directly from environment
                    if not info or 'target_radius' not in info:
                        try:
                            if demo_vec_env:
                                # Try to get info from underlying environment
                                if hasattr(demo_vec_env, 'venv') and hasattr(demo_vec_env.venv, 'envs'):
                                    direct_info = demo_vec_env.venv.envs[0]._computeInfo()
                                elif hasattr(demo_vec_env, 'envs'):
                                    direct_info = demo_vec_env.envs[0]._computeInfo()
                                else:
                                    direct_info = {}
                            else:
                                direct_info = env_for_demo._computeInfo()
                            
                            # Merge with existing info
                            if isinstance(direct_info, dict):
                                info.update(direct_info)
                            
                            if i == 0 and direct_info:
                                print(f"[INFO] Successfully retrieved info from environment directly")
                                
                        except Exception as e:
                            if i == 0:
                                print(f"[WARNING] Could not get info directly from environment: {e}")
                    
                    # Update metrics
                    metrics_collector.update(action, obs, reward, info, i)
                    
                    # Render if GUI enabled (only for non-vectorized env)
                    if gui and not demo_vec_env:
                        demo_env.render()
                        sync(i, start_time, demo_env.CTRL_TIMESTEP)
                    
                    # Log progress every 30 steps - NOW WITH OBSTACLE INFO
                    if i % 30 == 0:
                        dist = info.get('min_distance_to_target', 0)
                        radius = info.get('target_radius', 0)
                        success_rate = info.get('success_rate_last_100', 0)
                        num_obstacles = info.get('num_obstacles', 0)
                        min_obs_dist = info.get('min_obstacle_distance', np.inf)
                        
                        print(f"Step {i:3d} | Dist: {dist:.3f} | Radius: {radius:.3f} | " +
                              f"Success Rate: {success_rate:.3f} | Obstacles: {num_obstacles} | " +
                              f"Min Obs Dist: {min_obs_dist:.3f} | Reward: {reward:6.1f} | " +
                              f"Total: {metrics_collector.total_reward:6.1f}")
                    
                    # Optional logging for trajectory plotting - FIXED FOR KIN_DEPTH
                    if plot and logger:  # Remove DEFAULT_OBS check to handle both observation types
                        try:
                            # Handle different observation types
                            if isinstance(obs, dict) and 'kin' in obs:  # KIN_DEPTH observations
                                if demo_vec_env:
                                    kin_obs = obs['kin'][0] if obs['kin'].ndim > 1 and obs['kin'].shape[0] == 1 else obs['kin']
                                else:
                                    kin_obs = obs['kin'].squeeze() if obs['kin'].ndim > 1 else obs['kin']
                                obs_arr = kin_obs
                            else:  # KIN observations
                                if demo_vec_env:
                                    obs_arr = obs[0] if obs.ndim > 1 and obs.shape[0] == 1 else obs
                                else:
                                    obs_arr = obs.squeeze() if obs.ndim > 1 else obs
                            
                            act_arr = action.squeeze() if hasattr(action, 'ndim') and action.ndim > 1 else action
                            
                            for d in range(DEFAULT_DRONES):
                                # Extract state information safely
                                if hasattr(obs_arr, 'ndim') and obs_arr.ndim > 1:
                                    drone_obs = obs_arr[d]
                                else:
                                    drone_obs = obs_arr
                                
                                # Create state vector for logging
                                if len(drone_obs) >= 3:
                                    pos = drone_obs[0:3]
                                else:
                                    pos = np.zeros(3)  # Default position if not available
                                    
                                vel_etc = drone_obs[3:15] if len(drone_obs) > 15 else np.zeros(12)
                                
                                # Handle different action types
                                drone_action = np.zeros(4)  # Default action
                                if isinstance(act_arr, np.ndarray):
                                    if act_arr.ndim > 1 and act_arr.shape[0] >= DEFAULT_DRONES * 4:
                                        # Action array is [drone0_action[4], drone1_action[4], ...]
                                        drone_action = act_arr[d*4:(d+1)*4]
                                    elif act_arr.ndim == 1 and len(act_arr) == 4:
                                        # Single drone action
                                        drone_action = act_arr
                                
                                state = np.hstack([
                                    pos,                # position
                                    np.zeros(4),        # quaternion placeholder
                                    vel_etc,            # velocity, etc.
                                    drone_action        # action (4 values)
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
                                import traceback
                                traceback.print_exc()
                    
                    # Check for episode termination
                    if done or truncated:
                        outcome = "SUCCESS" if info.get('episode_success', False) else "FAILURE"
                        print(f"[INFO] Episode {episode + 1} finished at step {i}: {outcome}")
                        break
                        
                except Exception as e:
                    print(f"[ERROR] Error at step {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Get episode summary
            episode_summary = metrics_collector.get_summary()
            all_episode_summaries.append(episode_summary)
            all_episode_metrics.append(metrics_collector.episode_data)
            
            # Print episode results - NOW WITH OBSTACLE INFO
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
            
            # NEW: Print obstacle performance
            if 'obstacle_stats' in episode_summary:
                obs_stats = episode_summary['obstacle_stats']
                print(f"  • Number of Obstacles: {obs_stats.get('final_num_obstacles', 0)}")
                print(f"  • Min Obstacle Distance: {obs_stats.get('min_obstacle_distance_achieved', np.inf):.3f}")
                print(f"  • Mean Obstacle Distance: {obs_stats.get('mean_obstacle_distance', np.inf):.3f}")
                print(f"  • Collision Risk Steps: {obs_stats.get('collision_risk_steps', 0)}")
            
            # Save metrics to file if requested
            if save_metrics:
                try:
                    import json
                    metrics_file = os.path.join(demo_folder, f'episode_{episode + 1}_adaptive_obstacles_metrics.json')
                    
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
                    print(f"[INFO] Saved adaptive difficulty with obstacles metrics to: {metrics_file}")
                except Exception as e:
                    print(f"[WARNING] Failed to save metrics: {e}")
            
            # Generate enhanced plots for this episode
            if plot and detailed_analysis:
                plot_path = create_enhanced_adaptive_plots(metrics_collector, demo_folder, episode + 1)
                if plot_path:
                    print(f"[INFO] Generated adaptive difficulty with obstacles analysis plots")
            
            # Generate trajectory plots using logger
            if plot and logger:
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

    # Generate overall summary if multiple episodes - NOW WITH OBSTACLE STATS
    if num_episodes > 1 and all_episode_summaries:
        print(f"\n{'='*60}")
        print("[INFO] OVERALL ADAPTIVE DIFFICULTY WITH OBSTACLES DEMONSTRATION SUMMARY")
        print(f"{'='*60}")
        
        # Calculate aggregate statistics
        total_rewards = [ep.get('total_reward', 0) for ep in all_episode_summaries]
        success_outcomes = [ep.get('episode_success', False) for ep in all_episode_summaries]
        final_radii = [ep.get('final_target_radius', 0) for ep in all_episode_summaries]
        success_rates = [ep.get('final_success_rate', 0) for ep in all_episode_summaries]
        mean_distances = [ep.get('distance_stats', {}).get('mean', 0) for ep in all_episode_summaries]
        
        # NEW: Obstacle statistics
        obstacle_counts = [ep.get('obstacle_stats', {}).get('final_num_obstacles', 0) for ep in all_episode_summaries]
        min_obstacle_distances = [ep.get('obstacle_stats', {}).get('min_obstacle_distance_achieved', np.inf) for ep in all_episode_summaries]
        collision_risk_steps = [ep.get('obstacle_stats', {}).get('collision_risk_steps', 0) for ep in all_episode_summaries]
        
        success_rate_overall = np.mean(success_outcomes)
        
        print(f"Episodes: {len(all_episode_summaries)}")
        print(f"Overall Success Rate: {success_rate_overall:.3f}")
        print(f"Reward - Mean: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Reward - Range: [{np.min(total_rewards):.2f}, {np.max(total_rewards):.2f}]")
        print(f"Distance - Mean: {np.mean(mean_distances):.3f} ± {np.std(mean_distances):.3f}")
        print(f"Final Target Radius - Mean: {np.mean(final_radii):.3f} ± {np.std(final_radii):.3f}")
        print(f"Max Target Radius Achieved: {np.max(final_radii):.3f}m")
        
        # NEW: Print obstacle performance summary
        print(f"\nObstacle Navigation Performance:")
        print(f"Obstacles per Episode - Mean: {np.mean(obstacle_counts):.1f}")
        valid_distances = [d for d in min_obstacle_distances if d != np.inf]
        if valid_distances:
            print(f"Min Obstacle Distance - Mean: {np.mean(valid_distances):.3f} ± {np.std(valid_distances):.3f}")
            print(f"Best Obstacle Avoidance: {np.max(valid_distances):.3f}m")
        print(f"Total Collision Risk Steps: {np.sum(collision_risk_steps)}")
        
        # Save overall summary
        if save_metrics:
            try:
                import json
                overall_summary = {
                    'num_episodes': len(all_episode_summaries),
                    'algorithm': algorithm,
                    'obstacles_enabled': True,
                    'obstacle_probability': 1.0,
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
                    'obstacle_stats': {
                        'mean_obstacles_per_episode': float(np.mean(obstacle_counts)),
                        'min_obstacle_distances': [float(d) if d != np.inf else None for d in min_obstacle_distances],
                        'mean_min_obstacle_distance': float(np.mean(valid_distances)) if valid_distances else None,
                        'total_collision_risk_steps': int(np.sum(collision_risk_steps)),
                        'episodes_with_collision_risk': int(len([x for x in collision_risk_steps if x > 0]))
                    },
                    'individual_episodes': all_episode_summaries
                }
                
                summary_file = os.path.join(demo_folder, 'adaptive_obstacles_overall_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(overall_summary, f, indent=2)
                print(f"[INFO] Saved overall adaptive difficulty with obstacles summary to: {summary_file}")
            except Exception as e:
                print(f"[WARNING] Failed to save overall summary: {e}")

    print(f"\n[INFO] Adaptive difficulty demonstration with obstacles completed!")
    print(f"[INFO] Results saved to: {demo_folder}")
    return demo_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced demonstration of pre-trained adaptive difficulty drone model with obstacles')
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
    print("Adaptive Difficulty Drone Demonstration with Obstacles - Enhanced")
    print("="*60)
    print(f"Action type: {DEFAULT_ACT}")
    print(f"Multi-agent extractors: Available")
    print(f"Supports: TD3, SAC, PPO models")
    print(f"Adaptive difficulty: 0.1m -> 3.0m target radius")
    print(f"Obstacles: ENABLED with probability 1.0 (maximum density)")
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
        
        print(f"\n[SUCCESS] Adaptive difficulty demonstration with obstacles completed successfully!")
        print(f"[SUCCESS] Check results in: {demo_folder}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Demonstration interrupted by user")
        exit(0)
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)