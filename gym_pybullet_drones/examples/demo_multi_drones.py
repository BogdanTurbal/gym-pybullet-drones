#!/usr/bin/env python3
"""
demo_multi_rpm_enhanced.py: Load pre-trained multi-drone swarm model and run demonstration
Updated to work with multi-agent extractors and enhanced training script
"""
import os
import time
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Import your custom extractors (REQUIRED for model loading)
from multi_agent_extractors import (
    MultiAgentMatrixExtractor,
    MultiAgentSelfAttentionExtractor,
    MultiAgentMeanPoolExtractor,
    create_multiagent_ppo_model
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
DEFAULT_ACT           = ActionType('pid')  # Changed from 'rpm' to match training
DEFAULT_DRONES        = 1
DEFAULT_DURATION_SEC  = 6.0  # Match training script


class DemoMetricsCollector:
    """Enhanced metrics collection for demonstration analysis"""
    
    def __init__(self, num_drones, target_sequence):
        self.num_drones = num_drones
        self.target_sequence = target_sequence
        self.reset()
    
    def reset(self):
        """Reset all metrics for new episode"""
        self.episode_data = {
            'rewards': [],
            'phase_rewards': [],
            'distances': [],
            'actions': [],
            'positions': [],
            'targets_reached_per_phase': [],
            'collision_counts': [],
            'formation_errors': [],
            'velocities': [],
            'phase_transitions': [],
            'step_info': []
        }
        self.current_phase = -1
        self.total_reward = 0
        self.steps = 0
    
    def update(self, action, obs, reward, info, step):
        """Update metrics with current step data"""
        self.steps += 1
        self.total_reward += reward
        
        # Basic metrics
        self.episode_data['rewards'].append(reward)
        self.episode_data['actions'].append(action.copy() if hasattr(action, 'copy') else np.array(action))
        
        # Environment-specific metrics
        if 'phase' in info:
            phase = info['phase']
            if phase != self.current_phase:
                self.episode_data['phase_transitions'].append((step, phase))
                self.current_phase = phase
            
            # Ensure phase_rewards list is long enough
            while len(self.episode_data['phase_rewards']) <= phase:
                self.episode_data['phase_rewards'].append(0)
            self.episode_data['phase_rewards'][phase] += reward
        
        if 'distance_to_targets' in info:
            distances = info['distance_to_targets']
            self.episode_data['distances'].append({
                'mean': np.mean(distances),
                'individual': distances.copy(),
                'max': np.max(distances),
                'min': np.min(distances)
            })
        
        if 'targets_reached' in info:
            self.episode_data['targets_reached_per_phase'].append(np.sum(info['targets_reached']))
        
        if 'collision_count' in info:
            self.episode_data['collision_counts'].append(info['collision_count'])
        
        if 'formation_error' in info:
            self.episode_data['formation_errors'].append(info['formation_error'])
        
        # Store step info for detailed analysis
        self.episode_data['step_info'].append({
            'step': step,
            'phase': info.get('phase', -1),
            'reward': reward,
            'targets_reached': np.sum(info.get('targets_reached', [])),
            'mean_distance': np.mean(info.get('distance_to_targets', [0]))
        })
    
    def get_summary(self):
        """Get comprehensive episode summary"""
        if not self.episode_data['rewards']:
            return {}
        
        summary = {
            'total_reward': self.total_reward,
            'total_steps': self.steps,
            'mean_reward_per_step': self.total_reward / self.steps if self.steps > 0 else 0,
            'num_phases_completed': len(self.episode_data['phase_rewards']),
            'phase_rewards': self.episode_data['phase_rewards'].copy(),
        }
        
        # Distance statistics
        if self.episode_data['distances']:
            mean_distances = [d['mean'] for d in self.episode_data['distances']]
            summary.update({
                'distance_stats': {
                    'final': mean_distances[-1] if mean_distances else 0,
                    'mean': np.mean(mean_distances),
                    'min_achieved': np.min(mean_distances),
                    'max': np.max(mean_distances),
                    'std': np.std(mean_distances),
                    'improvement': mean_distances[0] - mean_distances[-1] if len(mean_distances) > 1 else 0
                }
            })
        
        # Performance metrics
        if self.episode_data['targets_reached_per_phase']:
            summary['targets_reached_stats'] = {
                'total': np.sum(self.episode_data['targets_reached_per_phase']),
                'max_simultaneous': np.max(self.episode_data['targets_reached_per_phase']),
                'mean': np.mean(self.episode_data['targets_reached_per_phase'])
            }
        
        # Collision and formation metrics
        if self.episode_data['collision_counts']:
            summary['collision_stats'] = {
                'total': np.sum(self.episode_data['collision_counts']),
                'final': self.episode_data['collision_counts'][-1]
            }
        
        if self.episode_data['formation_errors']:
            summary['formation_stats'] = {
                'mean_error': np.mean(self.episode_data['formation_errors']),
                'final_error': self.episode_data['formation_errors'][-1],
                'min_error': np.min(self.episode_data['formation_errors'])
            }
        
        return summary


def create_target_sequence(num_drones=4, scale=1.2):
    """Create target sequence matching the training script"""
    if num_drones == 4:
        # Use the same target sequence as in training
        targets = np.array([
            # Simple target: all drones go to same point to start
            [[ scale,  scale, 1.0]],
            
            [[ -scale,  scale, 2.0]],
            
            [[ scale,  -scale, 1.5]],
            
            [[ -scale,  -scale, 0.5]],
        ])
        # targets = np.array([
        #     # Phase 0: Simple horizontal line (easiest formation)
        #     [[-1.5*scale, 0.0, 1.2], [-0.5*scale, 0.0, 1.2], 
        #      [ 0.5*scale, 0.0, 1.2], [ 1.5*scale, 0.0, 1.2]],
            
        #     # Phase 1: Wide square formation  
        #     [[-scale, -scale, 1.2], [ scale, -scale, 1.2], 
        #      [ scale,  scale, 1.2], [-scale,  scale, 1.2]],
            
        #     # Phase 2: Diamond formation (45° rotation)
        #     [[ 0.0, -1.4*scale, 1.4], [ 1.4*scale, 0.0, 1.4], 
        #      [ 0.0,  1.4*scale, 1.4], [-1.4*scale, 0.0, 1.4]],
            
        #     # Phase 3: Tight square formation (precision training)
        #     [[-0.5*scale, -0.5*scale, 1.0], [ 0.5*scale, -0.5*scale, 1.0], 
        #      [ 0.5*scale,  0.5*scale, 1.0], [-0.5*scale,  0.5*scale, 1.0]]
        # ])
        # targets = np.array([
        #     # Simple target: all drones go to same point to start
        #     [[ scale,  scale, 1.0], [ scale,  scale, 1.0],
        #      [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            
        #     [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
        #      [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            
        #     [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
        #      [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            
        #     [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
        #      [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
        # ])
        
        # Alternative: Use the commented formation sequence from training
        # Uncomment this if you want more complex formations
        # targets = np.array([
        #     # Phase 0: Simple horizontal line (easiest formation)
        #     [[-1.5*scale, 0.0, 1.2], [-0.5*scale, 0.0, 1.2], 
        #      [ 0.5*scale, 0.0, 1.2], [ 1.5*scale, 0.0, 1.2]],
            
        #     # Phase 1: Wide square formation  
        #     [[-scale, -scale, 1.2], [ scale, -scale, 1.2], 
        #      [ scale,  scale, 1.2], [-scale,  scale, 1.2]],
            
        #     # Phase 2: Diamond formation (45° rotation)
        #     [[ 0.0, -1.4*scale, 1.4], [ 1.4*scale, 0.0, 1.4], 
        #      [ 0.0,  1.4*scale, 1.4], [-1.4*scale, 0.0, 1.4]],
            
        #     # Phase 3: Tight square formation (precision training)
        #     [[-0.5*scale, -0.5*scale, 1.0], [ 0.5*scale, -0.5*scale, 1.0], 
        #      [ 0.5*scale,  0.5*scale, 1.0], [-0.5*scale,  0.5*scale, 1.0]]
        # ])
    else:
        # Create circular formations for other numbers of drones
        targets = []
        n_phases = 4
        for phase in range(n_phases):
            phase_targets = []
            radius = scale * (1.0 + 0.2 * phase)
            height = 1.5 + 0.2 * phase
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones + phase * np.pi / 4
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                phase_targets.append([x, y, height])
            targets.append(phase_targets)
        targets = np.array(targets)
    
    return targets.astype(np.float32)


def create_enhanced_plots(metrics_collector, output_folder, episode_num):
    """Create comprehensive analysis plots"""
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
            ax1.plot(episode_data['rewards'], alpha=0.7, label='Step Reward')
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
        
        # Plot 2: Distance to targets
        ax2 = fig.add_subplot(gs[0, 1])
        if episode_data['distances']:
            mean_distances = [d['mean'] for d in episode_data['distances']]
            min_distances = [d['min'] for d in episode_data['distances']]
            max_distances = [d['max'] for d in episode_data['distances']]
            
            steps = range(len(mean_distances))
            ax2.plot(steps, mean_distances, label='Mean Distance', linewidth=2)
            ax2.fill_between(steps, min_distances, max_distances, alpha=0.3, label='Min-Max Range')
            ax2.set_title('Distance to Targets')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Distance')
            ax2.legend()
            ax2.grid(True)
        
        # Plot 3: Phase rewards
        ax3 = fig.add_subplot(gs[0, 2])
        if summary.get('phase_rewards'):
            phase_rewards = summary['phase_rewards']
            bars = ax3.bar(range(len(phase_rewards)), phase_rewards, alpha=0.7)
            ax3.set_title('Rewards per Phase')
            ax3.set_xlabel('Phase')
            ax3.set_ylabel('Total Reward')
            ax3.grid(True, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Plot 4: Targets reached over time
        ax4 = fig.add_subplot(gs[0, 3])
        if episode_data['targets_reached_per_phase']:
            ax4.plot(episode_data['targets_reached_per_phase'], marker='o', markersize=4)
            ax4.set_title('Targets Reached Over Time')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Targets Reached')
            ax4.grid(True)
        
        # Plot 5: Distance distribution
        ax5 = fig.add_subplot(gs[1, 0])
        if episode_data['distances']:
            mean_distances = [d['mean'] for d in episode_data['distances']]
            ax5.hist(mean_distances, bins=30, alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(mean_distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mean_distances):.3f}')
            ax5.set_title('Distance Distribution')
            ax5.set_xlabel('Distance to Targets')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            ax5.grid(True)
        
        # Plot 6: Phase transitions
        ax6 = fig.add_subplot(gs[1, 1])
        if episode_data['phase_transitions']:
            transitions = episode_data['phase_transitions']
            steps, phases = zip(*transitions)
            ax6.step(steps, phases, where='post', marker='o', markersize=6)
            ax6.set_title('Phase Transitions')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Phase')
            ax6.grid(True)
        
        # Plot 7: Formation errors (if available)
        ax7 = fig.add_subplot(gs[1, 2])
        if episode_data['formation_errors']:
            ax7.plot(episode_data['formation_errors'], alpha=0.7)
            ax7.set_title('Formation Error Over Time')
            ax7.set_xlabel('Steps')
            ax7.set_ylabel('Formation Error')
            ax7.grid(True)
        
        # Plot 8: Action analysis - Updated for PID actions
        ax8 = fig.add_subplot(gs[1, 3])
        if episode_data['actions']:
            actions = np.array(episode_data['actions'])
            if actions.ndim > 1 and actions.shape[1] > 0:
                # Plot action variance for first drone (4 PID values)
                if actions.ndim == 3:  # [time, drone, action]
                    drone_0_actions = actions[:, 0, :]  # First drone's actions over time
                    action_var = np.var(drone_0_actions, axis=1)
                elif actions.ndim == 2:  # [time, all_actions]
                    # Assume first 4 actions are for drone 0
                    drone_0_actions = actions[:, :4]
                    action_var = np.var(drone_0_actions, axis=1)
                else:
                    action_var = np.var(actions, axis=0 if actions.ndim > 1 else None)
                
                ax8.plot(action_var, alpha=0.7)
                ax8.set_title('Action Variance (Drone 0)')
                ax8.set_xlabel('Steps')
                ax8.set_ylabel('Action Variance')
                ax8.grid(True)
        
        # Plot 9: Performance summary (text)
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.axis('off')
        
        # Create summary text
        summary_text = f"""
        DEMONSTRATION SUMMARY - Episode {episode_num}
        
        Overall Performance:
        • Total Reward: {summary.get('total_reward', 0):.2f}
        • Total Steps: {summary.get('total_steps', 0)}
        • Mean Reward/Step: {summary.get('mean_reward_per_step', 0):.4f}
        • Phases Completed: {summary.get('num_phases_completed', 0)}
        
        Distance Performance:
        • Final Distance: {summary.get('distance_stats', {}).get('final', 0):.3f}
        • Mean Distance: {summary.get('distance_stats', {}).get('mean', 0):.3f}
        • Best Distance: {summary.get('distance_stats', {}).get('min_achieved', 0):.3f}
        • Distance Improvement: {summary.get('distance_stats', {}).get('improvement', 0):.3f}
        
        Target Tracking:
        • Total Targets Reached: {summary.get('targets_reached_stats', {}).get('total', 0)}
        • Max Simultaneous: {summary.get('targets_reached_stats', {}).get('max_simultaneous', 0)}
        • Mean Targets/Step: {summary.get('targets_reached_stats', {}).get('mean', 0):.2f}
        """
        
        if 'collision_stats' in summary:
            summary_text += f"\n        Safety:\n        • Total Collisions: {summary['collision_stats'].get('total', 0)}"
        
        if 'formation_stats' in summary:
            summary_text += f"\n        • Formation Error: {summary['formation_stats'].get('mean_error', 0):.3f}"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Plot 10: Score gauge
        ax10 = fig.add_subplot(gs[2, 2:])
        
        # Calculate overall score (0-100)
        distance_score = max(0, 100 * (1 - summary.get('distance_stats', {}).get('mean', 1)))
        reward_score = max(0, min(100, summary.get('total_reward', 0)))
        target_score = summary.get('targets_reached_stats', {}).get('mean', 0) * 25
        overall_score = (distance_score + reward_score + target_score) / 3
        
        # Create gauge plot
        categories = ['Distance\nPerformance', 'Reward\nScore', 'Target\nTracking', 'Overall\nScore']
        scores = [distance_score, reward_score, target_score, overall_score]
        colors = ['green' if s > 70 else 'orange' if s > 40 else 'red' for s in scores]
        
        bars = ax10.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
        ax10.set_ylim(0, 100)
        ax10.set_title('Performance Scores')
        ax10.set_ylabel('Score (0-100)')
        ax10.grid(True, axis='y')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Multi-Drone Swarm Demonstration Analysis - Episode {episode_num}', 
                     fontsize=16, fontweight='bold')
        
        # Save plot
        plot_path = os.path.join(output_folder, f'demo_episode_{episode_num}_enhanced_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved enhanced analysis plot: {plot_path}")
        plt.close()
        
        return plot_path
        
    except ImportError:
        print("[WARNING] Matplotlib not available, skipping enhanced plots")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to generate enhanced plots: {e}")
        return None


def run_demonstration(model_path, output_folder, gui, record_video, plot, num_episodes=1, 
                     detailed_analysis=True, save_metrics=True):
    """
    Load pre-trained model and run enhanced demonstration
    
    Args:
        model_path: Path to the saved PPO model (.zip file)
        output_folder: Folder to save demonstration results
        gui: Whether to show PyBullet GUI
        record_video: Whether to record video
        plot: Whether to generate plots
        num_episodes: Number of demonstration episodes to run
        detailed_analysis: Whether to perform detailed analysis
        save_metrics: Whether to save metrics to files
    """
    
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.zip'):
        print("[WARNING] Model path should typically end with .zip extension")
    
    print(f"[INFO] Loading model from: {model_path}")
    print(f"[INFO] Running {num_episodes} demonstration episode(s)")
    print(f"[INFO] Using action type: {DEFAULT_ACT} (PID control)")
    print(f"[INFO] Multi-agent extractors imported and available")
    
    # Prepare output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    demo_folder = os.path.join(output_folder, f'demo_{timestamp}')
    os.makedirs(demo_folder, exist_ok=True)

    # Get control frequency
    try:
        dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        freq = int(dummy_env.CTRL_FREQ)
        dummy_env.close()
    except Exception as e:
        print(f"[ERROR] Failed to create dummy environment: {e}")
        return

    # Create target sequence matching training script
    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=2.0)  # Use scale=1.0 as in training
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] Target sequence shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Total episode length: {len(target_sequence) * steps_per_target} steps")
    print(f"[INFO] Control frequency: {freq} Hz")

    # Load the trained model with error handling
    # The model should load correctly since we imported the custom extractors
    try:
        model = PPO.load(model_path)
        print("[INFO] Model loaded successfully!")
        print(f"[INFO] Model policy: {type(model.policy).__name__}")
        
        # Check if it has custom features extractor
        if hasattr(model.policy, 'features_extractor'):
            print(f"[INFO] Features extractor: {type(model.policy.features_extractor).__name__}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[ERROR] Make sure multi_agent_extractors.py is available and importable")
        raise RuntimeError(f"Failed to load model: {e}")

    # Create evaluation environment for quick assessment
    try:
        eval_env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=False,
            record=False
        )
        
        print('[INFO] Action space:', eval_env.action_space)
        print('[INFO] Observation space:', eval_env.observation_space)
        
        # Verify observation shape matches training expectations
        obs_shape = eval_env.observation_space.shape
        action_buffer_size = 0  # As set in training script
        expected_base_features = 12 + (action_buffer_size * 4)
        expected_total_features = expected_base_features + 8
        expected_shape = (DEFAULT_DRONES, expected_total_features)
        
        print(f'[INFO] Expected observation shape: {expected_shape}')
        print(f'[INFO] Actual observation shape: {obs_shape}')
        
        # Quick evaluation
        print(f"[INFO] Running quick evaluation...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=3, deterministic=True
        )
        print(f"[EVAL] Quick assessment - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        eval_env.close()
        
    except Exception as e:
        print(f"[WARNING] Failed to run quick evaluation: {e}")

    # Store all episode results
    all_episode_summaries = []
    all_episode_metrics = []

    # Run demonstration episodes
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"[INFO] Running demonstration episode {episode + 1}/{num_episodes}...")
        print(f"{'='*60}")
        
        try:
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
            
            # Initialize metrics collector
            metrics_collector = DemoMetricsCollector(DEFAULT_DRONES, target_sequence)
            
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
            obs, info = demo_env.reset(seed=42 + episode)
            start_time = time.time()
            
            max_steps = len(target_sequence) * steps_per_target
            print(f"[INFO] Running for maximum {max_steps} steps...")
            
            # Main demonstration loop
            for i in range(max_steps + 100):  # Add buffer steps
                try:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, done, truncated, info = demo_env.step(action)
                    
                    # Update metrics
                    metrics_collector.update(action, obs, reward, info, i)
                    
                    # Render if GUI enabled
                    if gui:
                        demo_env.render()
                        sync(i, start_time, demo_env.CTRL_TIMESTEP)
                    
                    # Log progress every 100 steps
                    if i % 100 == 0:
                        phase = info.get('phase', -1)
                        targets_reached = np.sum(info.get('targets_reached', []))
                        dist = np.mean(info.get('distance_to_targets', [0]))
                        print(f"Step {i:4d} | Phase {phase:2d} | Targets: {targets_reached} | "
                              f"Distance: {dist:.3f} | Reward: {reward:6.3f} | "
                              f"Total: {metrics_collector.total_reward:6.2f}")
                    
                    # Optional logging for trajectory plotting
                    if plot and logger and DEFAULT_OBS == ObservationType.KIN:
                        try:
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
                                
                                # Handle PID actions (4 values per drone)
                                if act_arr.ndim > 1 and act_arr.shape[0] >= DEFAULT_DRONES * 4:
                                    # Action array is [drone0_pid[4], drone1_pid[4], ...]
                                    drone_action = act_arr[d*4:(d+1)*4]
                                elif act_arr.ndim == 1 and len(act_arr) == 4:
                                    # Single drone or all drones same action
                                    drone_action = act_arr
                                else:
                                    drone_action = [0, 0, 0, 0]
                                
                                state = np.hstack([
                                    pos,                    # position
                                    np.zeros(4),           # quaternion placeholder
                                    vel_etc,               # velocity, etc.
                                    drone_action           # PID action (4 values)
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
                        print(f"[INFO] Episode {episode + 1} finished at step {i}")
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
            print(f"  • Total Reward: {episode_summary.get('total_reward', 0):.2f}")
            print(f"  • Total Steps: {episode_summary.get('total_steps', 0)}")
            print(f"  • Phases Completed: {episode_summary.get('num_phases_completed', 0)}")
            
            if 'distance_stats' in episode_summary:
                dist_stats = episode_summary['distance_stats']
                print(f"  • Final Distance: {dist_stats.get('final', 0):.3f}")
                print(f"  • Mean Distance: {dist_stats.get('mean', 0):.3f}")
                print(f"  • Best Distance: {dist_stats.get('min_achieved', 0):.3f}")
                print(f"  • Distance Improvement: {dist_stats.get('improvement', 0):.3f}")
            
            if 'targets_reached_stats' in episode_summary:
                target_stats = episode_summary['targets_reached_stats']
                print(f"  • Targets Reached: {target_stats.get('total', 0)}")
                print(f"  • Max Simultaneous: {target_stats.get('max_simultaneous', 0)}")
            
            # Save metrics to file if requested
            if save_metrics:
                try:
                    import json
                    metrics_file = os.path.join(demo_folder, f'episode_{episode + 1}_metrics.json')
                    
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
                    print(f"[INFO] Saved metrics to: {metrics_file}")
                except Exception as e:
                    print(f"[WARNING] Failed to save metrics: {e}")
            
            # Generate enhanced plots for this episode
            if plot and detailed_analysis:
                plot_path = create_enhanced_plots(metrics_collector, demo_folder, episode + 1)
                if plot_path:
                    print(f"[INFO] Generated enhanced analysis plots")
            
            # Generate trajectory plots using logger
            if plot and logger and DEFAULT_OBS == ObservationType.KIN:
                try:
                    logger.plot()
                    print(f"[INFO] Generated trajectory plots in: {demo_folder}")
                except Exception as e:
                    print(f"[WARNING] Failed to generate trajectory plots: {e}")
            
            demo_env.close()
            
        except Exception as e:
            print(f"[ERROR] Episode {episode + 1} failed: {e}")
            continue

    # Generate overall summary if multiple episodes
    if num_episodes > 1 and all_episode_summaries:
        print(f"\n{'='*60}")
        print("[INFO] OVERALL DEMONSTRATION SUMMARY")
        print(f"{'='*60}")
        
        # Calculate aggregate statistics
        total_rewards = [ep.get('total_reward', 0) for ep in all_episode_summaries]
        mean_distances = [ep.get('distance_stats', {}).get('mean', 0) for ep in all_episode_summaries]
        
        print(f"Episodes: {len(all_episode_summaries)}")
        print(f"Reward - Mean: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Reward - Range: [{np.min(total_rewards):.2f}, {np.max(total_rewards):.2f}]")
        print(f"Distance - Mean: {np.mean(mean_distances):.3f} ± {np.std(mean_distances):.3f}")
        print(f"Distance - Range: [{np.min(mean_distances):.3f}, {np.max(mean_distances):.3f}]")
        
        # Save overall summary
        if save_metrics:
            try:
                import json
                overall_summary = {
                    'num_episodes': len(all_episode_summaries),
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
                    'individual_episodes': all_episode_summaries
                }
                
                summary_file = os.path.join(demo_folder, 'overall_summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(overall_summary, f, indent=2)
                print(f"[INFO] Saved overall summary to: {summary_file}")
            except Exception as e:
                print(f"[WARNING] Failed to save overall summary: {e}")

    print(f"\n[INFO] Demonstration completed!")
    print(f"[INFO] Results saved to: {demo_folder}")
    return demo_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced demonstration of pre-trained multi-agent drone swarm model')
    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to the trained PPO model (.zip file)')
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
    
    args = parser.parse_args()

    # Validate arguments
    if not args.model_path:
        print("[ERROR] Model path is required!")
        exit(1)
    
    if args.num_episodes < 1:
        print("[ERROR] Number of episodes must be at least 1!")
        exit(1)
    
    print("="*60)
    print("Multi-Agent Drone Swarm Demonstration - Enhanced")
    print("="*60)
    print(f"Action type: {DEFAULT_ACT} (PID control)")
    print(f"Multi-agent extractors: Available")
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
            args.save_metrics
        )
        
        print(f"\n[SUCCESS] Demonstration completed successfully!")
        print(f"[SUCCESS] Check results in: {demo_folder}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Demonstration interrupted by user")
        exit(0)
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)