#!/usr/bin/env python3
"""
practical_algorithm_comparison.py - Compare SAC vs PPO for your multi-agent drone racing
A practical implementation that works with your existing setup
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gymnasium as gym
import wandb
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# Import your existing components
from multi_agent_extractors import (
    MultiAgentMatrixExtractor,
    MultiAgentSelfAttentionExtractor,
    MultiAgentMeanPoolExtractor,
    create_multiagent_ppo_model
)
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Configuration
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'algorithm_comparison_results'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_DRONES = 4
DEFAULT_DURATION_SEC = 3.0
NUM_VEC = 1


class AlgorithmComparisonCallback(BaseCallback):
    """Callback to track and compare algorithm performance"""
    def __init__(self, algorithm_name, save_freq=50000, min_episodes=20, verbose=0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.save_freq = save_freq
        self.min_episodes = min_episodes
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []  # Track final distances to targets
        self.episode_targets_reached = []  # Track targets reached per episode
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.best_mean_reward = -np.inf
        self.models_saved = 0
        
        # Performance metrics
        self.convergence_timestep = None  # When did it converge?
        self.stability_window = 50  # Episodes to check for stability
        
    def _on_step(self) -> bool:
        # Track episode progress
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            # Store episode data
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Get additional metrics from info if available
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                final_distance = info.get('mean_distance_to_targets', 0)
                targets_reached = np.sum(info.get('targets_reached', []))
                self.episode_distances.append(final_distance)
                self.episode_targets_reached.append(targets_reached)
            
            # Calculate performance metrics
            if len(self.episode_rewards) >= self.min_episodes:
                recent_rewards = self.episode_rewards[-self.min_episodes:]
                mean_reward = np.mean(recent_rewards)
                reward_std = np.std(recent_rewards)
                
                # Check for convergence (stability in performance)
                if (len(self.episode_rewards) >= self.stability_window and
                    self.convergence_timestep is None and
                    reward_std < 50 and mean_reward > -100):  # Algorithm-specific thresholds
                    self.convergence_timestep = self.num_timesteps
                    if self.verbose > 0:
                        print(f"[{self.algorithm_name}] Converged at timestep {self.convergence_timestep}")
                
                # Log detailed metrics
                wandb.log({
                    f'{self.algorithm_name}/episode_reward': self.current_episode_reward,
                    f'{self.algorithm_name}/mean_reward_{self.min_episodes}ep': mean_reward,
                    f'{self.algorithm_name}/reward_std_{self.min_episodes}ep': reward_std,
                    f'{self.algorithm_name}/episode_length': self.current_episode_length,
                    f'{self.algorithm_name}/episodes_completed': len(self.episode_rewards),
                    f'{self.algorithm_name}/convergence_timestep': self.convergence_timestep or 0,
                }, step=self.num_timesteps)
                
                # Additional metrics if available
                if self.episode_distances:
                    recent_distances = self.episode_distances[-self.min_episodes:]
                    wandb.log({
                        f'{self.algorithm_name}/mean_final_distance': np.mean(recent_distances),
                        f'{self.algorithm_name}/distance_improvement': recent_distances[0] - recent_distances[-1],
                    }, step=self.num_timesteps)
                
                if self.episode_targets_reached:
                    recent_targets = self.episode_targets_reached[-self.min_episodes:]
                    wandb.log({
                        f'{self.algorithm_name}/mean_targets_reached': np.mean(recent_targets),
                        f'{self.algorithm_name}/success_rate': np.mean(np.array(recent_targets) >= 2),  # At least 2 targets
                    }, step=self.num_timesteps)
                
                # Save best model
                if mean_reward > self.best_mean_reward and len(self.episode_rewards) >= 50:
                    self.best_mean_reward = mean_reward
                    self.models_saved += 1
                    
                    if self.verbose > 0:
                        print(f"[{self.algorithm_name}] New best mean reward: {mean_reward:.2f}")
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if len(self.episode_rewards) < self.min_episodes:
            return None
            
        recent_rewards = self.episode_rewards[-self.min_episodes:]
        
        summary = {
            'algorithm': self.algorithm_name,
            'episodes': len(self.episode_rewards),
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'best_reward': np.max(self.episode_rewards),
            'convergence_timestep': self.convergence_timestep,
            'sample_efficiency': self.convergence_timestep / 1000 if self.convergence_timestep else float('inf'),
            'stability': 1.0 / (1.0 + np.std(recent_rewards)),  # Higher is more stable
        }
        
        if self.episode_distances:
            recent_distances = self.episode_distances[-self.min_episodes:]
            summary.update({
                'mean_final_distance': np.mean(recent_distances),
                'distance_improvement': recent_distances[0] - recent_distances[-1],
            })
        
        if self.episode_targets_reached:
            recent_targets = self.episode_targets_reached[-self.min_episodes:]
            summary.update({
                'mean_targets_reached': np.mean(recent_targets),
                'success_rate': np.mean(np.array(recent_targets) >= 2),
            })
        
        return summary


def create_sac_model(env, features_dim=128, learning_rate=3e-4):
    """Create SAC model with custom multi-agent extractor"""
    
    # Custom policy for SAC with multi-agent features
    policy_kwargs = {
        "features_extractor_class": MultiAgentMatrixExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": [128, 128],  # Actor and critic network sizes
    }
    
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        batch_size=256,
        gamma=0.99,
        tau=0.01,  # Slightly higher for stability with multi-agent
        ent_coef='auto',  # Automatic entropy tuning
        target_update_interval=1,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,  # Longer warm-up for multi-agent
        buffer_size=int(1e6),
        verbose=1,
        device='auto'
    )
    
    return model


def create_ppo_model(env, features_dim=128, learning_rate=3e-4):
    """Create PPO model with your existing multi-agent extractor"""
    
    model = create_multiagent_ppo_model(
        env=env,
        extractor_type='matrix',
        features_dim=features_dim,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=512,  # Larger batch for stability
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    
    return model


def create_target_sequence_for_comparison(num_drones=4):
    """Create a target sequence optimized for algorithm comparison"""
    # Simpler sequence to make algorithm differences more apparent
    targets = np.array([
        # Phase 0: Line formation (tests basic coordination)
        [[-1.5, 0.0, 1.2], [-0.5, 0.0, 1.2], [0.5, 0.0, 1.2], [1.5, 0.0, 1.2]],
        
        # Phase 1: Square formation (tests precision)
        [[-1.0, -1.0, 1.5], [1.0, -1.0, 1.5], [1.0, 1.0, 1.5], [-1.0, 1.0, 1.5]],
        
        # Phase 2: Diamond formation (tests coordination)
        [[0.0, -1.5, 1.8], [1.5, 0.0, 1.8], [0.0, 1.5, 1.8], [-1.5, 0.0, 1.8]],
    ])
    
    return targets.astype(np.float32)


def run_algorithm_comparison(output_folder, wandb_project, wandb_entity, total_timesteps=500000):
    """Main function to compare SAC vs PPO"""
    
    run_name = f"sac_vs_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Consistent hyperparameters for both algorithms
    paper_reward_hyperparameters = {
        'lambda_1': 15.0,    # Progress reward
        'lambda_2': 1.0,     # Perception reward
        'lambda_3': 0.3,     # Perception alignment
        'lambda_4': 0.01,    # Action magnitude penalty
        'lambda_5': 0.05,    # Action smoothness penalty
        'crash_penalty': 10.0,
        'bounds_penalty': 5.0,
    }
    
    config = {
        'comparison': 'SAC_vs_PPO',
        'total_timesteps': total_timesteps,
        'paper_reward_hyperparameters': paper_reward_hyperparameters,
        'num_drones': DEFAULT_DRONES,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,
        'duration_sec': DEFAULT_DURATION_SEC,
        'features_dim': 128,
        'learning_rate': 3e-4,
    }
    
    # Initialize WandB
    wandb.finish()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        tags=["algorithm-comparison", "sac-vs-ppo", "multi-agent-drones"],
        notes="Comparing SAC vs PPO for multi-agent drone racing with paper-based rewards"
    )

    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Environment setup
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    target_sequence = create_target_sequence_for_comparison(DEFAULT_DRONES)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"\n{'='*80}")
    print("SAC vs PPO Algorithm Comparison for Multi-Agent Drone Racing")
    print(f"{'='*80}")
    print(f"Total timesteps per algorithm: {total_timesteps:,}")
    print(f"Target sequence shape: {target_sequence.shape}")
    print(f"Paper-based reward function: Enabled")
    print(f"Results will be saved to: {save_dir}")
    print(f"{'='*80}")

    # Environment factory
    def make_env():
        return Monitor(MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=False,
            record=False,
            **paper_reward_hyperparameters
        ))

    # Shared evaluation environment
    eval_env = MultiTargetAviary(
        num_drones=DEFAULT_DRONES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        target_sequence=target_sequence,
        steps_per_target=steps_per_target,
        gui=False,
        record=False,
        **paper_reward_hyperparameters
    )

    results = {}
    
    # TRAIN SAC
    print(f"\n🚀 Training SAC Algorithm...")
    try:
        sac_start = time.time()
        
        # Create SAC environment and model
        sac_env = make_vec_env(make_env, n_envs=NUM_VEC, seed=42)
        sac_model = create_sac_model(sac_env, features_dim=128, learning_rate=3e-4)
        
        # SAC callback
        sac_callback = AlgorithmComparisonCallback('SAC', verbose=1)
        sac_eval_callback = EvalCallback(
            eval_env, best_model_save_path=os.path.join(save_dir, 'sac_best'),
            log_path=os.path.join(save_dir, 'sac_logs'),
            eval_freq=25000, n_eval_episodes=5, deterministic=True
        )
        sac_callbacks = CallbackList([sac_callback, sac_eval_callback])
        
        # Train SAC
        sac_model.learn(
            total_timesteps=total_timesteps,
            callback=sac_callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        sac_training_time = time.time() - sac_start
        
        # Evaluate SAC
        sac_mean_reward, sac_std_reward = evaluate_policy(
            sac_model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        # Get performance summary
        sac_summary = sac_callback.get_performance_summary()
        sac_summary.update({
            'final_mean_reward': sac_mean_reward,
            'final_std_reward': sac_std_reward,
            'training_time': sac_training_time
        })
        
        results['SAC'] = sac_summary
        
        print(f"✅ SAC Training Complete!")
        print(f"   Final Performance: {sac_mean_reward:.2f} ± {sac_std_reward:.2f}")
        print(f"   Training Time: {sac_training_time:.1f} seconds")
        
        sac_env.close()
        
    except Exception as e:
        print(f"❌ SAC Training Failed: {e}")
        results['SAC'] = {'error': str(e)}

    # TRAIN PPO
    print(f"\n🚀 Training PPO Algorithm...")
    try:
        ppo_start = time.time()
        
        # Create PPO environment and model
        ppo_env = make_vec_env(make_env, n_envs=NUM_VEC, seed=42)
        ppo_model = create_ppo_model(ppo_env, features_dim=128, learning_rate=3e-4)
        
        # PPO callback
        ppo_callback = AlgorithmComparisonCallback('PPO', verbose=1)
        ppo_eval_callback = EvalCallback(
            eval_env, best_model_save_path=os.path.join(save_dir, 'ppo_best'),
            log_path=os.path.join(save_dir, 'ppo_logs'),
            eval_freq=25000, n_eval_episodes=5, deterministic=True
        )
        ppo_callbacks = CallbackList([ppo_callback, ppo_eval_callback])
        
        # Train PPO
        ppo_model.learn(
            total_timesteps=total_timesteps,
            callback=ppo_callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        ppo_training_time = time.time() - ppo_start
        
        # Evaluate PPO
        ppo_mean_reward, ppo_std_reward = evaluate_policy(
            ppo_model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        # Get performance summary
        ppo_summary = ppo_callback.get_performance_summary()
        ppo_summary.update({
            'final_mean_reward': ppo_mean_reward,
            'final_std_reward': ppo_std_reward,
            'training_time': ppo_training_time
        })
        
        results['PPO'] = ppo_summary
        
        print(f"✅ PPO Training Complete!")
        print(f"   Final Performance: {ppo_mean_reward:.2f} ± {ppo_std_reward:.2f}")
        print(f"   Training Time: {ppo_training_time:.1f} seconds")
        
        ppo_env.close()
        
    except Exception as e:
        print(f"❌ PPO Training Failed: {e}")
        results['PPO'] = {'error': str(e)}

    # COMPARISON ANALYSIS
    print(f"\n{'='*80}")
    print("🏆 ALGORITHM COMPARISON RESULTS")
    print(f"{'='*80}")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(valid_results) >= 2:
        # Determine winner based on multiple criteria
        comparison_metrics = {}
        
        for alg_name, result in valid_results.items():
            score = 0
            
            # Performance score (40% weight)
            score += result.get('final_mean_reward', -1000) * 0.4
            
            # Sample efficiency score (30% weight) - lower convergence timestep is better
            convergence = result.get('convergence_timestep', float('inf'))
            if convergence != float('inf'):
                efficiency_score = max(0, 100000 - convergence) / 1000  # Normalize
                score += efficiency_score * 0.3
            
            # Stability score (20% weight)
            stability = result.get('stability', 0)
            score += stability * 100 * 0.2  # Normalize stability
            
            # Success rate (10% weight)
            success_rate = result.get('success_rate', 0)
            score += success_rate * 100 * 0.1
            
            comparison_metrics[alg_name] = score
        
        # Sort by composite score
        ranked_algorithms = sorted(comparison_metrics.items(), key=lambda x: x[1], reverse=True)
        
        print("📊 DETAILED COMPARISON:")
        print("-" * 80)
        
        for i, (alg_name, score) in enumerate(ranked_algorithms):
            result = valid_results[alg_name]
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            
            print(f"{rank_emoji} {alg_name} (Score: {score:.1f})")
            print(f"   📈 Final Performance: {result.get('final_mean_reward', 0):.2f} ± {result.get('final_std_reward', 0):.2f}")
            print(f"   ⚡ Sample Efficiency: {result.get('convergence_timestep', 'N/A')} timesteps")
            print(f"   📊 Stability: {result.get('stability', 0):.3f}")
            print(f"   🎯 Success Rate: {result.get('success_rate', 0):.2%}")
            print(f"   ⏱️  Training Time: {result.get('training_time', 0):.1f} seconds")
            print()
        
        # Winner announcement
        winner = ranked_algorithms[0][0]
        print(f"🏆 WINNER: {winner}")
        
        # Log comparison to WandB
        comparison_table = wandb.Table(
            columns=["Algorithm", "Final Reward", "Sample Efficiency", "Stability", "Success Rate", "Training Time", "Overall Score"],
            data=[[alg, 
                   valid_results[alg].get('final_mean_reward', 0),
                   valid_results[alg].get('convergence_timestep', 0),
                   valid_results[alg].get('stability', 0),
                   valid_results[alg].get('success_rate', 0),
                   valid_results[alg].get('training_time', 0),
                   score] for alg, score in ranked_algorithms]
        )
        wandb.log({"algorithm_comparison_detailed": comparison_table})
        
        # Summary metrics
        wandb.log({
            'comparison/winner': winner,
            'comparison/performance_gap': ranked_algorithms[0][1] - ranked_algorithms[1][1],
            'comparison/best_final_reward': max(r.get('final_mean_reward', -1000) for r in valid_results.values()),
            'comparison/best_convergence': min(r.get('convergence_timestep', float('inf')) for r in valid_results.values() if r.get('convergence_timestep')),
        })
        
    else:
        print("❌ Not enough successful runs for comparison")
    
    print(f"{'='*80}")
    
    eval_env.close()
    wandb.finish()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare SAC vs PPO for multi-agent drone racing')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Results folder')
    parser.add_argument('--wandb_project', default='drone-sac-vs-ppo', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='Weights & Biases entity/username')
    parser.add_argument('--timesteps', default=500000, type=int,
                        help='Training timesteps per algorithm')
    
    args = parser.parse_args()

    print("🤖 Multi-Agent Drone Racing: SAC vs PPO Comparison")
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"WandB project: {args.wandb_project}")
    
    results = run_algorithm_comparison(
        args.output_folder,
        args.wandb_project,
        args.wandb_entity,
        args.timesteps
    )
    
    print("\n✅ Comparison completed! Check WandB for detailed results.")