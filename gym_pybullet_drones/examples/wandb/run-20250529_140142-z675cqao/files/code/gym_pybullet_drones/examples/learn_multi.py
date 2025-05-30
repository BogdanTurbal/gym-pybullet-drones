#!/usr/bin/env python3
"""
train_multi_rpm_paper_based.py - UPDATED for paper-based reward function
Training script optimized for the new paper-based reward formulation
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gymnasium as gym
import wandb
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# Import your custom extractors
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

# Updated settings for paper-based reward function
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('rpm')  # RPM actions as in paper
DEFAULT_DRONES        = 4
DEFAULT_DURATION_SEC  = 3.0
NUM_VEC = 1

# Multi-agent architecture settings
DEFAULT_EXTRACTOR_TYPE = "matrix"
DEFAULT_FEATURES_DIM = 128  # Slightly smaller for faster training with new reward


class PaperBasedWandbCallback(BaseCallback):
    """Enhanced callback for paper-based reward function with detailed component logging"""
    def __init__(self, log_freq=100, save_freq=25000, min_reward_improvement=5.0, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.min_reward_improvement = min_reward_improvement
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.best_mean_reward = -np.inf
        
        # Track reward components
        self.progress_rewards = []
        self.perception_rewards = []
        self.command_rewards = []
        self.crash_penalties = []
        self.target_bonuses = []
        
        # Track when we last saved a model
        self.last_saved_reward = -np.inf
        self.models_saved = 0
        
        if verbose > 0:
            print(f"[PaperBasedWandbCallback] Tracking paper-based reward components")
            print(f"[PaperBasedWandbCallback] Minimum reward improvement: {min_reward_improvement}")
        
    def _on_step(self) -> bool:
        # Get reward and update episode tracking
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            mean_reward_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0
            
            wandb.log({
                'train/episode_reward': self.current_episode_reward,
                'train/episode_length': self.current_episode_length,
                'train/mean_episode_reward_last_100': mean_reward_100,
                'train/episodes_completed': len(self.episode_rewards),
                'train/best_mean_reward': self.best_mean_reward,
                'train/reward_improvement_needed': max(0, self.best_mean_reward + self.min_reward_improvement - mean_reward_100),
            }, step=self.num_timesteps)
            
            # Save model if improvement exceeds minimum threshold
            reward_improvement = mean_reward_100 - self.best_mean_reward
            should_save = (
                mean_reward_100 > self.best_mean_reward and
                reward_improvement >= self.min_reward_improvement and
                len(self.episode_rewards) >= 10
            )
            
            if should_save:
                self.best_mean_reward = mean_reward_100
                self.last_saved_reward = mean_reward_100
                self.models_saved += 1
                
                self._save_model_artifact(f"paper_reward_model_{mean_reward_100:.2f}_improvement_{reward_improvement:.2f}")
                
                if self.verbose > 0:
                    print(f"\n[Model Saved] Timestep {self.num_timesteps}: "
                          f"Paper-based reward improved by {reward_improvement:.3f} "
                          f"(from {self.best_mean_reward - reward_improvement:.2f} to {mean_reward_100:.2f})")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log detailed metrics periodically
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                
                log_dict = {
                    'train/timestep': self.num_timesteps,
                    'train/current_reward': reward,
                    'train/models_saved_count': self.models_saved,
                }
                
                # Add environment-specific metrics
                if 'phase' in info:
                    log_dict['train/current_phase'] = info['phase']
                if 'targets_reached' in info:
                    log_dict['train/targets_reached'] = np.sum(info['targets_reached'])
                if 'collision_count' in info:
                    log_dict['train/collision_count'] = info['collision_count']
                if 'mean_distance_to_targets' in info:
                    log_dict['train/mean_distance_to_targets'] = info['mean_distance_to_targets']
                if 'formation_error' in info:
                    log_dict['train/formation_error'] = info['formation_error']
                
                wandb.log(log_dict, step=self.num_timesteps)
        
        # Save model periodically as artifacts
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_model_artifact(f"paper_reward_checkpoint_step_{self.num_timesteps}")
            
        return True
    
    def _save_model_artifact(self, artifact_name):
        """Save model as WandB artifact"""
        try:
            temp_path = f"/tmp/{artifact_name}.zip"
            self.model.save(temp_path)
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"PPO model with paper-based reward at timestep {self.num_timesteps}",
                metadata={
                    "timestep": self.num_timesteps,
                    "episodes": len(self.episode_rewards),
                    "mean_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0,
                    "algorithm": "PPO",
                    "reward_function": "paper_based",
                    "architecture": "MultiAgent",
                    "models_saved_count": self.models_saved,
                    "min_improvement_threshold": self.min_reward_improvement,
                }
            )
            
            artifact.add_file(temp_path)
            wandb.log_artifact(artifact)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if self.verbose > 0:
                print(f"[WandB] Saved paper-based reward model artifact: {artifact_name}")
                
        except Exception as e:
            print(f"[WARNING] Failed to save model artifact: {e}")


def create_target_sequence(num_drones=4, scale=1.5):
    """Create target sequence optimized for paper-based reward function"""
    if num_drones == 4:
        # Design more challenging but achievable formations
        # Optimized for the paper's reward function components
        targets = np.array([
            # Phase 0: Simple line formation (good for progress reward)
            [[-1.0*scale, 0.0, 1.2], [-0.3*scale, 0.0, 1.2], 
             [ 0.3*scale, 0.0, 1.2], [ 1.0*scale, 0.0, 1.2]],
        
            # Phase 1: Square formation (tests alignment and coordination)
            [[-scale, -scale, 1.5], [ scale, -scale, 1.5], 
             [ scale,  scale, 1.5], [-scale,  scale, 1.5]],
            
            # Phase 2: Diamond formation (tests precise control)
            [[ 0.0, -1.2*scale, 1.8], [ 1.2*scale, 0.0, 1.8], 
             [ 0.0,  1.2*scale, 1.8], [-1.2*scale, 0.0, 1.8]],
            
            # Phase 3: Compact formation (final precision test)
            [[-0.4*scale, -0.4*scale, 1.3], [ 0.4*scale, -0.4*scale, 1.3], 
             [ 0.4*scale,  0.4*scale, 1.3], [-0.4*scale,  0.4*scale, 1.3]]
        ])
    else:
        # Create circular formations for other numbers of drones
        targets = []
        n_phases = 4
        for phase in range(n_phases):
            phase_targets = []
            radius = scale * (0.8 + 0.3 * phase)
            height = 1.3 + 0.2 * phase
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones + phase * np.pi / 4
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                phase_targets.append([x, y, height])
            targets.append(phase_targets)
        targets = np.array(targets)
    
    return targets.astype(np.float32)


def save_final_artifacts(model, config, save_dir, run_name, mean_reward, std_reward):
    """Save final model and configuration as WandB artifacts"""
    
    # 1. Save final model
    final_model_path = os.path.join(save_dir, 'final_model.zip')
    model.save(final_model_path)
    
    final_model_artifact = wandb.Artifact(
        name=f"final_paper_reward_model_{run_name}",
        type="model",
        description="Final trained PPO model with paper-based reward function",
        metadata={
            "final_mean_reward": mean_reward,
            "final_std_reward": std_reward,
            "total_timesteps": config['total_timesteps'],
            "algorithm": "PPO",
            "reward_function": "paper_based",
            "architecture": config['architecture'],
            "extractor_type": config['extractor_type'],
            "features_dim": config['features_dim'],
            "num_drones": config['num_drones'],
            "paper_reward_hyperparameters": config['paper_reward_hyperparameters'],
        }
    )
    final_model_artifact.add_file(final_model_path)
    wandb.log_artifact(final_model_artifact)
    print(f"[WandB] Saved final paper-based reward model: final_paper_reward_model_{run_name}")
    
    # 2. Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    config_artifact = wandb.Artifact(
        name=f"paper_reward_config_{run_name}",
        type="config",
        description="Training configuration for paper-based reward function",
        metadata=config
    )
    config_artifact.add_file(config_path)
    wandb.log_artifact(config_artifact)
    print(f"[WandB] Saved config artifact: paper_reward_config_{run_name}")


def run(output_folder, gui, record_video, plot, local, wandb_project, wandb_entity, 
        extractor_type, features_dim, min_reward_improvement=5.0):
    """Main training function optimized for paper-based reward function"""
    
    run_name = f"paper_reward_{extractor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Paper-based reward hyperparameters (tunable)
    paper_reward_hyperparameters = {
        'lambda_1': 20.0,    # Progress reward - increased for faster learning
        'lambda_2': 1.5,     # Perception reward
        'lambda_3': 0.4,     # Perception alignment exponent
        'lambda_4': 0.01,    # Action magnitude penalty
        'lambda_5': 0.1,     # Action smoothness penalty
        'crash_penalty': 15.0,   # Crash penalty - increased
        'bounds_penalty': 8.0,   # Out-of-bounds penalty
    }
    
    config = {
        'algo': 'PPO',
        'architecture': 'MultiAgent_Matrix',
        'extractor_type': extractor_type,
        'features_dim': features_dim,
        'num_drones': DEFAULT_DRONES,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,
        'reward_function': 'paper_based',
        'paper_reward_hyperparameters': paper_reward_hyperparameters,
        'duration_sec': DEFAULT_DURATION_SEC,
        'total_timesteps': int(2e6) if local else int(5e5),  # Longer training for paper-based reward
        'learning_rate': 8e-5,  # Slightly lower for stability with new reward
        'batch_size': 512,  # Larger batch size for stability
        'n_epochs': 10,
        'gamma': 0.99,  # Slightly lower gamma for more immediate rewards
        'gae_lambda': 0.95,
        'n_steps': 2048,
        'clip_range': 0.15,  # Slightly higher for exploration with new reward
        'ent_coef': 0.01,  # Lower entropy for more focused learning
        'eval_freq': 25000 // NUM_VEC,
        'log_freq': 500,  # More frequent logging
        'save_freq': 25000,
        'min_reward_improvement': min_reward_improvement
    }
    
    # Enhanced WandB initialization
    wandb.finish()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=["multi-agent", "drone-swarm", "ppo", "rpm-actions", "paper-based-reward", extractor_type],
        notes=f"Paper-based reward function training with {extractor_type} extractor"
    )

    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get control frequency
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    # Create target sequence optimized for paper-based reward
    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=1.2)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] Paper-based reward function training")
    print(f"[INFO] Reward hyperparameters: {paper_reward_hyperparameters}")
    print(f"[INFO] Target sequence shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Using {extractor_type} architecture with {features_dim} features")

    # Environment creation function with paper-based reward hyperparameters
    def make_env():
        env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=False,
            record=False,
            **paper_reward_hyperparameters  # Pass paper-based reward hyperparameters
        )
        return Monitor(env)
    
    # Create vectorized training environment
    train_env = make_vec_env(make_env, n_envs=NUM_VEC, seed=0)
    
    # Create evaluation environment
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

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    
    # CREATE MODEL WITH CUSTOM MULTI-AGENT ARCHITECTURE
    print(f"\n[INFO] Creating PPO model optimized for paper-based reward function...")
    model = create_multiagent_ppo_model(
        env=train_env,
        extractor_type=extractor_type,
        features_dim=features_dim,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        verbose=1,
        tensorboard_log=os.path.join(save_dir, 'tb')
    )
    
    # Log model information
    total_params = sum(p.numel() for p in model.policy.parameters())
    extractor_params = sum(p.numel() for p in model.policy.features_extractor.parameters())
    
    print(f"\n[INFO] Model created for paper-based reward function!")
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Feature extractor parameters: {extractor_params:,}")
    
    # Test forward pass
    print(f"\n[INFO] Testing model forward pass...")
    test_obs, _ = eval_env.reset()
    test_obs_tensor = torch.from_numpy(test_obs[None, :, :]).float()
    try:
        with torch.no_grad():
            features = model.policy.features_extractor(test_obs_tensor)
        print(f"[INFO] ✓ Forward pass successful! Features shape: {features.shape}")
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        raise

    # ENHANCED CALLBACKS FOR PAPER-BASED REWARD
    paper_wandb_cb = PaperBasedWandbCallback(
        log_freq=config['log_freq'], 
        save_freq=config['save_freq'],
        min_reward_improvement=config['min_reward_improvement'],
        verbose=1
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    cb_list = CallbackList([paper_wandb_cb, eval_cb])

    # Log setup information
    wandb.log({
        'setup/num_drones': DEFAULT_DRONES,
        'setup/episode_length': len(target_sequence) * steps_per_target,
        'setup/phases': len(target_sequence),
        'setup/steps_per_phase': steps_per_target,
        'setup/control_frequency': freq,
        'setup/action_type': DEFAULT_ACT.name,
        'setup/reward_function': 'paper_based',
        'setup/reward_hyperparameters': paper_reward_hyperparameters,
        'setup/extractor_type': extractor_type,
        'setup/features_dim': features_dim,
        'setup/total_parameters': total_params,
        'setup/extractor_parameters': extractor_params,
        'setup/artifact_save_frequency': config['save_freq'],
    })

    # TRAINING
    print(f"\n[INFO] Starting paper-based reward training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=cb_list, 
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"[INFO] Paper-based reward training completed in {training_time:.2f} seconds")

    # Final evaluation
    print("\n[INFO] Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[RESULTS] Final Mean reward (paper-based): {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({
        'final_eval/mean_reward': mean_reward, 
        'final_eval/std_reward': std_reward,
        'final_eval/training_time_seconds': training_time,
        'final_eval/training_time_minutes': training_time / 60,
        'final_eval/reward_function': 'paper_based',
    })

    # SAVE FINAL ARTIFACTS
    print("\n[INFO] Saving final paper-based reward artifacts to WandB...")
    save_final_artifacts(model, config, save_dir, run_name, mean_reward, std_reward)

    # Demonstration
    if gui or record_video:
        print("\n[INFO] Running paper-based reward demonstration...")
        demo_env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=gui,
            record=record_video,
            **paper_reward_hyperparameters
        )
        
        obs, info = demo_env.reset(seed=42)
        start = time.time()
        episode_reward = 0
        
        max_steps = len(target_sequence) * steps_per_target
        print(f"[INFO] Running demonstration for {max_steps} steps...")
        
        for i in range(max_steps + 100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(action)
            episode_reward += reward
            
            if gui:
                demo_env.render()
                sync(i, start, demo_env.CTRL_TIMESTEP)
            
            if i % 100 == 0:
                phase = info.get('phase', -1)
                targets_reached = np.sum(info.get('targets_reached', []))
                mean_dist = info.get('mean_distance_to_targets', 0)
                print(f"Step {i:3d} | Phase {phase} | Targets: {targets_reached}/4 | "
                      f"Dist: {mean_dist:.2f} | Reward: {reward:.2f}")
            
            if done:
                print(f"[INFO] Episode finished at step {i}")
                break
        
        print(f"[INFO] Total episode reward (paper-based): {episode_reward:.2f}")
        demo_env.close()

    wandb.finish()
    print(f"\n[INFO] Paper-based reward training completed successfully!")
    print(f"[INFO] Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent drone swarm training with paper-based reward function')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, 
                        help='Use PyBullet GUI during demonstration')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, 
                        help='Record video of demonstration')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='Results folder')
    parser.add_argument('--plot', default=True, type=bool, 
                        help='Generate plots at end')
    parser.add_argument('--local', default=True, type=bool, 
                        help='Run locally (longer training)')
    parser.add_argument('--wandb_project', default='drone-swarm-paper-reward', type=str, 
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str, 
                        help='Weights & Biases entity/username')
    
    # Multi-agent architecture arguments
    parser.add_argument('--extractor_type', default=DEFAULT_EXTRACTOR_TYPE, 
                        choices=['matrix', 'attention', 'meanpool'], type=str,
                        help='Type of multi-agent feature extractor')
    parser.add_argument('--features_dim', default=DEFAULT_FEATURES_DIM, type=int,
                        help='Dimension of feature representation')
    parser.add_argument('--min_reward_improvement', default=5.0, type=float,
                        help='Minimum reward improvement required to save new best model')
    
    args = parser.parse_args()

    print("="*60)
    print("Multi-Agent Drone Swarm Training - Paper-Based Reward Function")
    print("="*60)
    print(f"Reward function: Paper-based (Nature 2023)")
    print(f"Action type: {DEFAULT_ACT} (4 RPM values per drone)")
    print(f"Extractor type: {args.extractor_type}")
    print(f"Features dim: {args.features_dim}")
    print(f"WandB project: {args.wandb_project}")
    print("="*60)

    run(
        args.output_folder,
        args.gui,
        args.record_video,
        args.plot,
        args.local,
        args.wandb_project,
        args.wandb_entity,
        args.extractor_type,
        args.features_dim,
        args.min_reward_improvement
    )