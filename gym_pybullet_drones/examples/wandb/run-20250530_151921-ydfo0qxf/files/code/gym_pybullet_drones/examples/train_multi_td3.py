#!/usr/bin/env python3
"""
train_multi_td3.py - Multi-algorithm training script supporting TD3, SAC, and PPO
Optimized for continuous control with multi-agent drone environments
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gymnasium as gym
import wandb
import torch
from typing import Optional, Dict, Any

from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback

# Import your custom extractors
from multi_agent_extractors_td3 import create_multiagent_model

# Import your environment
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Default settings
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('rpm')
DEFAULT_DRONES        = 1
DEFAULT_DURATION_SEC  = 3.0
DEFAULT_ALGORITHM     = 'td3'
DEFAULT_EXTRACTOR     = 'matrix'
DEFAULT_FEATURES_DIM  = 128


class MultiAlgorithmWandbCallback(BaseCallback):
    """Enhanced callback supporting TD3, SAC, and PPO with detailed logging"""
    
    def __init__(self, algorithm: str, log_freq: int = 100, save_freq: int = 25000, 
                 min_reward_improvement: float = 5.0, verbose: int = 0):
        super().__init__(verbose)
        self.algorithm = algorithm.upper()
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.min_reward_improvement = min_reward_improvement
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.best_mean_reward = -np.inf
        self.models_saved = 0
        
        # For off-policy algorithms
        self.is_off_policy = algorithm.lower() in ['td3', 'sac']
        
        if verbose > 0:
            print(f"[MultiAlgorithmWandbCallback] Initialized for {self.algorithm}")
            print(f"[MultiAlgorithmWandbCallback] Off-policy: {self.is_off_policy}")
        
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
                f'{self.algorithm}/episode_reward': self.current_episode_reward,
                f'{self.algorithm}/episode_length': self.current_episode_length,
                f'{self.algorithm}/mean_episode_reward_last_100': mean_reward_100,
                f'{self.algorithm}/episodes_completed': len(self.episode_rewards),
                f'{self.algorithm}/best_mean_reward': self.best_mean_reward,
            }, step=self.num_timesteps)
            
            # Save model if significant improvement
            reward_improvement = mean_reward_100 - self.best_mean_reward
            should_save = (
                mean_reward_100 > self.best_mean_reward and
                reward_improvement >= self.min_reward_improvement and
                len(self.episode_rewards) >= 10
            )
            
            if should_save:
                self.best_mean_reward = mean_reward_100
                self.models_saved += 1
                self._save_model_artifact(f"{self.algorithm}_model_{mean_reward_100:.2f}")
                
                if self.verbose > 0:
                    print(f"\n[Model Saved] {self.algorithm} at timestep {self.num_timesteps}: "
                          f"Reward improved by {reward_improvement:.3f} to {mean_reward_100:.2f}")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log detailed metrics periodically
        if self.n_calls % self.log_freq == 0:
            log_dict = {
                f'{self.algorithm}/timestep': self.num_timesteps,
                f'{self.algorithm}/models_saved': self.models_saved,
            }
            
            # Add algorithm-specific metrics
            if self.is_off_policy and hasattr(self.model, 'replay_buffer'):
                log_dict[f'{self.algorithm}/replay_buffer_size'] = self.model.replay_buffer.size()
            
            # Environment metrics
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                if 'phase' in info:
                    log_dict[f'{self.algorithm}/current_phase'] = info['phase']
                if 'mean_distance_to_targets' in info:
                    log_dict[f'{self.algorithm}/mean_distance_to_targets'] = info['mean_distance_to_targets']
                if 'collision_count' in info:
                    log_dict[f'{self.algorithm}/collision_count'] = info['collision_count']
            
            wandb.log(log_dict, step=self.num_timesteps)
        
        # Save checkpoints
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_model_artifact(f"{self.algorithm}_checkpoint_{self.num_timesteps}")
            
        return True
    
    def _save_model_artifact(self, artifact_name: str):
        """Save model as WandB artifact"""
        try:
            temp_path = f"/tmp/{artifact_name}.zip"
            self.model.save(temp_path)
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"{self.algorithm} model at timestep {self.num_timesteps}",
                metadata={
                    "algorithm": self.algorithm,
                    "timestep": self.num_timesteps,
                    "episodes": len(self.episode_rewards),
                    "mean_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0,
                    "models_saved": self.models_saved,
                }
            )
            
            artifact.add_file(temp_path)
            wandb.log_artifact(artifact)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if self.verbose > 0:
                print(f"[WandB] Saved {self.algorithm} model artifact: {artifact_name}")
                
        except Exception as e:
            print(f"[WARNING] Failed to save model artifact: {e}")


def create_action_noise(env, noise_type: str = "normal", noise_std: float = 0.1):
    """Create action noise for TD3/SAC"""
    n_actions = env.action_space.shape[-1]
    
    if noise_type == "normal":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions)
        )
    else:
        action_noise = None
    
    return action_noise


def create_target_sequence(num_drones: int = 4, scale: float = 1.5):
    """Create target sequence for multi-agent navigation"""
    if True:#num_drones == 4:
        targets = np.array([
            # Phase 0: Simple line formation (good for progress reward)
            [[scale, 0.0, 2.0]],
            [[scale, 0.0, 2.0]],
            [[scale, 0.0, 2.0]],
            [[scale, 0.0, 2.0]]
        ])
        # targets = np.array([
        #     # Phase 0: Line formation
        #     [[-1.0*scale, 0.0, 1.2], [-0.3*scale, 0.0, 1.2], 
        #      [ 0.3*scale, 0.0, 1.2], [ 1.0*scale, 0.0, 1.2]],
            
        #     # Phase 1: Square formation
        #     [[-scale, -scale, 1.5], [ scale, -scale, 1.5], 
        #      [ scale,  scale, 1.5], [-scale,  scale, 1.5]],
            
        #     # Phase 2: Diamond formation
        #     [[ 0.0, -1.2*scale, 1.8], [ 1.2*scale, 0.0, 1.8], 
        #      [ 0.0,  1.2*scale, 1.8], [-1.2*scale, 0.0, 1.8]],
            
        #     # Phase 3: Compact formation
        #     [[-0.4*scale, -0.4*scale, 1.3], [ 0.4*scale, -0.4*scale, 1.3], 
        #      [ 0.4*scale,  0.4*scale, 1.3], [-0.4*scale,  0.4*scale, 1.3]]
        # ])
    else:
        # Circular formations for other drone counts
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


def get_algorithm_config(algorithm: str) -> Dict[str, Any]:
    """Get algorithm-specific configuration"""
    configs = {
        'td3': {
            'total_timesteps': int(1e6),
            'learning_rate': 1e-3,
            'buffer_size': 1_000_000,
            'learning_starts': 10000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 100,
            'gradient_steps': 100,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'eval_freq': 10000,
            'n_eval_episodes': 5,
        },
        'sac': {
            'total_timesteps': int(1e6),
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_entropy': 'auto',
            'eval_freq': 10000,
            'n_eval_episodes': 5,
        },
        'ppo': {
            'total_timesteps': int(2e6),
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'eval_freq': 25000,
            'n_eval_episodes': 5,
        }
    }
    
    return configs.get(algorithm.lower(), configs['td3'])


def run(algorithm: str, output_folder: str, gui: bool, record_video: bool,
        extractor_type: str, features_dim: int, num_drones: int,
        wandb_project: str, wandb_entity: Optional[str],
        noise_type: str = "normal", noise_std: float = 0.1,
        normalize_observations: bool = True, num_vec_envs: int = 4):
    """Main training function supporting multiple algorithms"""
    
    run_name = f"{algorithm}_{extractor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Get algorithm-specific configuration
    config = get_algorithm_config(algorithm)
    config.update({
        'algorithm': algorithm,
        'extractor_type': extractor_type,
        'features_dim': features_dim,
        'num_drones': num_drones,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,
        'noise_type': noise_type,
        'noise_std': noise_std,
        'normalize_observations': normalize_observations,
        'num_vec_envs': num_vec_envs,
    })
    
    # Initialize WandB
    wandb.finish()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=[algorithm, "multi-agent", "drone-swarm", extractor_type],
        notes=f"{algorithm} training with {extractor_type} extractor for {num_drones} drones"
    )

    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get control frequency
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    # Create target sequence
    target_sequence = create_target_sequence(num_drones, scale=1.2)
    print(target_sequence)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"\n[INFO] {algorithm.upper()} Training Configuration")
    print(f"[INFO] Number of drones: {num_drones}")
    print(f"[INFO] Extractor type: {extractor_type} with {features_dim} features")
    print(f"[INFO] Target sequence shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Vectorized environments: {num_vec_envs}")

    # Paper-based reward hyperparameters
    reward_hyperparameters = {
        'lambda_1': 15.0,    # Progress reward
        'lambda_2': 1.0,     # Perception reward
        'lambda_3': 0.3,     # Perception alignment
        'lambda_4': 0.005,   # Action magnitude penalty
        'lambda_5': 0.05,    # Action smoothness penalty
        'crash_penalty': 10.0,
        'bounds_penalty': 5.0,
    }

    # Environment creation function
    def make_env():
        env = MultiTargetAviary(
            num_drones=num_drones,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=False,
            record=False,
            **reward_hyperparameters
        )
        return Monitor(env)
    
    # Create vectorized environments
    if algorithm.lower() in ['td3', 'sac']:
        # Off-policy algorithms can use fewer parallel envs
        train_env = make_vec_env(make_env, n_envs=min(num_vec_envs, 4), seed=0)
    else:
        # On-policy algorithms benefit from more parallel envs
        train_env = make_vec_env(make_env, n_envs=num_vec_envs, seed=0)
    
    # Optionally normalize observations
    #if normalize_observations:
    #    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    
    # Create evaluation environment
    eval_env = MultiTargetAviary(
        num_drones=num_drones,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        target_sequence=target_sequence,
        steps_per_target=steps_per_target,
        gui=False,
        record=False,
        **reward_hyperparameters
    )

    print(f"\n[INFO] Action space: {train_env.action_space}")
    print(f"[INFO] Observation space: {train_env.observation_space}")
    
    # Create model with appropriate algorithm
    if algorithm.lower() == 'td3':
        # Add action noise for TD3
        action_noise = create_action_noise(train_env, noise_type, noise_std)
        config['action_noise'] = action_noise
        
        model = create_multiagent_model(
            train_env,
            algorithm="td3",
            extractor_type=extractor_type,
            features_dim=features_dim,
            action_noise=action_noise,
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            tau=config['tau'],
            gamma=config['gamma'],
            train_freq=config['train_freq'],
            gradient_steps=config['gradient_steps'],
            policy_delay=config['policy_delay'],
            target_policy_noise=config['target_policy_noise'],
            target_noise_clip=config['target_noise_clip'],
            verbose=1,
            tensorboard_log=os.path.join(save_dir, 'tb')
        )
    
    elif algorithm.lower() == 'sac':
        model = create_multiagent_model(
            train_env,
            algorithm="sac",
            extractor_type=extractor_type,
            features_dim=features_dim,
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            tau=config['tau'],
            gamma=config['gamma'],
            train_freq=config['train_freq'],
            gradient_steps=config['gradient_steps'],
            ent_coef=config['ent_coef'],
            target_entropy=config['target_entropy'],
            verbose=1,
            tensorboard_log=os.path.join(save_dir, 'tb')
        )
    
    elif algorithm.lower() == 'ppo':
        model = create_multiagent_model(
            train_env,
            algorithm="ppo",
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
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Log model information
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[INFO] Model created with {algorithm.upper()}!")
    print(f"[INFO] Total parameters: {total_params:,}")
    
    # Create callbacks
    wandb_cb = MultiAlgorithmWandbCallback(
        algorithm=algorithm,
        log_freq=500,
        save_freq=25000,
        min_reward_improvement=5.0,
        verbose=1
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1
    )
    
    cb_list = CallbackList([wandb_cb, eval_cb])

    # Log setup information
    wandb.log({
        'setup/algorithm': algorithm,
        'setup/num_drones': num_drones,
        'setup/episode_length': len(target_sequence) * steps_per_target,
        'setup/phases': len(target_sequence),
        'setup/steps_per_phase': steps_per_target,
        'setup/extractor_type': extractor_type,
        'setup/features_dim': features_dim,
        'setup/total_parameters': total_params,
    })

    # Training
    print(f"\n[INFO] Starting {algorithm.upper()} training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=cb_list,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {training_time:.2f} seconds")

    # Final evaluation
    print("\n[INFO] Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[RESULTS] Final Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({
        f'{algorithm}/final_mean_reward': mean_reward,
        f'{algorithm}/final_std_reward': std_reward,
        f'{algorithm}/training_time_seconds': training_time,
    })

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.zip')
    model.save(final_model_path)
    
    # Save normalization stats if used
    # if normalize_observations:
    #     train_env.save(os.path.join(save_dir, 'vec_normalize.pkl'))

    # Demonstration
    if gui or record_video:
        print(f"\n[INFO] Running {algorithm.upper()} demonstration...")
        demo_env = MultiTargetAviary(
            num_drones=num_drones,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=gui,
            record=record_video,
            **reward_hyperparameters
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
                mean_dist = info.get('mean_distance_to_targets', 0)
                print(f"Step {i:3d} | Phase {phase} | Dist: {mean_dist:.2f} | Reward: {reward:.2f}")
            
            if done:
                print(f"[INFO] Episode finished at step {i}")
                break
        
        print(f"[INFO] Total episode reward: {episode_reward:.2f}")
        demo_env.close()

    wandb.finish()
    print(f"\n[INFO] {algorithm.upper()} training completed successfully!")
    print(f"[INFO] Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-algorithm multi-agent drone swarm training')
    
    # Algorithm selection
    parser.add_argument('--algorithm', default=DEFAULT_ALGORITHM, 
                        choices=['td3', 'sac', 'ppo'], type=str,
                        help='RL algorithm to use')
    
    # Environment settings
    parser.add_argument('--num_drones', default=DEFAULT_DRONES, type=int,
                        help='Number of drones in the swarm')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                        help='Use PyBullet GUI during demonstration')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool,
                        help='Record video of demonstration')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Results folder')
    
    # Multi-agent architecture
    parser.add_argument('--extractor_type', default=DEFAULT_EXTRACTOR,
                        choices=['matrix', 'attention', 'meanpool'], type=str,
                        help='Type of multi-agent feature extractor')
    parser.add_argument('--features_dim', default=DEFAULT_FEATURES_DIM, type=int,
                        help='Dimension of feature representation')
    
    # TD3/SAC specific
    parser.add_argument('--noise_type', default='normal',
                        choices=['normal', 'ornstein-uhlenbeck', 'none'], type=str,
                        help='Type of action noise for TD3')
    parser.add_argument('--noise_std', default=0.1, type=float,
                        help='Standard deviation of action noise')
    parser.add_argument('--normalize_observations', default=True, type=str2bool,
                        help='Normalize observations using running statistics')
    
    # Training settings
    parser.add_argument('--num_vec_envs', default=4, type=int,
                        help='Number of parallel environments')
    
    # WandB settings
    parser.add_argument('--wandb_project', default='drone-swarm-multi-algo', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='Weights & Biases entity/username')
    
    args = parser.parse_args()

    print("="*60)
    print(f"Multi-Agent Drone Swarm Training - {args.algorithm.upper()}")
    print("="*60)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Number of drones: {args.num_drones}")
    print(f"Extractor type: {args.extractor_type}")
    print(f"Features dim: {args.features_dim}")
    print(f"Action noise: {args.noise_type} (std={args.noise_std})")
    print(f"Normalize observations: {args.normalize_observations}")
    print(f"Parallel environments: {args.num_vec_envs}")
    print(f"WandB project: {args.wandb_project}")
    print("="*60)

    run(
        algorithm=args.algorithm,
        output_folder=args.output_folder,
        gui=args.gui,
        record_video=args.record_video,
        extractor_type=args.extractor_type,
        features_dim=args.features_dim,
        num_drones=args.num_drones,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
        normalize_observations=args.normalize_observations,
        num_vec_envs=args.num_vec_envs
    )