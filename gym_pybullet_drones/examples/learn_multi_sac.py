#!/usr/bin/env python3
"""
train_sac_paper_based.py - FIXED VERSION - SAC with paper-based reward function
Training script using SAC for better continuous control performance
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gymnasium as gym
import wandb
import torch

from stable_baselines3 import SAC
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
)

# Import your updated environment
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Updated settings for SAC with paper-based reward function
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('pid')  # RPM actions work great with SAC
DEFAULT_DRONES        = 4
DEFAULT_DURATION_SEC  = 3.0
NUM_VEC = 1

# Multi-agent architecture settings for SAC
DEFAULT_EXTRACTOR_TYPE = "matrix"
DEFAULT_FEATURES_DIM = 128


class SACPaperBasedCallback(BaseCallback):
    """Enhanced callback for SAC with paper-based reward function"""
    def __init__(self, log_freq=100, save_freq=25000, min_reward_improvement=10.0, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.min_reward_improvement = min_reward_improvement
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.best_mean_reward = -np.inf
        
        # Track SAC-specific metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        
        # Track when we last saved a model
        self.last_saved_reward = -np.inf
        self.models_saved = 0
        
        if verbose > 0:
            print(f"[SACPaperBasedCallback] Tracking SAC + paper-based reward training")
            print(f"[SACPaperBasedCallback] Minimum reward improvement: {min_reward_improvement}")
        
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
                'sac_train/episode_reward': self.current_episode_reward,
                'sac_train/episode_length': self.current_episode_length,
                'sac_train/mean_episode_reward_last_100': mean_reward_100,
                'sac_train/episodes_completed': len(self.episode_rewards),
                'sac_train/best_mean_reward': self.best_mean_reward,
                'sac_train/reward_improvement_needed': max(0, self.best_mean_reward + self.min_reward_improvement - mean_reward_100),
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
                
                self._save_model_artifact(f"sac_paper_reward_model_{mean_reward_100:.2f}_improvement_{reward_improvement:.2f}")
                
                if self.verbose > 0:
                    print(f"\n[SAC Model Saved] Timestep {self.num_timesteps}: "
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
                    'sac_train/timestep': self.num_timesteps,
                    'sac_train/current_reward': reward,
                    'sac_train/models_saved_count': self.models_saved,
                }
                
                # Add environment-specific metrics
                if 'phase' in info:
                    log_dict['sac_train/current_phase'] = info['phase']
                if 'targets_reached' in info:
                    log_dict['sac_train/targets_reached'] = np.sum(info['targets_reached'])
                if 'collision_count' in info:
                    log_dict['sac_train/collision_count'] = info['collision_count']
                if 'mean_distance_to_targets' in info:
                    log_dict['sac_train/mean_distance_to_targets'] = info['mean_distance_to_targets']
                if 'formation_error' in info:
                    log_dict['sac_train/formation_error'] = info['formation_error']
                
                wandb.log(log_dict, step=self.num_timesteps)
        
        # Save model periodically as artifacts
        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_model_artifact(f"sac_paper_reward_checkpoint_step_{self.num_timesteps}")
            
        return True
    
    def _save_model_artifact(self, artifact_name):
        """Save SAC model as WandB artifact"""
        try:
            temp_path = f"/tmp/{artifact_name}.zip"
            self.model.save(temp_path)
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"SAC model with paper-based reward at timestep {self.num_timesteps}",
                metadata={
                    "timestep": self.num_timesteps,
                    "episodes": len(self.episode_rewards),
                    "mean_reward_100": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0,
                    "algorithm": "SAC",
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
                print(f"[WandB] Saved SAC paper-based reward model artifact: {artifact_name}")
                
        except Exception as e:
            print(f"[WARNING] Failed to save SAC model artifact: {e}")


def create_sac_model(env, extractor_type="matrix", features_dim=128, learning_rate=3e-4):
    """Create SAC model with custom multi-agent feature extractor - FIXED VERSION"""
    
    # Choose the extractor class
    if extractor_type == "matrix":
        extractor_class = MultiAgentMatrixExtractor
    elif extractor_type == "attention":
        extractor_class = MultiAgentSelfAttentionExtractor
    elif extractor_type == "meanpool":
        extractor_class = MultiAgentMeanPoolExtractor
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    print(f"[create_sac_model] Using extractor class: {extractor_class.__name__}")
    
    # Policy kwargs for SAC with custom multi-agent extractor
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": [128, 128],  # Actor and critic network sizes
    }
    
    print(f"[create_sac_model] Policy kwargs: {policy_kwargs}")
    
    try:
        # Create SAC model optimized for continuous control
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            batch_size=256,  # Good batch size for SAC
            gamma=0.99,
            tau=0.01,  # Soft update rate - slightly higher for multi-agent stability
            ent_coef='auto',  # Automatic entropy tuning - great for exploration
            target_update_interval=1,
            train_freq=1,  # Train after every step
            gradient_steps=1,  # Number of gradient steps per update
            learning_starts=10000,  # Longer warm-up for multi-agent scenarios
            buffer_size=int(1e6),  # Large replay buffer for sample efficiency
            verbose=1,
            device='auto'
        )
        
        print(f"[create_sac_model] SAC model created successfully")
        
        # Verify the features extractor was properly initialized
        if hasattr(model.policy, 'features_extractor') and model.policy.features_extractor is not None:
            print(f"[create_sac_model] ✓ Features extractor properly initialized: {type(model.policy.features_extractor)}")
        else:
            print(f"[create_sac_model] ⚠️ Features extractor is None - using default MLP")
            print(f"[create_sac_model] Policy type: {type(model.policy)}")
            if hasattr(model.policy, 'features_extractor'):
                print(f"[create_sac_model] Features extractor value: {model.policy.features_extractor}")
            
        # Check if we have access to the actor/critic networks
        if hasattr(model.policy, 'actor') and model.policy.actor is not None:
            print(f"[create_sac_model] ✓ Actor network initialized: {type(model.policy.actor)}")
        if hasattr(model.policy, 'critic') and model.policy.critic is not None:
            print(f"[create_sac_model] ✓ Critic network initialized: {type(model.policy.critic)}")
            
    except Exception as e:
        print(f"[ERROR] Failed to create SAC model: {e}")
        print(f"[ERROR] This might be due to incompatible custom feature extractor")
        print(f"[ERROR] Falling back to default SAC model without custom extractor")
        
        # Fallback: create SAC without custom extractor
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=256,
            gamma=0.99,
            tau=0.01,
            ent_coef='auto',
            target_update_interval=1,
            train_freq=1,
            gradient_steps=1,
            learning_starts=10000,
            buffer_size=int(1e6),
            verbose=1,
            device='auto'
        )
        print(f"[create_sac_model] Fallback SAC model created successfully")
    
    print(f"[create_sac_model] Created SAC with {extractor_type} extractor")
    print(f"[create_sac_model] Optimized for continuous multi-agent control")
    
    return model


def safe_count_parameters(model):
    """Safely count model parameters, handling cases where features_extractor might be None"""
    try:
        # Count total parameters
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"[safe_count_parameters] Total policy parameters: {total_params:,}")
        
        # Try to count feature extractor parameters
        extractor_params = 0
        if hasattr(model.policy, 'features_extractor') and model.policy.features_extractor is not None:
            extractor_params = sum(p.numel() for p in model.policy.features_extractor.parameters())
            print(f"[safe_count_parameters] Feature extractor parameters: {extractor_params:,}")
        else:
            print(f"[safe_count_parameters] No custom feature extractor found - using default MLP")
            # Try to count MLP extractor parameters if it exists
            if hasattr(model.policy, 'mlp_extractor') and model.policy.mlp_extractor is not None:
                extractor_params = sum(p.numel() for p in model.policy.mlp_extractor.parameters())
                print(f"[safe_count_parameters] MLP extractor parameters: {extractor_params:,}")
        
        # Count actor/critic parameters separately
        actor_params = 0
        critic_params = 0
        
        if hasattr(model.policy, 'actor') and model.policy.actor is not None:
            actor_params = sum(p.numel() for p in model.policy.actor.parameters())
            print(f"[safe_count_parameters] Actor parameters: {actor_params:,}")
            
        if hasattr(model.policy, 'critic') and model.policy.critic is not None:
            critic_params = sum(p.numel() for p in model.policy.critic.parameters())
            print(f"[safe_count_parameters] Critic parameters: {critic_params:,}")
        
        return total_params, extractor_params, actor_params, critic_params
        
    except Exception as e:
        print(f"[ERROR] Failed to count parameters: {e}")
        return 0, 0, 0, 0


def create_target_sequence(num_drones=4, scale=1.2):
    """Create target sequence optimized for SAC training"""
    if num_drones == 4:
        # Design sequences that benefit from SAC's exploration capabilities
        targets = np.array([
            # Phase 0: Simple line formation (good for initial learning)
            [[-scale, 0.0, 1.2], [*scale, 0.0, 2.0], 
             [scale, -scale, 1.2], [scale, 0.0, 1.2]],
            
            # Phase 1: Square formation (tests coordination)
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
    """Save final SAC model and configuration as WandB artifacts"""
    
    # 1. Save final model
    final_model_path = os.path.join(save_dir, 'final_sac_model.zip')
    model.save(final_model_path)
    
    final_model_artifact = wandb.Artifact(
        name=f"final_sac_paper_reward_model_{run_name}",
        type="model",
        description="Final trained SAC model with paper-based reward function",
        metadata={
            "final_mean_reward": mean_reward,
            "final_std_reward": std_reward,
            "total_timesteps": config['total_timesteps'],
            "algorithm": "SAC",
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
    print(f"[WandB] Saved final SAC paper-based reward model: final_sac_paper_reward_model_{run_name}")
    
    # 2. Save configuration
    config_path = os.path.join(save_dir, 'sac_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    config_artifact = wandb.Artifact(
        name=f"sac_paper_reward_config_{run_name}",
        type="config",
        description="SAC training configuration for paper-based reward function",
        metadata=config
    )
    config_artifact.add_file(config_path)
    wandb.log_artifact(config_artifact)
    print(f"[WandB] Saved SAC config artifact: sac_paper_reward_config_{run_name}")


def run(output_folder, gui, record_video, plot, local, wandb_project, wandb_entity, 
        extractor_type, features_dim, min_reward_improvement=10.0):
    """Main training function using SAC with paper-based reward function"""
    
    run_name = f"sac_paper_reward_{extractor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Paper-based reward hyperparameters (optimized for SAC)
    paper_reward_hyperparameters = {
        'lambda_1': 15.0,    # Progress reward - SAC handles this well
        'lambda_2': 1.0,     # Perception reward - lower for SAC's exploration
        'lambda_3': 0.3,     # Perception alignment exponent
        'lambda_4': 0.005,   # Action magnitude penalty - lower for SAC
        'lambda_5': 0.05,    # Action smoothness penalty - SAC naturally smooth
        'crash_penalty': 12.0,   # Crash penalty
        'bounds_penalty': 6.0,   # Out-of-bounds penalty
    }
    
    config = {
        'algo': 'SAC',  # Changed from PPO
        'architecture': 'MultiAgent_SAC',
        'extractor_type': extractor_type,
        'features_dim': features_dim,
        'num_drones': DEFAULT_DRONES,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,
        'reward_function': 'paper_based',
        'paper_reward_hyperparameters': paper_reward_hyperparameters,
        'duration_sec': DEFAULT_DURATION_SEC,
        'total_timesteps': int(1e6) if local else int(3e5),  # SAC is more sample efficient
        'learning_rate': 3e-4,  # Good learning rate for SAC
        'batch_size': 256,  # Standard SAC batch size
        'gamma': 0.99,
        'tau': 0.01,  # SAC soft update parameter
        'ent_coef': 'auto',  # Automatic entropy tuning
        'learning_starts': 10000,  # Warm-up period
        'train_freq': 1,  # Train after every step
        'buffer_size': int(1e6),  # Large replay buffer
        'eval_freq': 25000 // NUM_VEC,
        'log_freq': 500,
        'save_freq': 25000,
        'min_reward_improvement': min_reward_improvement
    }
    
    # Enhanced WandB initialization for SAC
    wandb.finish()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=["multi-agent", "drone-swarm", "sac", "rpm-actions", "paper-based-reward", extractor_type],
        notes=f"SAC with paper-based reward function training using {extractor_type} extractor"
    )

    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get control frequency
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    # Create target sequence optimized for SAC training
    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=1.2)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] SAC with paper-based reward function training")
    print(f"[INFO] SAC reward hyperparameters: {paper_reward_hyperparameters}")
    print(f"[INFO] Target sequence shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Using SAC with {extractor_type} architecture and {features_dim} features")

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
    
    # CREATE SAC MODEL WITH CUSTOM MULTI-AGENT ARCHITECTURE
    print(f"\n[INFO] Creating SAC model optimized for paper-based reward function...")
    model = create_sac_model(
        env=train_env,
        extractor_type=extractor_type,
        features_dim=features_dim,
        learning_rate=config['learning_rate']
    )
    
    # FIXED: Safe parameter counting
    print(f"\n[INFO] Counting SAC model parameters...")
    total_params, extractor_params, actor_params, critic_params = safe_count_parameters(model)
    
    print(f"\n[INFO] SAC model created for paper-based reward function!")
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Feature extractor parameters: {extractor_params:,}")
    print(f"[INFO] Actor parameters: {actor_params:,}")
    print(f"[INFO] Critic parameters: {critic_params:,}")
    print(f"[INFO] SAC automatic entropy tuning: {config['ent_coef']}")
    print(f"[INFO] SAC replay buffer size: {config['buffer_size']:,}")
    
    # Test forward pass
    print(f"\n[INFO] Testing SAC model forward pass...")
    test_obs, _ = eval_env.reset()
    test_obs_tensor = torch.from_numpy(test_obs[None, :, :]).float()
    try:
        with torch.no_grad():
            # Try to get features from the model
            if hasattr(model.policy, 'features_extractor') and model.policy.features_extractor is not None:
                features = model.policy.features_extractor(test_obs_tensor)
                print(f"[INFO] ✓ SAC forward pass successful! Features shape: {features.shape}")
            else:
                # Test with policy's extract_features method if available
                features = model.policy.extract_features(test_obs_tensor)
                print(f"[INFO] ✓ SAC forward pass successful! Features shape: {features.shape}")
                print(f"[INFO] Using default MLP feature extraction")
    except Exception as e:
        print(f"[ERROR] SAC forward pass failed: {e}")
        print(f"[WARNING] This might affect training performance but won't prevent training")

    # ENHANCED CALLBACKS FOR SAC WITH PAPER-BASED REWARD
    sac_paper_cb = SACPaperBasedCallback(
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
    
    cb_list = CallbackList([sac_paper_cb, eval_cb])

    # Log setup information
    wandb.log({
        'setup/algorithm': 'SAC',
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
        'setup/actor_parameters': actor_params,
        'setup/critic_parameters': critic_params,
        'setup/sac_buffer_size': config['buffer_size'],
        'setup/sac_learning_starts': config['learning_starts'],
        'setup/artifact_save_frequency': config['save_freq'],
    })

    # SAC TRAINING
    print(f"\n[INFO] Starting SAC training with paper-based rewards for {config['total_timesteps']} timesteps...")
    print(f"[INFO] SAC will warm up for {config['learning_starts']} steps before training")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=cb_list, 
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"[INFO] SAC paper-based reward training completed in {training_time:.2f} seconds")

    # Final evaluation
    print("\n[INFO] Running final SAC evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[RESULTS] Final SAC Mean reward (paper-based): {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({
        'final_eval/algorithm': 'SAC',
        'final_eval/mean_reward': mean_reward, 
        'final_eval/std_reward': std_reward,
        'final_eval/training_time_seconds': training_time,
        'final_eval/training_time_minutes': training_time / 60,
        'final_eval/reward_function': 'paper_based',
    })

    # SAVE FINAL SAC ARTIFACTS
    print("\n[INFO] Saving final SAC paper-based reward artifacts to WandB...")
    save_final_artifacts(model, config, save_dir, run_name, mean_reward, std_reward)

    # Demonstration
    if gui or record_video:
        print("\n[INFO] Running SAC paper-based reward demonstration...")
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
        print(f"[INFO] Running SAC demonstration for {max_steps} steps...")
        
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
                print(f"[INFO] SAC episode finished at step {i}")
                break
        
        print(f"[INFO] Total SAC episode reward (paper-based): {episode_reward:.2f}")
        demo_env.close()

    wandb.finish()
    print(f"\n[INFO] SAC paper-based reward training completed successfully!")
    print(f"[INFO] Results saved to: {save_dir}")
    print(f"[INFO] SAC should provide better sample efficiency and exploration than PPO!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent drone swarm training with SAC and paper-based reward function')
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
    parser.add_argument('--wandb_project', default='drone-swarm-sac-paper-reward', type=str, 
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str, 
                        help='Weights & Biases entity/username')
    
    # Multi-agent architecture arguments
    parser.add_argument('--extractor_type', default=DEFAULT_EXTRACTOR_TYPE, 
                        choices=['matrix', 'attention', 'meanpool'], type=str,
                        help='Type of multi-agent feature extractor')
    parser.add_argument('--features_dim', default=DEFAULT_FEATURES_DIM, type=int,
                        help='Dimension of feature representation')
    parser.add_argument('--min_reward_improvement', default=10.0, type=float,
                        help='Minimum reward improvement required to save new best model')
    
    args = parser.parse_args()

    print("="*60)
    print("Multi-Agent Drone Swarm Training - SAC with Paper-Based Reward")
    print("="*60)
    print(f"Algorithm: SAC (Soft Actor-Critic)")
    print(f"Reward function: Paper-based (Nature 2023)")
    print(f"Action type: {DEFAULT_ACT} (4 RPM values per drone)")
    print(f"Extractor type: {args.extractor_type}")
    print(f"Features dim: {args.features_dim}")
    print(f"WandB project: {args.wandb_project}")
    print("SAC should provide better continuous control than PPO!")
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