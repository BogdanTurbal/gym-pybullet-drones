#!/usr/bin/env python3
"""
train_multi_rpm.py - UPDATED for ActionType.RPM (4 actions per drone)
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
# from MultiTargetAviary import MultiTargetAviary  # Your updated environment

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Updated settings for RPM actions
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('rpm')  # CHANGED: Now using RPM (4 values per drone)
DEFAULT_DRONES        = 4
DEFAULT_DURATION_SEC  = 3.0
NUM_VEC = 1

# Multi-agent architecture settings
DEFAULT_EXTRACTOR_TYPE = "matrix"  # "matrix", "attention", "meanpool"
DEFAULT_FEATURES_DIM = 256


class EnhancedWandbCallback(BaseCallback):
    """Enhanced callback for detailed logging"""
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Get reward and update episode tracking
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            wandb.log({
                'train/episode_reward': self.current_episode_reward,
                'train/episode_length': self.current_episode_length,
                'train/mean_episode_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0,
                'train/episodes_completed': len(self.episode_rewards),
            }, step=self.num_timesteps)
            
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
                }
                
                # Add environment-specific metrics if available
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
        
        return True


def create_target_sequence(num_drones=4, scale=1.2):
    """Create target sequence for training"""
    if num_drones == 4:
        targets = np.array([
            [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
             [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
             [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
             [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            [[ scale,  scale, 1.0], [ scale,  scale, 1.0], 
             [ scale,  scale, 1.0], [ scale,  scale, 1.0]],
            # Phase 0: Square formation
            # [[ scale,  scale, 1.5], [-scale,  scale, 1.5], 
            #  [-scale, -scale, 1.5], [ scale, -scale, 1.5]],
            
            # # Phase 1: Rotate clockwise
            # [[-scale,  scale, 1.5], [-scale, -scale, 1.5], 
            #  [ scale, -scale, 1.5], [ scale,  scale, 1.5]],
            
            # # Phase 2: Diamond formation
            # [[ 0.0,  scale*1.2, 1.8], [-scale*1.2,  0.0, 1.8], 
            #  [ 0.0, -scale*1.2, 1.8], [ scale*1.2,  0.0, 1.8]],
            
            # # Phase 3: Tight formation
            # [[ 0.4,  0.4, 1.5], [-0.4,  0.4, 1.5], 
            #  [-0.4, -0.4, 1.5], [ 0.4, -0.4, 1.5]]
        ])
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


def run(output_folder, gui, record_video, plot, local, wandb_project, wandb_entity, 
        extractor_type, features_dim):
    """
    Main training function updated for RPM actions
    """
    run_name = f"multiagent_rpm_{extractor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = {
        'algo': 'PPO',
        'architecture': 'MultiAgent_Matrix',
        'extractor_type': extractor_type,
        'features_dim': features_dim,
        'num_drones': DEFAULT_DRONES,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,  # Now 'rpm'
        'duration_sec': DEFAULT_DURATION_SEC,
        'total_timesteps': int(5e5) if local else int(1e5),
        'learning_rate': 3e-4,
        'batch_size': 128,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'n_steps': 2048,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'eval_freq': 25000 // NUM_VEC,
        'log_freq': 1000,
    }
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get control frequency
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    # Create target sequence
    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=1.0)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] Created target sequence with shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Using {extractor_type} architecture with {features_dim} features")
    print(f"[INFO] Action type: {DEFAULT_ACT} (4 RPM values per drone)")

    # Environment creation function
    def make_env():
        env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,  # Using RPM actions
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=False,
            record=False
        )
        return Monitor(env)
    
    # Create vectorized training environment
    train_env = make_vec_env(make_env, n_envs=NUM_VEC, seed=0)
    
    # Create evaluation environment
    eval_env = MultiTargetAviary(
        num_drones=DEFAULT_DRONES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,  # Using RPM actions
        target_sequence=target_sequence,
        steps_per_target=steps_per_target,
        gui=False,
        record=False
    )

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Action space shape:', train_env.action_space.shape)  # Should be (4, 4)
    print('[INFO] Observation space:', train_env.observation_space)
    print('[INFO] Observation space shape:', train_env.observation_space.shape)
    
    # Calculate expected observation dimensions
    # For ActionType.RPM with ctrl_freq=30:
    # - Base kinematic: 12 features
    # - Action buffer: ACTION_BUFFER_SIZE * 4 = 15 * 4 = 60 features  
    # - Target features: 8 features
    # - Total: 12 + 60 + 8 = 80 features per drone
    
    obs_shape = train_env.observation_space.shape
    action_buffer_size = freq // 2  # ACTION_BUFFER_SIZE
    expected_base_features = 12 + (action_buffer_size * 4)  # 12 + (15 * 4) = 72
    expected_total_features = expected_base_features + 8  # 72 + 8 = 80
    expected_shape = (DEFAULT_DRONES, expected_total_features)
    
    print(f'[INFO] Expected observation breakdown:')
    print(f'  - Kinematic features: 12')
    print(f'  - Action buffer: {action_buffer_size} timesteps * 4 RPM = {action_buffer_size * 4} features')
    print(f'  - Target features: 8')
    print(f'  - Total per drone: {expected_total_features}')
    print(f'  - Expected shape: {expected_shape}')
    print(f'  - Actual shape: {obs_shape}')
    
    # Verify observation shape is reasonable (might vary slightly due to implementation details)
    assert obs_shape[0] == DEFAULT_DRONES, f"Expected {DEFAULT_DRONES} drones, got {obs_shape[0]}"
    assert obs_shape[1] >= expected_total_features - 5, f"Expected ~{expected_total_features} features, got {obs_shape[1]}"
    print(f'[INFO] ✓ Observation shape verified')

    # CREATE MODEL WITH CUSTOM MULTI-AGENT ARCHITECTURE
    print(f"\n[INFO] Creating PPO model with {extractor_type} extractor...")
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
    
    print(f"\n[INFO] Model created successfully!")
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Feature extractor parameters: {extractor_params:,}")
    
    # Test forward pass to verify everything works
    print(f"\n[INFO] Testing model forward pass...")
    test_obs, _ = eval_env.reset()
    test_obs_tensor = torch.from_numpy(test_obs[None, :, :]).float()  # Add batch dimension
    try:
        with torch.no_grad():
            features = model.policy.features_extractor(test_obs_tensor)
        print(f"[INFO] ✓ Forward pass successful! Features shape: {features.shape}")
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        raise

    # Set up callbacks
    enhanced_wandb_cb = EnhancedWandbCallback(log_freq=config['log_freq'], verbose=1)
    
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
    
    standard_wandb_cb = WandbCallback(
        model_save_path=save_dir,
        verbose=1,
        model_save_freq=50000
    )
    
    cb_list = CallbackList([enhanced_wandb_cb, eval_cb, standard_wandb_cb])

    # Log setup information
    wandb.log({
        'setup/num_drones': DEFAULT_DRONES,
        'setup/episode_length': len(target_sequence) * steps_per_target,
        'setup/phases': len(target_sequence),
        'setup/steps_per_phase': steps_per_target,
        'setup/control_frequency': freq,
        'setup/action_type': DEFAULT_ACT.name,
        'setup/action_space_shape': obs_shape,
        'setup/extractor_type': extractor_type,
        'setup/features_dim': features_dim,
        'setup/total_parameters': total_params,
        'setup/extractor_parameters': extractor_params,
        'setup/obs_shape': obs_shape,
        'setup/action_buffer_size': action_buffer_size,
    })

    # TRAINING
    print(f"\n[INFO] Starting training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=cb_list, 
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"[INFO] Training completed in {training_time:.2f} seconds")

    # Save final model
    final_path = os.path.join(save_dir, 'final_model.zip')
    model.save(final_path)

    # Final evaluation
    print("\n[INFO] Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[RESULTS] Final Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({
        'final_eval/mean_reward': mean_reward, 
        'final_eval/std_reward': std_reward,
        'final_eval/training_time_seconds': training_time,
        'final_eval/training_time_minutes': training_time / 60,
    })

    # Demonstration
    if gui or record_video:
        print("\n[INFO] Running demonstration...")
        demo_env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,  # Using RPM actions
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=gui,
            record=record_video
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
            
            # Print progress and action info
            if i % 100 == 0:
                phase = info.get('phase', -1)
                targets_reached = np.sum(info.get('targets_reached', []))
                mean_dist = info.get('mean_distance_to_targets', 0)
                print(f"Step {i:3d} | Phase {phase} | Targets: {targets_reached}/4 | "
                      f"Dist: {mean_dist:.2f} | Reward: {reward:.2f}")
                print(f"  Action shape: {action.shape} | Action range: [{action.min():.2f}, {action.max():.2f}]")
            
            if done:
                print(f"[INFO] Episode finished at step {i}")
                break
        
        print(f"[INFO] Total episode reward: {episode_reward:.2f}")
        demo_env.close()

    wandb.finish()
    print(f"\n[INFO] Training completed successfully!")
    print(f"[INFO] Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent drone swarm training with RPM actions')
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
    parser.add_argument('--wandb_project', default='drone-swarm-rpm', type=str, 
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str, 
                        help='Weights & Biases entity/username')
    
    # Multi-agent architecture arguments
    parser.add_argument('--extractor_type', default=DEFAULT_EXTRACTOR_TYPE, 
                        choices=['matrix', 'attention', 'meanpool'], type=str,
                        help='Type of multi-agent feature extractor')
    parser.add_argument('--features_dim', default=DEFAULT_FEATURES_DIM, type=int,
                        help='Dimension of feature representation')
    
    args = parser.parse_args()

    print("="*60)
    print("Multi-Agent Drone Swarm Training - RPM Actions")
    print("="*60)
    print(f"Action type: {DEFAULT_ACT} (4 RPM values per drone)")
    print(f"Extractor type: {args.extractor_type}")
    print(f"Features dim: {args.features_dim}")
    print(f"Observation format: Matrix ({DEFAULT_DRONES}, ~80 features)")
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
        args.features_dim
    )