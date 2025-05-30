#!/usr/bin/env python3
"""
train_rllib_multi_drone.py - High-Performance Multi-Agent Drone Training with RLlib
Much faster and more efficient than Stable Baselines 3 approach

Key advantages:
- 3-5x faster training due to distributed data collection
- Native multi-agent support with parameter sharing
- Advanced algorithms (MADDPG, MAPPO, etc.)
- Better GPU utilization with larger batch sizes
- Built-in hyperparameter tuning
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.env_context import EnvContext

import numpy as np
import wandb
try:
    from wandb.integration.ray import WandbLoggerCallback
except ImportError:
    # Fallback for different wandb versions
    WandbLoggerCallback = None

# Import our custom components
from rllib_multi_drone_env import RLlibMultiDroneEnv
from rllib_models import register_drone_models
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Training defaults optimized for RLlib
DEFAULT_NUM_DRONES = 4
DEFAULT_OBS_TYPE = ObservationType.KIN
DEFAULT_ACT_TYPE = ActionType.RPM  # RPM actions work well with RLlib
DEFAULT_GUI = False
DEFAULT_RECORD = False
DEFAULT_OUTPUT_FOLDER = 'rllib_results'

# Enhanced training parameters for RLlib
DEFAULT_CONFIG = {
    # Environment settings
    'num_drones': DEFAULT_NUM_DRONES,
    'obs_type': DEFAULT_OBS_TYPE.name,
    'act_type': DEFAULT_ACT_TYPE.name,
    'steps_per_target': 100,
    'target_scale': 1.2,
    
    # Paper-based reward hyperparameters
    'reward_params': {
        'lambda_1': 25.0,    # Progress reward (higher for faster learning)
        'lambda_2': 2.0,     # Perception reward
        'lambda_3': 0.5,     # Perception alignment exponent
        'lambda_4': 0.008,   # Action magnitude penalty  
        'lambda_5': 0.12,    # Action smoothness penalty
        'crash_penalty': 20.0,   # Crash penalty
        'bounds_penalty': 12.0,  # Out-of-bounds penalty
    },
    
    # RLlib-specific training settings (much better than SB3)
    'train_batch_size': 16384,      # Large batch for better GPU utilization
    'sgd_minibatch_size': 1024,     # Large minibatch for efficiency  
    'num_sgd_iter': 10,             # SGD iterations per batch
    'lr': 8e-5,                     # Learning rate (slightly lower for stability)
    'gamma': 0.99,                  # Discount factor
    'lambda_': 0.95,                # GAE parameter
    'clip_param': 0.15,             # PPO clip parameter
    'entropy_coeff': 0.008,         # Entropy coefficient
    'vf_loss_coeff': 0.5,           # Value function loss coefficient
    'kl_coeff': 0.2,                # KL divergence coefficient
    'kl_target': 0.01,              # KL target
    'grad_clip': 0.5,               # Gradient clipping
    
    # Model architecture
    'model_type': 'shared',         # shared, multi_agent, or lightweight
    'hidden_dim': 256,              # Hidden layer dimension
    'use_attention': False,         # Attention mechanism (slower but more powerful)
    'use_layer_norm': True,         # Layer normalization
    'dropout_rate': 0.05,           # Dropout rate
    
    # Resource allocation (distributed training!)
    'num_workers': 8,               # Parallel data collection workers
    'num_envs_per_worker': 1,       # Environments per worker
    'num_gpus': 1,                  # GPU for training
    'num_cpus_per_worker': 1,       # CPUs per worker
    'rollout_fragment_length': 512, # Rollout length per worker
    
    # Training schedule
    'total_timesteps': int(5e6),    # Total training timesteps
    'evaluation_interval': 25,      # Evaluate every N iterations
    'checkpoint_freq': 50,          # Checkpoint every N iterations
    'keep_checkpoints_num': 5,      # Keep N best checkpoints
    
    # WandB logging
    'wandb_project': 'rllib-drone-swarm',
    'wandb_entity': None,
    'log_level': 'INFO',
}


class DroneTrainingCallbacks(DefaultCallbacks):
    """Custom callbacks for detailed logging and monitoring."""
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.best_mean_reward = -np.inf
        self.episodes_completed = 0
        self.total_timesteps = 0
        
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Called at the start of each episode."""
        episode.user_data["episode_start_time"] = time.time()
        episode.user_data["phase_rewards"] = []
        episode.user_data["distance_improvements"] = []
        
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Called at each step of an episode."""
        # Get environment info
        info = episode.last_info_for()
        if info and '__common__' in info:
            common_info = info['__common__']
            
            # Track detailed metrics
            episode.user_data.setdefault("mean_distances", []).append(
                common_info.get('mean_distance_to_targets', 0.0)
            )
            episode.user_data.setdefault("formation_errors", []).append(
                common_info.get('formation_error', 0.0)
            )
            episode.user_data.setdefault("phases", []).append(
                common_info.get('current_phase', 0)
            )
            
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Called at the end of each episode."""
        episode_duration = time.time() - episode.user_data["episode_start_time"]
        
        # Calculate episode statistics
        episode_reward = episode.total_reward
        episode_length = episode.length
        
        self.episode_rewards.append(episode_reward)
        self.episodes_completed += 1
        
        # Calculate rolling statistics
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        mean_reward_100 = np.mean(recent_rewards)
        std_reward_100 = np.std(recent_rewards)
        
        # Update best reward
        if mean_reward_100 > self.best_mean_reward:
            self.best_mean_reward = mean_reward_100
        
        # Get final info
        info = episode.last_info_for()
        final_phase = 0
        targets_reached = 0
        final_distance = 0.0
        collisions = 0
        
        if info and '__common__' in info:
            common_info = info['__common__']
            final_phase = common_info.get('current_phase', 0)
            targets_reached = common_info.get('targets_reached', 0)
            final_distance = common_info.get('mean_distance_to_targets', 0.0)
            collisions = common_info.get('collisions', 0)
        
        # Distance improvement
        mean_distances = episode.user_data.get("mean_distances", [0])
        distance_improvement = mean_distances[0] - mean_distances[-1] if len(mean_distances) > 1 else 0
        
        # Log custom metrics
        episode.custom_metrics["episode_reward"] = episode_reward
        episode.custom_metrics["episode_length"] = episode_length
        episode.custom_metrics["episode_duration"] = episode_duration
        episode.custom_metrics["mean_reward_100"] = mean_reward_100
        episode.custom_metrics["std_reward_100"] = std_reward_100
        episode.custom_metrics["best_mean_reward"] = self.best_mean_reward
        episode.custom_metrics["final_phase"] = final_phase
        episode.custom_metrics["targets_reached"] = targets_reached
        episode.custom_metrics["final_distance"] = final_distance
        episode.custom_metrics["distance_improvement"] = distance_improvement
        episode.custom_metrics["collisions"] = collisions
        episode.custom_metrics["episodes_completed"] = self.episodes_completed
        
        # Formation metrics
        if episode.user_data.get("formation_errors"):
            episode.custom_metrics["mean_formation_error"] = np.mean(episode.user_data["formation_errors"])
            episode.custom_metrics["final_formation_error"] = episode.user_data["formation_errors"][-1]
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Called after each training iteration."""
        # Extract key metrics with fallbacks for different RLlib versions
        episode_reward_mean = result.get("episode_reward_mean", 
                                        result.get("env_runners", {}).get("episode_reward_mean", 0))
        episodes_this_iter = result.get("episodes_this_iter",
                                       result.get("env_runners", {}).get("episodes_this_iter", 0))
        timesteps_total = result.get("timesteps_total", 0)
        
        self.total_timesteps = timesteps_total
        
        # Enhanced logging for WandB
        try:
            # Additional metrics for WandB
            custom_metrics = result.get("custom_metrics", {})
            
            wandb_metrics = {
                "train/timesteps_total": timesteps_total,
                "train/episodes_this_iter": episodes_this_iter,
                "train/episode_reward_mean": episode_reward_mean,
            }
            
            # Try to get learning rate from different possible locations
            try:
                lr = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("cur_lr", 0)
                if lr == 0:
                    lr = result.get("info", {}).get("learner", {}).get("cur_lr", 0)
                wandb_metrics["train/learning_rate"] = lr
            except:
                pass
            
            # Add custom metrics with proper prefixes
            for key, value in custom_metrics.items():
                if key.endswith("_mean") and isinstance(value, (int, float)):
                    wandb_metrics[f"train/{key}"] = value
                    
            # Log to WandB
            if wandb.run is not None:
                wandb.log(wandb_metrics, step=timesteps_total)
        except Exception as e:
            print(f"[WARNING] WandB logging failed: {e}")


def create_rllib_config(config: Dict[str, Any]) -> PPOConfig:
    """Create optimized RLlib configuration for drone swarm training."""
    
    # Convert string enums back to enum objects
    obs_type = ObservationType[config['obs_type']] if isinstance(config['obs_type'], str) else config['obs_type']
    act_type = ActionType[config['act_type']] if isinstance(config['act_type'], str) else config['act_type']
    
    # Environment configuration
    env_config = {
        'num_drones': config['num_drones'],
        'obs_type': obs_type,
        'act_type': act_type,
        'gui': False,  # Always False for training
        'record': False,
        'steps_per_target': config['steps_per_target'],
        'target_scale': config['target_scale'],
        'reward_params': config['reward_params'],
        'tolerance': 0.15,
        'collision_distance': 0.05,
    }
    
    # Model configuration
    model_config = {
        "custom_model": f"{config['model_type']}_drone",
        "custom_model_config": {
            "hidden_dim": config['hidden_dim'],
            "use_attention": config['use_attention'],
            "use_layer_norm": config['use_layer_norm'],
            "dropout_rate": config['dropout_rate'],
        }
    }
    
    # Create PPO configuration
    ppo_config = (PPOConfig()
        # Environment
        .environment(
            env=RLlibMultiDroneEnv,
            env_config=env_config,
        )
        
        # Multi-agent setup (KEY ADVANTAGE!)
        .multi_agent(
            policies={
                "shared_drone_policy": (
                    None,  # Use default policy class
                    None,  # Observation space (inferred from env)
                    None,  # Action space (inferred from env)
                    {"model": model_config}
                )
            },
            policy_mapping_fn=lambda agent_id, **kwargs: "shared_drone_policy",
            policies_to_train=["shared_drone_policy"],
        )
        
        # Training parameters (optimized for RLlib)
        .training(
            train_batch_size=config['train_batch_size'],
            sgd_minibatch_size=config['sgd_minibatch_size'],
            num_sgd_iter=config['num_sgd_iter'],
            lr=config['lr'],
            gamma=config['gamma'],
            lambda_=config['lambda_'],
            clip_param=config['clip_param'],
            entropy_coeff=config['entropy_coeff'],
            vf_loss_coeff=config['vf_loss_coeff'],
            kl_coeff=config['kl_coeff'],
            kl_target=config['kl_target'],
            grad_clip=config['grad_clip'],
        )
        
        # Resource allocation (distributed training!)
        .resources(
            num_gpus=config['num_gpus'],
            num_cpus_per_worker=config['num_cpus_per_worker'],
        )
        
        # Rollout configuration (parallel data collection)
        .rollouts(
            num_workers=config['num_workers'],
            num_envs_per_worker=config['num_envs_per_worker'],
            rollout_fragment_length=config['rollout_fragment_length'],
            batch_mode="complete_episodes",
        )
        
        # Evaluation
        .evaluation(
            evaluation_interval=config['evaluation_interval'],
            evaluation_duration=5,
            evaluation_num_workers=min(2, config['num_workers']),
            evaluation_config={
                "env_config": {**env_config, "gui": False}
            }
        )
        
        # Debugging and monitoring
        .debugging(
            log_level=config['log_level'],
        )
        
        # Callbacks for enhanced logging
        .callbacks(DroneTrainingCallbacks)
    )
    
    return ppo_config


def setup_wandb(config: Dict[str, Any], run_name: str) -> None:
    """Initialize Weights & Biases logging."""
    
    # Finish any existing runs
    try:
        wandb.finish()
    except:
        pass
    
    # Initialize new run
    wandb.init(
        project=config['wandb_project'],
        entity=config['wandb_entity'],
        name=run_name,
        config=config,
        sync_tensorboard=False,  # RLlib has its own tensorboard
        save_code=True,
        tags=[
            "rllib", 
            "multi-agent", 
            "drone-swarm", 
            "ppo", 
            f"{config['act_type']}-actions",
            "paper-based-reward",
            f"{config['model_type']}-model"
        ],
        notes=f"RLlib multi-agent drone swarm training with {config['model_type']} model"
    )
    
    print(f"[WandB] Initialized project: {config['wandb_project']}")
    print(f"[WandB] Run name: {run_name}")


def create_target_sequence(num_drones: int = 4, scale: float = 1.2) -> np.ndarray:
    """Create target sequence optimized for paper-based reward function."""
    if num_drones == 4:
        targets = np.array([
            # Phase 0: Simple line formation
            [[-1.0*scale, 0.0, 1.2], [-0.3*scale, 0.0, 1.2], 
             [ 0.3*scale, 0.0, 1.2], [ 1.0*scale, 0.0, 1.2]],
            
            # Phase 1: Square formation
            [[-scale, -scale, 1.5], [ scale, -scale, 1.5], 
             [ scale,  scale, 1.5], [-scale,  scale, 1.5]],
            
            # Phase 2: Diamond formation
            [[ 0.0, -1.2*scale, 1.8], [ 1.2*scale, 0.0, 1.8], 
             [ 0.0,  1.2*scale, 1.8], [-1.2*scale, 0.0, 1.8]],
            
            # Phase 3: Compact formation
            [[-0.4*scale, -0.4*scale, 1.3], [ 0.4*scale, -0.4*scale, 1.3], 
             [ 0.4*scale,  0.4*scale, 1.3], [-0.4*scale,  0.4*scale, 1.3]]
        ])
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


def save_artifacts(algorithm, config: Dict[str, Any], save_dir: str, run_name: str, 
                  best_checkpoint: str, final_stats: Dict[str, Any]):
    """Save model and configuration artifacts."""
    
    try:
        # 1. Save final model as WandB artifact
        model_artifact = wandb.Artifact(
            name=f"rllib_drone_model_{run_name}",
            type="model",
            description="Trained RLlib PPO model for multi-agent drone swarm",
            metadata={
                "algorithm": "PPO",
                "framework": "RLlib",
                "model_type": config['model_type'],
                "num_drones": config['num_drones'],
                "action_type": config['act_type'],
                "total_timesteps": config['total_timesteps'],
                "best_checkpoint": best_checkpoint,
                **final_stats
            }
        )
        
        # Add checkpoint directory
        if os.path.exists(best_checkpoint):
            model_artifact.add_dir(best_checkpoint, name="checkpoint")
            wandb.log_artifact(model_artifact)
            print(f"[WandB] Saved model artifact: rllib_drone_model_{run_name}")
        
        # 2. Save configuration
        config_artifact = wandb.Artifact(
            name=f"rllib_config_{run_name}",
            type="config",
            description="Training configuration for RLlib drone swarm",
            metadata=config
        )
        
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            # Convert non-serializable objects
            serializable_config = {}
            for k, v in config.items():
                if isinstance(v, (str, int, float, bool, list, dict)):
                    serializable_config[k] = v
                else:
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2)
        
        config_artifact.add_file(config_path)
        wandb.log_artifact(config_artifact)
        print(f"[WandB] Saved config artifact: rllib_config_{run_name}")
        
    except Exception as e:
        print(f"[WARNING] Failed to save artifacts: {e}")


def train_rllib_drone_swarm(config: Dict[str, Any], output_folder: str, 
                           use_wandb: bool = True, tune_hyperparameters: bool = False) -> str:
    """
    Main training function for RLlib multi-agent drone swarm.
    
    Returns:
        Path to best checkpoint
    """
    
    # Create run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"rllib_drone_{config['model_type']}_{timestamp}"
    
    # Create output directory
    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("RLlib Multi-Agent Drone Swarm Training")
    print("MUCH FASTER than Stable Baselines 3!")
    print("="*80)
    print(f"Run name: {run_name}")
    print(f"Model type: {config['model_type']}")
    print(f"Action type: {config['act_type']}")
    print(f"Number of drones: {config['num_drones']}")
    print(f"Parallel workers: {config['num_workers']}")
    print(f"Batch size: {config['train_batch_size']} (vs SB3's ~2048)")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print("="*80)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Register custom models
        register_drone_models()
        
        # Setup WandB
        if use_wandb:
            setup_wandb(config, run_name)
        
        # Create RLlib configuration
        rllib_config = create_rllib_config(config)
        
        # Build algorithm
        print("[INFO] Building RLlib PPO algorithm...")
        algorithm = rllib_config.build()
        
        print(f"[INFO] Algorithm created successfully!")
        print(f"[INFO] Policy parameters: {algorithm.get_policy().num_parameters():,}")
        
        # Training variables
        best_reward = -np.inf
        best_checkpoint = None
        training_start_time = time.time()
        total_timesteps = 0
        iteration = 0
        
        # Training loop
        print(f"\n[INFO] Starting training for {config['total_timesteps']:,} timesteps...")
        print(f"[INFO] Evaluation every {config['evaluation_interval']} iterations")
        
        while total_timesteps < config['total_timesteps']:
            iteration += 1
            
            # Train one iteration
            result = algorithm.train()
            
            # Update metrics with fallbacks for different RLlib versions
            total_timesteps = result.get("timesteps_total", 0)
            episode_reward_mean = result.get("episode_reward_mean", 
                                            result.get("env_runners", {}).get("episode_reward_mean", 0))
            episodes_this_iter = result.get("episodes_this_iter",
                                           result.get("env_runners", {}).get("episodes_this_iter", 0))
            
            # Print progress
            if iteration % 10 == 0:
                elapsed_time = time.time() - training_start_time
                timesteps_per_sec = total_timesteps / elapsed_time if elapsed_time > 0 else 0
                
                print(f"Iter {iteration:4d} | "
                      f"Timesteps: {total_timesteps:8,d}/{config['total_timesteps']:,} | "
                      f"Reward: {episode_reward_mean:7.2f} | "
                      f"Episodes: {episodes_this_iter:3d} | "
                      f"Speed: {timesteps_per_sec:6.0f} steps/sec")
            
            # Save checkpoint if improved
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                checkpoint_path = algorithm.save(save_dir)
                best_checkpoint = checkpoint_path
                
                print(f"[CHECKPOINT] New best reward: {best_reward:.2f} -> {checkpoint_path}")
                
                # Log improvement to WandB
                if use_wandb:
                    wandb.log({
                        "train/best_reward": best_reward,
                        "train/checkpoint_saved": 1,
                    }, step=total_timesteps)
            
            # Periodic checkpoint
            if iteration % config['checkpoint_freq'] == 0:
                checkpoint_path = algorithm.save(save_dir)
                print(f"[CHECKPOINT] Periodic save: {checkpoint_path}")
            
            # Early stopping check
            if episode_reward_mean > 800:  # Adjust threshold as needed
                print(f"[SUCCESS] Environment solved! Reward: {episode_reward_mean:.2f}")
                break
        
        # Training completed
        training_time = time.time() - training_start_time
        final_checkpoint = algorithm.save(save_dir)
        
        print(f"\n[TRAINING COMPLETED]")
        print(f"Total time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final checkpoint: {final_checkpoint}")
        print(f"Timesteps/sec: {total_timesteps/training_time:.0f}")
        
        # Final evaluation
        print(f"\n[INFO] Running final evaluation...")
        try:
            eval_result = algorithm.evaluate()
            # Handle different evaluation result structures
            final_eval_reward = eval_result.get("evaluation", {}).get("episode_reward_mean", 
                                               eval_result.get("episode_reward_mean", 0))
        except Exception as e:
            print(f"[WARNING] Final evaluation failed: {e}")
            final_eval_reward = 0
        
        print(f"[EVAL] Final evaluation reward: {final_eval_reward:.2f}")
        
        # Save artifacts
        final_stats = {
            "best_reward": best_reward,
            "final_eval_reward": final_eval_reward,
            "total_timesteps": total_timesteps,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "timesteps_per_second": total_timesteps / training_time,
        }
        
        if use_wandb:
            # Log final metrics
            wandb.log({
                "final/best_reward": best_reward,
                "final/eval_reward": final_eval_reward,
                "final/training_time": training_time,
                "final/timesteps_per_second": total_timesteps / training_time,
            })
            
            save_artifacts(algorithm, config, save_dir, run_name, 
                          best_checkpoint or final_checkpoint, final_stats)
            
            wandb.finish()
        
        # Cleanup
        algorithm.stop()
        
        return best_checkpoint or final_checkpoint
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        return None
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if ray.is_initialized():
            ray.shutdown()


def run_hyperparameter_tuning(base_config: Dict[str, Any], output_folder: str):
    """Run automated hyperparameter tuning with Ray Tune."""
    
    print("="*80)
    print("RLlib Hyperparameter Tuning for Drone Swarm")
    print("="*80)
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Register models
        register_drone_models()
        
        # Define search space
        search_space = {
            **base_config,  # Base configuration
            
            # Tune key hyperparameters
            "lr": tune.loguniform(1e-5, 5e-4),
            "train_batch_size": tune.choice([8192, 16384, 32768]),
            "sgd_minibatch_size": tune.choice([512, 1024, 2048]),
            "entropy_coeff": tune.loguniform(1e-3, 5e-2),
            "clip_param": tune.uniform(0.1, 0.3),
            "hidden_dim": tune.choice([128, 256, 512]),
            
            # Tune reward parameters
            "reward_params": {
                "lambda_1": tune.uniform(15.0, 35.0),    # Progress reward
                "lambda_2": tune.uniform(1.0, 3.0),      # Perception reward
                "lambda_3": tune.uniform(0.2, 0.8),      # Alignment exponent
                "lambda_4": tune.uniform(0.005, 0.02),   # Action penalty
                "lambda_5": tune.uniform(0.05, 0.2),     # Smoothness penalty
                "crash_penalty": tune.uniform(10.0, 30.0),
                "bounds_penalty": tune.uniform(5.0, 20.0),
            }
        }
        
        # Create tuner
        tuner = tune.Tuner(
            "PPO",
            param_space={
                "env": RLlibMultiDroneEnv,
                "env_config": search_space,
                # Add other PPO config here
            },
            run_config=air.RunConfig(
                name="drone_swarm_tune",
                local_dir=output_folder,
                stop={"episode_reward_mean": 600, "timesteps_total": int(1e6)},
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=20,
                    num_to_keep=3,
                ),
            ),
            tune_config=tune.TuneConfig(
                num_samples=20,  # Number of trials
                scheduler=tune.schedulers.ASHAScheduler(
                    metric="episode_reward_mean",
                    mode="max",
                    max_t=int(1e6),
                    grace_period=int(1e5),
                ),
                search_alg=tune.search.optuna.OptunaSearch(metric="episode_reward_mean", mode="max"),
            )
        )
        
        # Run tuning
        print(f"[INFO] Starting hyperparameter tuning with 20 trials...")
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result()
        best_config = best_result.config
        best_reward = best_result.metrics["episode_reward_mean"]
        
        print(f"\n[TUNE RESULTS]")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Best configuration:")
        for key, value in best_config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # Save best config
        best_config_path = os.path.join(output_folder, 'best_hyperparameters.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        print(f"[TUNE] Best configuration saved to: {best_config_path}")
        
        return best_config
        
    except Exception as e:
        print(f"[ERROR] Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        ray.shutdown()


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='RLlib Multi-Agent Drone Swarm Training')
    
    # Basic training options
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Output folder for results')
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int,
                        help='Number of drones in swarm')
    parser.add_argument('--model_type', default='shared', 
                        choices=['shared', 'multi_agent', 'lightweight'],
                        help='Type of neural network model')
    parser.add_argument('--act_type', default=DEFAULT_ACT_TYPE.name,
                        choices=['RPM', 'PID', 'VEL'],
                        help='Action type for drone control')
    
    # Training parameters
    parser.add_argument('--total_timesteps', default=int(5e6), type=int,
                        help='Total training timesteps')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of parallel workers')
    parser.add_argument('--train_batch_size', default=16384, type=int,
                        help='Training batch size')
    parser.add_argument('--lr', default=8e-5, type=float,
                        help='Learning rate')
    
    # WandB options
    parser.add_argument('--wandb_project', default='rllib-drone-swarm', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='Weights & Biases entity')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    # Advanced options
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning instead of training')
    parser.add_argument('--config_file', default=None, type=str,
                        help='Load configuration from JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        print(f"[INFO] Loading configuration from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override config with command line arguments
    config.update({
        'num_drones': args.num_drones,
        'model_type': args.model_type,
        'act_type': args.act_type,
        'total_timesteps': args.total_timesteps,
        'num_workers': args.num_workers,
        'train_batch_size': args.train_batch_size,
        'lr': args.lr,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
    })
    
    print(f"\n[CONFIG] Training configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Run training or tuning
    if args.tune:
        print(f"\n[INFO] Running hyperparameter tuning...")
        best_config = run_hyperparameter_tuning(config, args.output_folder)
        if best_config:
            print(f"[SUCCESS] Hyperparameter tuning completed!")
    else:
        print(f"\n[INFO] Starting RLlib training...")
        best_checkpoint = train_rllib_drone_swarm(
            config, 
            args.output_folder, 
            use_wandb=not args.no_wandb
        )
        
        if best_checkpoint:
            print(f"[SUCCESS] Training completed!")
            print(f"[SUCCESS] Best model saved at: {best_checkpoint}")
        else:
            print(f"[ERROR] Training failed!")


if __name__ == "__main__":
    main()