#!/usr/bin/env python3
"""
train_multi_td3.py - Multi-algorithm training script with ADAPTIVE DIFFICULTY
Optimized for single drone with increasing challenge, supports KIN_DEPTH obs.
Now includes DDPG algorithm support.
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gymnasium as gym # Changed from gym to gymnasium
import wandb
import torch
from typing import Optional, Dict, Any
from stable_baselines3 import TD3, SAC, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from stable_baselines3.common.vec_env import VecNormalize # Not used currently for obs norm

from wandb.integration.sb3 import WandbCallback

# Import your custom extractors
from multi_agent_extractors_td3 import create_multiagent_model # This now handles KinDepthExtractor
# Import your environment
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
# from gym_pybullet_drones.envs.HoverAviary import HoverAviary # Not used for main training
# from gym_pybullet_drones.utils.Logger import Logger # Not used
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType # Ensure KIN_DEPTH is here or defined in MultiTargetAviary

# Define ObservationType locally if not properly imported with KIN_DEPTH
try:
    from gym_pybullet_drones.utils.enums import ObservationType
    if not hasattr(ObservationType, 'KIN_DEPTH'):
        print("Redefining ObservationType locally to include KIN_DEPTH for train script.")
        from enum import Enum
        class TrainObservationType(Enum):
            KIN = "kin"; RGB = "rgb"; KIN_DEPTH = "kin_depth"
        ObservationType = TrainObservationType
except ImportError:
    print("Failed to import ObservationType, defining locally for train script.")
    from enum import Enum
    class TrainObservationType(Enum):
        KIN = "kin"; RGB = "rgb"; KIN_DEPTH = "kin_depth"
    ObservationType = TrainObservationType


# Default settings
DEFAULT_GUI           = False # Typically False for training
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType.KIN # Changed to KIN_DEPTH
DEFAULT_ACT           = ActionType.RPM 
DEFAULT_DRONES        = 1 # Focus on single drone for adaptive KIN_DEPTH
DEFAULT_DURATION_SEC  = 6.0 # Increased for more learning time per ep
DEFAULT_ALGORITHM     = 'ppo'
DEFAULT_EXTRACTOR     = 'matrix' # This will be overridden by KinDepthExtractor if obs is KIN_DEPTH
DEFAULT_FEATURES_DIM  = 64 # For KinDepthExtractor, this is the final output dim

# New default obstacle settings
DEFAULT_ADD_OBSTACLES = False
DEFAULT_OBS_PROB      = 0.6
DEFAULT_OBSTACLE_SIZE = 0.2

# Using existing AdaptiveDifficultyWandbCallback, ensure it handles info dict robustly.
class AdaptiveDifficultyWandbCallback(BaseCallback):
    """Enhanced callback for adaptive difficulty system with detailed logging"""
    def __init__(self, algorithm: str, log_freq: int = 100, save_freq: int = 25000, 
                 min_reward_improvement: float = 10.0, verbose: int = 0):
        super().__init__(verbose)
        self.algorithm = algorithm.upper(); self.log_freq = log_freq; self.save_freq = save_freq
        self.min_reward_improvement = min_reward_improvement; self.episode_rewards = []
        self.episode_lengths = []; self.episode_successes = []; self.current_episode_reward = 0
        self.current_episode_length = 0; self.best_mean_reward = -np.inf; self.models_saved = 0
        self.target_radius_history = []; self.success_rate_history = []
        self.is_off_policy = algorithm.lower() in ['td3', 'sac', 'ddpg']  # Added DDPG to off-policy algorithms
        if verbose > 0: print(f"[AdaptiveDifficultyCallback] Initialized for {self.algorithm}, Off-policy: {self.is_off_policy}")
    def _on_step(self) -> bool:
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward; self.current_episode_length += 1
        if self.locals.get('dones', [False])[0]:
            #print('logging_1')
            self.episode_rewards.append(self.current_episode_reward); self.episode_lengths.append(self.current_episode_length)
            infos = self.locals.get('infos', [{}]); info = infos[0] if len(infos) > 0 else {}
            episode_success = info.get('episode_success', False); self.episode_successes.append(episode_success)
            target_radius = info.get('target_radius', 0.0); success_rate = info.get('success_rate_last_100', 0.0)
            total_episodes = info.get('total_episodes', 0); self.target_radius_history.append(target_radius)
            self.success_rate_history.append(success_rate)
            mean_reward_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0
            mean_success_100 = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) > 0 else 0
            log_data = {
                f'{self.algorithm}/episode_reward': self.current_episode_reward, f'{self.algorithm}/episode_length': self.current_episode_length,
                f'{self.algorithm}/episode_success': episode_success, f'{self.algorithm}/mean_episode_reward_last_100': mean_reward_100,
                f'{self.algorithm}/mean_success_rate_last_100': mean_success_100, f'{self.algorithm}/episodes_completed': len(self.episode_rewards),
                f'{self.algorithm}/best_mean_reward': self.best_mean_reward, f'{self.algorithm}/adaptive_target_radius': target_radius,
                f'{self.algorithm}/adaptive_success_rate': success_rate, f'{self.algorithm}/adaptive_total_episodes': total_episodes,
                f'{self.algorithm}/min_distance_to_target': info.get('min_distance_to_target', np.inf),
            }
            
            # Add obstacle collision metrics if available
            if 'obstacles' in info:
                log_data[f'{self.algorithm}/num_obstacles'] = info.get('num_obstacles', 0)
                log_data[f'{self.algorithm}/min_obstacle_distance'] = info.get('min_obstacle_distance', np.inf)
                
            wandb.log(log_data, step=self.num_timesteps)
            if (mean_reward_100 > self.best_mean_reward and (mean_reward_100 - self.best_mean_reward) >= self.min_reward_improvement and len(self.episode_rewards) >= 100):
                #print('fuck')
                #print(mean_reward_100, self.best_mean_reward, self.min_reward_improvement, len(self.episode_rewards))
                self.best_mean_reward = mean_reward_100; self.models_saved += 1
                self._save_model_artifact(f"{self.algorithm}_model_r{mean_reward_100:.1f}_rad{target_radius:.2f}")
                if self.verbose > 0: print(f"\n[Model Saved] {self.algorithm} at ts {self.num_timesteps}: Reward: {mean_reward_100:.2f}, Radius: {target_radius:.2f}")
            self.current_episode_reward = 0; self.current_episode_length = 0
        if self.n_calls % self.log_freq == 0:
            #print('logging_2')
            log_dict = {f'{self.algorithm}/timestep': self.num_timesteps, f'{self.algorithm}/models_saved': self.models_saved}
            if self.is_off_policy and hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
                log_dict[f'{self.algorithm}/replay_buffer_size'] = self.model.replay_buffer.size()
            wandb.log(log_dict, step=self.num_timesteps)
        if self.n_calls > 0 and self.n_calls % self.save_freq == 0 :
            #print('logging_3')
            current_radius = self.target_radius_history[-1] if self.target_radius_history else 0.0
            self._save_model_artifact(f"{self.algorithm}_checkpoint_{self.num_timesteps}_rad{current_radius:.2f}")
        return True
    def _save_model_artifact(self, artifact_name: str):
        #print('logging_4')
        try:
            temp_path = f"/tmp/{artifact_name}.zip"; self.model.save(temp_path)
            current_radius = self.target_radius_history[-1] if self.target_radius_history else 0.0
            current_success = self.success_rate_history[-1] if self.success_rate_history else 0.0
            mean_rwd = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else (np.mean(self.episode_rewards) if self.episode_rewards else 0)
            mean_succ = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else (np.mean(self.episode_successes) if self.episode_successes else 0)
            artifact = wandb.Artifact(
                name=artifact_name, type="model", description=f"{self.algorithm} model at ts {self.num_timesteps}",
                metadata={
                    "algorithm": self.algorithm, "timestep": self.num_timesteps, "episodes": len(self.episode_rewards),
                    "mean_reward_100": mean_rwd, "success_rate_100": mean_succ,
                    "target_radius": current_radius, "adaptive_success_rate": current_success, "models_saved": self.models_saved,
                })
            artifact.add_file(temp_path); wandb.log_artifact(artifact)
            if os.path.exists(temp_path): os.remove(temp_path)
            if self.verbose > 0: print(f"[WandB] Saved {self.algorithm} model artifact: {artifact_name}")
        except Exception as e: print(f"[WARNING] Failed to save model artifact {artifact_name}: {e}")

def create_action_noise(env, noise_type: str = "normal", noise_std: float = 0.1):
    n_actions = env.action_space.shape[-1]
    if noise_type == "normal": return NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    if noise_type == "ornstein-uhlenbeck": return OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    return None

def get_algorithm_config(algorithm: str) -> Dict[str, Any]:
    configs = {
        'td3': {'total_timesteps': int(1e7), 'learning_rate': 1e-4, 'buffer_size': 500_000, # Reduced buffer for faster iteration
                'learning_starts': 10000, 'batch_size': 256, 'tau': 0.005, 'gamma': 0.98, # Adjusted gamma
                'train_freq': 1, 'gradient_steps': 1, 'policy_delay': 2,
                'target_policy_noise': 0.2, 'target_noise_clip': 0.5, 'eval_freq': 5000, 'n_eval_episodes': 5,}, # Increased eval_freq
        'ddpg': {'total_timesteps': int(1e7), 'learning_rate': 1e-4, 'buffer_size': 500_000,
                'learning_starts': 10000, 'batch_size': 256, 'tau': 0.005, 'gamma': 0.98,
                'train_freq': 1, 'gradient_steps': 1, 'eval_freq': 5000, 'n_eval_episodes': 5,},
        'sac': {'total_timesteps': int(1e7), 'learning_rate': 3e-4, 'buffer_size': 500_000,
                'learning_starts': 10000, 'batch_size': 256, 'tau': 0.005, 'gamma': 0.98, 
                'train_freq': 1, 'gradient_steps': 1, 'ent_coef': 'auto', 'target_entropy': 'auto',
                'eval_freq': 25000, 'n_eval_episodes': 5,},
        'ppo': {'total_timesteps': int(1e7), 'learning_rate': 3e-3, 'n_steps': 2048, 
                'batch_size': 128, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 
                'clip_range': 0.2, 'ent_coef': 0.001, 'eval_freq': 100000, 'n_eval_episodes': 2,}
    }
    return configs.get(algorithm.lower(), configs['td3'])
#python train_multi_td3.py --algorithm ddpg --num_drones 1 --features_dim 32  --num_vec_envs 4 --ctrl_freq 40
def run(algorithm: str, output_folder: str, gui: bool, record_video: bool,
        extractor_type: str, features_dim: int, num_drones: int,
        wandb_project: str, wandb_entity: Optional[str],
        noise_type: str = "normal", noise_std: float = 0.1,
        normalize_observations: bool = False, num_vec_envs: int = 1, # Default to 1 vec_env for Dict obs space initially
        ctrl_freq: int = 30,
        add_obstacles: bool = True,  # New parameter for obstacles
        obs_prob: float = 0.5,        # New parameter for obstacle density
        obstacle_size: float = 0.2):  # New parameter for obstacle size
    
    run_name = f"{algorithm}_{DEFAULT_OBS.value}_{extractor_type if DEFAULT_OBS != ObservationType.KIN_DEPTH else 'KinDepthExt'}_{'obs' if add_obstacles else 'noobs'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = get_algorithm_config(algorithm)
    config.update({
        'algorithm': algorithm, 'extractor_type': extractor_type, 'features_dim': features_dim,
        'num_drones': num_drones, 'obs_type': DEFAULT_OBS.value, 'act_type': DEFAULT_ACT.value,
        'noise_type': noise_type, 'noise_std': noise_std, 'normalize_observations': normalize_observations,
        'num_vec_envs': num_vec_envs, 'adaptive_difficulty': True, 'episode_length_sec': DEFAULT_DURATION_SEC,
        'ctrl_freq': ctrl_freq,
        'add_obstacles': add_obstacles, 'obs_prob': obs_prob, 'obstacle_size': obstacle_size,  # Add obstacle parameters
    })
    
    # Initialize WandB
    if wandb.run is not None: # Finish previous run if any
        wandb.finish()
    tags = [algorithm.upper(), "ADAPTIVE", DEFAULT_OBS.value, f"{num_drones}-DRONE"]
    if add_obstacles:
        tags.append("OBSTACLES")
    wandb.init(project=wandb_project, entity=wandb_entity, name=run_name, config=config,
               sync_tensorboard=True, monitor_gym=True, save_code=True,
               tags=tags)
    
    save_dir = os.path.join(output_folder, wandb.run.name if wandb.run else run_name) # Use wandb run name for save dir
    os.makedirs(save_dir, exist_ok=True)
    
    adaptive_params = {
        'episode_length_sec': DEFAULT_DURATION_SEC, 'target_radius_start': 0.1, # Slightly larger start
        'target_radius_max': 3.0, 'target_radius_increment': 0.1, 'target_tolerance': 0.05,
        'success_threshold': 0.9, 'evaluation_window': 100, # Smaller window for faster adaptation
        'crash_penalty': 400.0, 'bounds_penalty': 100.0, 'lambda_distance': 20, #100.0,
        'lambda_angle': 1, 'pyb_freq': 240, 'ctrl_freq': ctrl_freq, # Pass ctrl_freq
        'individual_target_reward': 400.0, # Increased reward
        'add_obstacles': add_obstacles,    # New obstacle parameters
        'obs_prob': obs_prob,
        'obstacle_size': obstacle_size,
    }
    
    print(f"\n[INFO] {algorithm.upper()} Training with {DEFAULT_OBS.value} observations.")
    print(f"[INFO] Num drones: {num_drones}, Ctrl Freq: {ctrl_freq} Hz, Episode length: {DEFAULT_DURATION_SEC:.1f}s")
    if add_obstacles:
        print(f"[INFO] Obstacles enabled with density: {obs_prob:.2f}, size: {obstacle_size:.2f}m")
    
    def make_env_fn():
        env = MultiTargetAviary(
            num_drones=num_drones, obs=DEFAULT_OBS, act=DEFAULT_ACT,
            gui=False, record=False, **adaptive_params
        )
        return env#Monitor(env) # Monitor wrapper for SB3 callbacks
    
    # Create vectorized environments
    # For Dict observation space, num_vec_envs=1 is often simpler to start with.
    # SB3's VecEnv wrappers can handle Dict spaces.
    train_env = VecMonitor(make_vec_env(make_env_fn, n_envs=num_vec_envs, seed=0))
    
    # Normalize observations if specified (only for Box spaces usually, handle Dict carefully)
    # VecNormalize is tricky with Dict spaces. Usually applied per-key or not at all.
    # For now, assuming normalize_observations=False for KIN_DEPTH.
    if normalize_observations and isinstance(train_env.observation_space, gym.spaces.Box):
        print("[INFO] Normalizing observations (Box space).")
        # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)
        # Not using VecNormalize for Dict spaces by default, as it requires per-key handling
    elif normalize_observations:
         print("[WARNING] Observation normalization requested for Dict space, but not applied by default. Implement per-key normalization if needed.")

    eval_env_params = adaptive_params.copy()
    eval_env_params['record'] = False # No recording for standard eval
    eval_env = MultiTargetAviary(num_drones=num_drones, obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False, **eval_env_params)
    eval_env = Monitor(eval_env) # Wrap eval_env with Monitor for EvalCallback

    print(f"\n[INFO] Training Env Action space: {train_env.action_space}")
    print(f"[INFO] Training Env Observation space: {train_env.observation_space}")
    
    # Action noise for TD3/SAC/DDPG
    action_noise = create_action_noise(train_env, noise_type, noise_std) if algorithm.lower() in ['td3', 'sac', 'ddpg'] else None
    
    # Model creation
    model_params = config.copy()
    # Remove params not directly accepted by SB3 model constructors
    # AND also remove parameters that are explicitly passed to create_multiagent_model
    extra_params_to_pop = ['total_timesteps', 'eval_freq', 'n_eval_episodes',
                           'extractor_type', # Explicitly passed
                           'features_dim',   # Explicitly passed, REMOVE FROM model_params
                           'algorithm',      # Explicitly passed
                           'num_drones', 'obs_type', 'act_type',
                           'noise_type', 'noise_std', 'normalize_observations',
                           'num_vec_envs', 'adaptive_difficulty', 'episode_length_sec', 'ctrl_freq',
                           'add_obstacles', 'obs_prob', 'obstacle_size']  # Remove obstacle params
    for p in extra_params_to_pop: model_params.pop(p, None) # features_dim will now be popped

    if action_noise: model_params["action_noise"] = action_noise
    
    model = create_multiagent_model(
        train_env, # env
        algorithm=algorithm, # algorithm kwarg
        extractor_type=extractor_type,
        features_dim=features_dim,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, 'tb/'),
        **model_params
    )
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[INFO] Model created: {algorithm.upper()} with {total_params:,} parameters.")
    
    wandb_cb = AdaptiveDifficultyWandbCallback(algorithm=algorithm, log_freq=100, save_freq=config.get('eval_freq', 5000) * 2, verbose=1) # Save less frequently than eval
    eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.join(save_dir, 'best_model/'),
                           log_path=os.path.join(save_dir, 'eval_logs/'), eval_freq=config['eval_freq'],
                           n_eval_episodes=config['n_eval_episodes'], deterministic=True, render=False, verbose=1)
    cb_list = CallbackList([wandb_cb, eval_cb])
    
    wandb.log({
        'setup/algorithm': algorithm, 'setup/num_drones': num_drones, 'setup/obs_type': DEFAULT_OBS.value,
        'setup/episode_length_sec': DEFAULT_DURATION_SEC, 'setup/ctrl_freq': ctrl_freq,
        'setup/extractor_used': model.policy.features_extractor.__class__.__name__,
        'setup/features_dim': features_dim, 'setup/total_parameters': total_params,
        'setup/adaptive_start_radius': adaptive_params['target_radius_start'],
        'setup/adaptive_max_radius': adaptive_params['target_radius_max'],
        'setup/obstacles_enabled': add_obstacles,
        'setup/obstacle_probability': obs_prob if add_obstacles else 0,
        'setup/obstacle_size': obstacle_size if add_obstacles else 0,
    })
    
    print(f"\n[INFO] Starting {algorithm.upper()} training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=config['total_timesteps'], callback=cb_list, log_interval=100, progress_bar=True)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training_time = time.time() - start_time
        print(f"\n[INFO] Training duration: {training_time:.2f} seconds")

        # Final evaluation
        print("\n[INFO] Running final evaluation...")
        if eval_env: # Check if eval_env was successfully created
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
            print(f"[RESULTS] Final Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Get final adaptive difficulty metrics from a reset (or last logged)
            # For a proper final state, could run one more episode on eval_env and get info
            obs_final, info_final = eval_env.reset() 
            final_radius = info_final.get('target_radius', wandb_cb.target_radius_history[-1] if wandb_cb.target_radius_history else 0.0)
            final_success_rate = info_final.get('success_rate_last_100', wandb_cb.success_rate_history[-1] if wandb_cb.success_rate_history else 0.0)

            wandb.log({
                f'{algorithm}/final_mean_reward': mean_reward, f'{algorithm}/final_std_reward': std_reward,
                f'{algorithm}/final_target_radius': final_radius, f'{algorithm}/final_success_rate': final_success_rate,
                f'{algorithm}/training_time_seconds': training_time,})
        else:
            print("[WARNING] eval_env not available for final evaluation.")

        # Save final model
        final_model_path = os.path.join(save_dir, 'final_model.zip')
        model.save(final_model_path)
        print(f"[INFO] Final model saved to: {final_model_path}")

        # Demonstration (if GUI or record is True)
        if gui or record_video:
            print(f"\n[INFO] Running {algorithm.upper()} demonstration...")
            demo_env_params = adaptive_params.copy()
            demo_env_params['gui'] = gui
            demo_env_params['record'] = record_video # If true, BaseAviary handles video path
            demo_env_params['output_folder'] = save_dir # Save recordings in run folder
            
            demo_env = MultiTargetAviary(num_drones=num_drones, obs=DEFAULT_OBS, act=DEFAULT_ACT, **demo_env_params)
            
            for demo_episode in range(3):
                obs, info = demo_env.reset(seed=42 + demo_episode)
                ep_start_time = time.time()
                episode_reward = 0
                print(f"\n[DEMO {demo_episode + 1}] Target radius: {info['target_radius']:.2f}, Target: {info['current_targets'][0]}")
                if add_obstacles:
                    print(f"[DEMO {demo_episode + 1}] Obstacles: {info.get('num_obstacles', 0)}")
                
                max_demo_steps = int(demo_env.EPISODE_LEN_SEC * demo_env.CTRL_FREQ)
                for i in range(max_demo_steps + 50): # Run a bit longer
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = demo_env.step(action)
                    episode_reward += reward
                    
                    if gui: demo_env.render() # PyBullet's render is implicit with p.GUI, this is for other modes if any
                    if record_video and not gui: # For p.DIRECT recording, might need explicit frame saving in step
                        pass # BaseAviary handles this with self.RECORD
                    
                    sync(i, ep_start_time, demo_env.CTRL_TIMESTEP)
                    
                    if i % 30 == 0:
                        min_dist = info.get('min_distance_to_target', np.inf)
                        obs_dist = info.get('min_obstacle_distance', np.inf) if add_obstacles else np.inf
                        print(f"Demo Step {i:3d} | Target Dist: {min_dist:.3f} | Obstacle Dist: {obs_dist:.3f} | Reward: {reward:.2f}")
                    
                    done = terminated or truncated[0] if isinstance(truncated, tuple) else truncated
                    if done:
                        success = info.get('episode_success', False)
                        print(f"[DEMO {demo_episode + 1}] Finished: {'SUCCESS' if success else 'FAILURE'}, Reward: {episode_reward:.1f}")
                        break
            demo_env.close()
        
        wandb.finish()
        print(f"\n[INFO] {algorithm.upper()} adaptive difficulty training process finished!")
        print(f"[INFO] Results and logs saved to: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive difficulty drone training with KIN_DEPTH')
    parser.add_argument('--algorithm', default=DEFAULT_ALGORITHM, choices=['td3', 'sac', 'ppo', 'ddpg'], type=str, help='Reinforcement learning algorithm to use')
    parser.add_argument('--num_drones', default=DEFAULT_DRONES, type=int) # Should be 1 for KIN_DEPTH focus
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool)
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--extractor_type', default=DEFAULT_EXTRACTOR, choices=['matrix', 'attention', 'meanpool', 'kin_depth_auto'], type=str, help="For Box obs, or 'kin_depth_auto' to use KinDepthExtractor for Dict obs")
    parser.add_argument('--features_dim', default=DEFAULT_FEATURES_DIM, type=int)
    parser.add_argument('--noise_type', default='normal', choices=['normal', 'ornstein-uhlenbeck', 'none'], type=str)
    parser.add_argument('--noise_std', default=0.1, type=float)
    # parser.add_argument('--normalize_observations', default=False, type=str2bool) # Normalization for Dict is complex
    parser.add_argument('--num_vec_envs', default=1, type=int, help="Number of parallel envs (usually 1 for Dict space unless wrapper handles it well)")
    parser.add_argument('--ctrl_freq', default=30, type=int, help="Control frequency of the environment")
    parser.add_argument('--wandb_project', default='drone-adaptive-kin-depth', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str) # Your WandB username or team
    
    # New obstacle parameters
    parser.add_argument('--add_obstacles', default=DEFAULT_ADD_OBSTACLES, type=str2bool, help='Whether to add obstacles between drone and target')
    parser.add_argument('--obs_prob', default=DEFAULT_OBS_PROB, type=float, help='Probability/density of obstacles')
    parser.add_argument('--obstacle_size', default=DEFAULT_OBSTACLE_SIZE, type=float, help='Size of obstacles')
    
    args = parser.parse_args()

    if args.num_drones != 1 and DEFAULT_OBS == ObservationType.KIN_DEPTH:
        print(f"[WARNING] KIN_DEPTH with KinDepthExtractor is primarily designed for num_drones=1. Using num_drones={args.num_drones}. Ensure VecEnv and extractor handle batching correctly.")

    # The extractor_type 'kin_depth_auto' is conceptual for the argparser;
    # create_multiagent_model will choose KinDepthExtractor if observation space is Dict.
    # If obs space is Box, it will use the specified extractor_type.
    
    print_config = vars(args).copy()
    print_config["DEFAULT_OBS"] = DEFAULT_OBS.value
    print_config["DEFAULT_ACT"] = DEFAULT_ACT.value
    print_config["DEFAULT_DURATION_SEC"] = DEFAULT_DURATION_SEC
    print("="*60); print("Training Configuration:"); print("="*60)
    for key, value in print_config.items(): print(f"{key:<25}: {value}")
    print("="*60)

    run(algorithm=args.algorithm, output_folder=args.output_folder, gui=args.gui, record_video=args.record_video,
        extractor_type=args.extractor_type, features_dim=args.features_dim, num_drones=args.num_drones,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, noise_type=args.noise_type,
        noise_std=args.noise_std, normalize_observations=False, # Keep False for Dict obs
        num_vec_envs=args.num_vec_envs, ctrl_freq=args.ctrl_freq,
        add_obstacles=args.add_obstacles, obs_prob=args.obs_prob, obstacle_size=args.obstacle_size)