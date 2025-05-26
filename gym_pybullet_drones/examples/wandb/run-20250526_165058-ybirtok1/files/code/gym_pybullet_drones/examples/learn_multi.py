#!/usr/bin/env python3
"""
train_multi_improved.py: Train a multi-drone swarm with improved target-following behavior
Enhanced with comprehensive WandB logging and model weight pushing after evaluations
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary  # Your improved environment
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Default settings
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType('kin')   # 'kin' or 'rgb'
DEFAULT_ACT           = ActionType('one_d_rpm')  # 'rpm','pid','vel','one_d_rpm','one_d_pid'
DEFAULT_DRONES        = 4
DEFAULT_DURATION_SEC  = 5.0  # seconds per target phase
NUM_VEC = 8


class EnhancedWandbCallback(BaseCallback):
    """
    Enhanced callback for comprehensive WandB logging of training metrics
    """
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Accumulate reward and length for current episode
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode metrics
            wandb.log({
                'train/episode_reward': self.current_episode_reward,
                'train/episode_length': self.current_episode_length,
                'train/mean_episode_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0,
                'train/episodes_completed': len(self.episode_rewards),
            }, step=self.num_timesteps)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log detailed metrics every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Get info from environment if available
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                
                # Log environment-specific metrics
                log_dict = {
                    'train/timestep': self.num_timesteps,
                    'train/current_reward': self.locals.get('rewards', [0])[0],
                }
                
                # Add custom environment metrics if available
                if 'phase' in info:
                    log_dict['train/current_phase'] = info['phase']
                if 'targets_reached' in info:
                    log_dict['train/targets_reached'] = np.sum(info['targets_reached'])
                if 'collision_count' in info:
                    log_dict['train/collision_count'] = info['collision_count']
                if 'distance_to_targets' in info:
                    log_dict['train/mean_distance_to_targets'] = np.mean(info['distance_to_targets'])
                if 'formation_error' in info:
                    log_dict['train/formation_error'] = info['formation_error']
                
                wandb.log(log_dict, step=self.num_timesteps)
            
            # Log policy metrics if available
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                logger_dict = {}
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        logger_dict[f'train/{key}'] = value
                
                if logger_dict:
                    wandb.log(logger_dict, step=self.num_timesteps)
        
        return True


class DetailedEvalCallback(EvalCallback):
    """Enhanced evaluation callback with detailed logging and model weight pushing"""
    
    def __init__(self, eval_env, save_freq_evals=1, push_to_wandb=True, **kwargs):
        """
        Parameters:
        - save_freq_evals: Save and push model weights every N evaluations (default: 1 = every evaluation)
        - push_to_wandb: Whether to push model weights to wandb as artifacts
        """
        super().__init__(eval_env, **kwargs)
        self.save_freq_evals = save_freq_evals
        self.push_to_wandb = push_to_wandb
        self.eval_count = 0
        self.model_save_path = kwargs.get('best_model_save_path', './models')
        
        # Ensure model save directory exists
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Check if evaluation occurred
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            # Get the latest evaluation results
            if len(self.evaluations_results) > 0:
                latest_results = self.evaluations_results[-1]
                mean_reward = np.mean(latest_results)
                std_reward = np.std(latest_results)
                min_reward = np.min(latest_results)
                max_reward = np.max(latest_results)
                
                # Log evaluation metrics
                wandb.log({
                    'eval/mean_reward': mean_reward,
                    'eval/std_reward': std_reward,
                    'eval/min_reward': min_reward,
                    'eval/max_reward': max_reward,
                    'eval/num_episodes': len(latest_results),
                    'eval/evaluation_count': self.eval_count,
                }, step=self.num_timesteps)
                
                print(f"[EVAL] Step {self.num_timesteps}: Mean={mean_reward:.2f}±{std_reward:.2f}, Range=[{min_reward:.2f}, {max_reward:.2f}]")
                
                # Save and push model weights at specified intervals
                if self.eval_count % self.save_freq_evals == 0:
                    self._save_and_push_model(mean_reward)
        
        return result
    
    def _save_and_push_model(self, mean_reward):
        """Save model weights and push to wandb"""
        try:
            # Create timestamped model filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_eval_{self.eval_count}_step_{self.num_timesteps}_reward_{mean_reward:.2f}_{timestamp}.zip"
            model_path = os.path.join(self.model_save_path, model_filename)
            
            # Save the model
            self.model.save(model_path)
            print(f"[MODEL SAVE] Saved model to: {model_path}")
            
            # Push to wandb as artifact if enabled
            if self.push_to_wandb:
                self._push_to_wandb(model_path, model_filename, mean_reward)
                
        except Exception as e:
            print(f"[ERROR] Failed to save/push model: {e}")
    
    def _push_to_wandb(self, model_path, model_filename, mean_reward):
        """Push model to wandb as artifact"""
        try:
            # Create artifact with metadata
            artifact_name = f"model_eval_{self.eval_count}"
            artifact = wandb.Artifact(
                artifact_name, 
                type='model',
                metadata={
                    'eval_count': self.eval_count,
                    'timestep': self.num_timesteps,
                    'mean_reward': mean_reward,
                    'model_type': 'PPO',
                    'eval_timestamp': datetime.now().isoformat()
                }
            )
            
            # Add model file
            artifact.add_file(model_path, name=model_filename)
            
            # Log artifact to wandb
            wandb.log_artifact(artifact, aliases=[f"eval_{self.eval_count}", "latest_eval"])
            
            print(f"[WANDB] Pushed model artifact: {artifact_name}")
            
            # Log model push event
            wandb.log({
                'model_push/eval_count': self.eval_count,
                'model_push/timestep': self.num_timesteps,
                'model_push/mean_reward': mean_reward,
                'model_push/artifact_name': artifact_name,
            }, step=self.num_timesteps)
            
        except Exception as e:
            print(f"[ERROR] Failed to push model to wandb: {e}")


def create_target_sequence(num_drones=4, scale=0.5):
    """Create a challenging but achievable target sequence"""
    
    if num_drones == 4:
        # 4-drone formations
        targets = np.array([
            # Phase 0: Square formation
            [[ scale,  scale, 0.5], [-scale,  scale, 0.5], 
             [-scale, -scale, 0.5], [ scale, -scale, 0.5]],
            [[ scale,  scale, 0.5], [-scale,  scale, 0.5], 
             [-scale, -scale, 0.5], [ scale, -scale, 0.5]],
            [[ scale,  scale, 0.5], [-scale,  scale, 0.5], 
             [-scale, -scale, 0.5], [ scale, -scale, 0.5]],
            [[ scale,  scale, 0.5], [-scale,  scale, 0.5], 
             [-scale, -scale, 0.5], [ scale, -scale, 0.5]],
            
            # # Phase 1: Rotate clockwise
            # [[-scale,  scale, 0.5], [-scale, -scale, 0.5], 
            #  [ scale, -scale, 0.5], [ scale,  scale, 0.5]],
            
            # # Phase 2: Diamond formation (higher altitude)
            # [[ 0.0,  scale*1.2, 2.0], [-scale*1.2,  0.0, 2.0], 
            #  [ 0.0, -scale*1.2, 2.0], [ scale*1.2,  0.0, 2.0]],
            
            # # Phase 3: Tight formation at center
            # [[ 0.3,  0.3, 1.8], [-0.3,  0.3, 1.8], 
            #  [-0.3, -0.3, 1.8], [ 0.3, -0.3, 1.8]],
             
            # # Phase 4: Line formation
            # [[ 0.0,  scale, 0.5], [ 0.0,  scale/3, 0.5], 
            #  [ 0.0, -scale/3, 0.5], [ 0.0, -scale, 0.5]]
        ])
    else:
        # For other numbers of drones, create circular formations
        targets = []
        n_phases = 4
        for phase in range(n_phases):
            phase_targets = []
            radius = scale * (1.0 + 0.2 * phase)  # Varying radius
            height = 0.5 + 0.3 * phase  # Varying height
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones + phase * np.pi / 4  # Rotate each phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                phase_targets.append([x, y, height])
            targets.append(phase_targets)
        targets = np.array(targets)
    
    return targets.astype(np.float32)


def run(output_folder, gui, record_video, plot, local, wandb_project, wandb_entity, save_freq_evals):
    # Initialize Weights & Biases with comprehensive config
    run_name = f"multi_target_swarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = {
        'algo': 'PPO',
        'num_drones': DEFAULT_DRONES,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,
        'duration_sec': DEFAULT_DURATION_SEC,
        'total_timesteps': int(1e6), #int(2e6) if local else int(1e5),
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'n_steps': 2048,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'eval_freq': 100000,
        'log_freq': 1000,
        'save_freq_evals': save_freq_evals,  # New parameter
    }
    
    # Initialize wandb with more detailed settings
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
        sync_tensorboard=True,  # Also sync tensorboard logs
        monitor_gym=True,       # Monitor gym environments
        save_code=True,         # Save code for reproducibility
    )

    # Prepare output directory
    save_dir = os.path.join(output_folder, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Get control frequency
    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    # Create target sequence
    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=1.2)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] Created target sequence with shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Total episode length: {len(target_sequence) * steps_per_target} steps")
    print(f"[INFO] Model weights will be saved every {save_freq_evals} evaluation(s)")

    # Vectorized training environment
    def make_env():
        env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=False,
            record=False
        )
        return Monitor(env)
    
    train_env = make_vec_env(make_env, n_envs=NUM_VEC, seed=0)
    
    # Evaluation environment
    eval_env = MultiTargetAviary(
        num_drones=DEFAULT_DRONES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        target_sequence=target_sequence,
        steps_per_target=steps_per_target,
        gui=False,
        record=False
    )

    # Print environment info
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # Initialize PPO model with better hyperparameters
    model = PPO(
        'MlpPolicy', 
        train_env,
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
    
    # Log model architecture and watch gradients
    wandb.watch(model.policy, log='all', log_freq=1000, log_graph=True)

    # Enhanced callbacks with comprehensive logging and model pushing
    enhanced_wandb_cb = EnhancedWandbCallback(
        log_freq=config['log_freq'],
        verbose=1
    )
    
    detailed_eval_cb = DetailedEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=config['eval_freq'],
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
        save_freq_evals=save_freq_evals,  # New parameter
        push_to_wandb=True
    )
    
    # Standard WandB callback for additional SB3 metrics
    standard_wandb_cb = WandbCallback(
        model_save_path=save_dir,
        verbose=1,
        model_save_freq=10000,
        gradient_save_freq=1000,  # Save gradients periodically
    )
    
    cb_list = CallbackList([enhanced_wandb_cb, detailed_eval_cb, standard_wandb_cb])

    # Log initial metrics
    wandb.log({
        'setup/num_drones': DEFAULT_DRONES,
        'setup/episode_length': len(target_sequence) * steps_per_target,
        'setup/phases': len(target_sequence),
        'setup/steps_per_phase': steps_per_target,
        'setup/control_frequency': freq,
        'setup/save_freq_evals': save_freq_evals,
    })

    # Train the model
    print(f"[INFO] Starting training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=cb_list, 
        log_interval=10,  # More frequent console logging
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"[INFO] Training completed in {training_time:.2f} seconds")

    # Save final model
    final_path = os.path.join(save_dir, 'final_model.zip')
    model.save(final_path)
    
    # Upload final model as wandb artifact
    final_artifact = wandb.Artifact('final_model', type='model', metadata={
        'training_complete': True,
        'total_timesteps': config['total_timesteps'],
        'training_time_seconds': training_time,
        'final_model': True
    })
    final_artifact.add_file(final_path)
    wandb.log_artifact(final_artifact, aliases=["final", "best"])

    # Final evaluation with detailed logging
    print("[INFO] Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"[RESULTS] Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({
        'final_eval/mean_reward': mean_reward, 
        'final_eval/std_reward': std_reward,
        'final_eval/training_time_seconds': training_time,
        'final_eval/training_time_minutes': training_time / 60,
        'final_eval/timesteps_per_second': config['total_timesteps'] / training_time,
    })

    # Create summary table for wandb
    summary_table = wandb.Table(columns=["Metric", "Value"])
    summary_table.add_data("Final Mean Reward", f"{mean_reward:.2f}")
    summary_table.add_data("Final Std Reward", f"{std_reward:.2f}")
    summary_table.add_data("Training Time (min)", f"{training_time/60:.2f}")
    summary_table.add_data("Total Timesteps", f"{config['total_timesteps']:,}")  # Format as string with commas
    summary_table.add_data("Number of Drones", f"{DEFAULT_DRONES}")  # Convert to string
    summary_table.add_data("Model Save Frequency", f"Every {save_freq_evals} evaluation(s)")
    wandb.log({"training_summary": summary_table})

    # Demonstration
    if gui or record_video:
        print("[INFO] Running demonstration...")
        test_env = MultiTargetAviary(
            num_drones=DEFAULT_DRONES,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            target_sequence=target_sequence,
            steps_per_target=steps_per_target,
            gui=gui,
            record=record_video
        )
        
        # Optional: Create logger for visualization
        if plot:
            logger = Logger(
                logging_freq_hz=freq, 
                num_drones=DEFAULT_DRONES, 
                output_folder=save_dir
            )
        
        obs, info = test_env.reset(seed=42)
        start = time.time()
        episode_reward = 0
        demo_metrics = {
            'phase_rewards': [],
            'targets_reached_per_phase': [],
            'distances': []
        }
        
        max_steps = len(target_sequence) * steps_per_target
        print(f"[INFO] Running demonstration for {max_steps} steps...")
        
        for i in range(max_steps + 100):  # Add buffer steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            
            # Collect demo metrics
            if 'phase' in info:
                current_phase = info['phase']
                if len(demo_metrics['phase_rewards']) <= current_phase:
                    demo_metrics['phase_rewards'].extend([0] * (current_phase - len(demo_metrics['phase_rewards']) + 1))
                demo_metrics['phase_rewards'][current_phase] += reward
            
            if 'distance_to_targets' in info:
                demo_metrics['distances'].append(np.mean(info['distance_to_targets']))
            
            if gui:
                test_env.render()
                sync(i, start, test_env.CTRL_TIMESTEP)
            
            # Log progress
            if i % 100 == 0:
                phase = info.get('phase', -1)
                targets_reached = np.sum(info.get('targets_reached', []))
                print(f"Step {i}, Phase {phase}, Targets reached: {targets_reached}, Reward: {reward:.2f}")
            
            # Optional logging for plotting
            if plot and DEFAULT_OBS == ObservationType.KIN:
                obs_arr = obs.squeeze() if obs.ndim > 1 else obs
                act_arr = action.squeeze() if action.ndim > 1 else action
                
                for d in range(DEFAULT_DRONES):
                    state = np.hstack([
                        obs_arr[d][0:3] if obs_arr.ndim > 1 else obs_arr[0:3],
                        np.zeros(4),  # quaternion placeholder
                        obs_arr[d][3:15] if obs_arr.ndim > 1 else obs_arr[3:15],
                        act_arr[d] if act_arr.ndim > 0 else act_arr
                    ])
                    logger.log(drone=d, timestamp=i/freq, state=state, control=np.zeros(12))
            
            if done:
                print(f"[INFO] Episode finished at step {i}")
                break
        
        print(f"[INFO] Total episode reward: {episode_reward:.2f}")
        
        # Log demonstration metrics
        wandb.log({
            'demo/total_reward': episode_reward,
            'demo/mean_distance': np.mean(demo_metrics['distances']) if demo_metrics['distances'] else 0,
            'demo/final_distance': demo_metrics['distances'][-1] if demo_metrics['distances'] else 0,
        })
        
        # Create demonstration plots
        if demo_metrics['distances']:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Distance plot
            ax1.plot(demo_metrics['distances'])
            ax1.set_title('Distance to Targets During Demonstration')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Mean Distance')
            ax1.grid(True)
            
            # Phase rewards plot
            if demo_metrics['phase_rewards']:
                ax2.bar(range(len(demo_metrics['phase_rewards'])), demo_metrics['phase_rewards'])
                ax2.set_title('Rewards per Phase')
                ax2.set_xlabel('Phase')
                ax2.set_ylabel('Total Reward')
                ax2.grid(True)
            
            plt.tight_layout()
            wandb.log({"demo_plots": wandb.Image(fig)})
            plt.close()
        
        test_env.close()
        
        if plot and DEFAULT_OBS == ObservationType.KIN:
            logger.plot()

    wandb.finish()
    print(f"[INFO] Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-target drone swarm training with PPO')
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
    parser.add_argument('--wandb_project', default='drone-swarm-multitarget', type=str, 
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str, 
                        help='Weights & Biases entity/username')
    parser.add_argument('--save_freq_evals', default=1, type=int,
                        help='Save and push model weights every N evaluations (default: 1)')
    
    args = parser.parse_args()

    run(
        args.output_folder,
        args.gui,
        args.record_video,
        args.plot,
        args.local,
        args.wandb_project,
        args.wandb_entity,
        args.save_freq_evals
    )