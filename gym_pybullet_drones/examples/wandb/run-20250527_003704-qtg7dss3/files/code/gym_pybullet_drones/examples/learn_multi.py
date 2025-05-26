#!/usr/bin/env python3
"""
Modified train_multi_improved.py with custom multi-agent architecture
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

# Import your custom extractors (put the previous artifact code in a file called multi_agent_extractors.py)
from multi_agent_extractors import (
    MultiAgentSelfAttentionExtractor,
    MultiAgentMeanPoolExtractor, 
    MultiAgentDeepsetsExtractor,
    create_multiagent_ppo_model
)

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Default settings - same as before
DEFAULT_GUI           = True
DEFAULT_RECORD_VIDEO  = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS           = ObservationType('kin')
DEFAULT_ACT           = ActionType('rpm')
DEFAULT_DRONES        = 4
DEFAULT_DURATION_SEC  = 3.0
NUM_VEC = 1

# NEW: Multi-agent architecture settings
DEFAULT_EXTRACTOR_TYPE = "attention"  # "attention", "meanpool", or "deepsets"
DEFAULT_FEATURES_DIM = 256


class EnhancedWandbCallback(BaseCallback):
    """Enhanced callback - same as before"""
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Same implementation as before...
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
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
        
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                
                log_dict = {
                    'train/timestep': self.num_timesteps,
                    'train/current_reward': self.locals.get('rewards', [0])[0],
                }
                
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
            
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                logger_dict = {}
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        logger_dict[f'train/{key}'] = value
                
                if logger_dict:
                    wandb.log(logger_dict, step=self.num_timesteps)
        
        return True


class DetailedEvalCallback(EvalCallback):
    """Enhanced evaluation callback - same as before"""
    def __init__(self, eval_env, save_freq_evals=1, push_to_wandb=True, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.save_freq_evals = save_freq_evals
        self.push_to_wandb = push_to_wandb
        self.eval_count = 0
        self.model_save_path = kwargs.get('best_model_save_path', './models')
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            if len(self.evaluations_results) > 0:
                latest_results = self.evaluations_results[-1]
                mean_reward = np.mean(latest_results)
                std_reward = np.std(latest_results)
                min_reward = np.min(latest_results)
                max_reward = np.max(latest_results)
                
                wandb.log({
                    'eval/mean_reward': mean_reward,
                    'eval/std_reward': std_reward,
                    'eval/min_reward': min_reward,
                    'eval/max_reward': max_reward,
                    'eval/num_episodes': len(latest_results),
                    'eval/evaluation_count': self.eval_count,
                }, step=self.num_timesteps)
                
                print(f"[EVAL] Step {self.num_timesteps}: Mean={mean_reward:.2f}±{std_reward:.2f}, Range=[{min_reward:.2f}, {max_reward:.2f}]")
                
                if self.eval_count % self.save_freq_evals == 0:
                    self._save_and_push_model(mean_reward)
        
        return result
    
    def _save_and_push_model(self, mean_reward):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_eval_{self.eval_count}_step_{self.num_timesteps}_reward_{mean_reward:.2f}_{timestamp}.zip"
            model_path = os.path.join(self.model_save_path, model_filename)
            
            self.model.save(model_path)
            print(f"[MODEL SAVE] Saved model to: {model_path}")
            
            if self.push_to_wandb:
                self._push_to_wandb(model_path, model_filename, mean_reward)
                
        except Exception as e:
            print(f"[ERROR] Failed to save/push model: {e}")
    
    def _push_to_wandb(self, model_path, model_filename, mean_reward):
        try:
            artifact_name = f"model_eval_{self.eval_count}"
            artifact = wandb.Artifact(
                artifact_name, 
                type='model',
                metadata={
                    'eval_count': self.eval_count,
                    'timestep': self.num_timesteps,
                    'mean_reward': mean_reward,
                    'model_type': 'PPO_MultiAgent',
                    'eval_timestamp': datetime.now().isoformat()
                }
            )
            
            artifact.add_file(model_path, name=model_filename)
            wandb.log_artifact(artifact, aliases=[f"eval_{self.eval_count}", "latest_eval"])
            
            print(f"[WANDB] Pushed model artifact: {artifact_name}")
            
            wandb.log({
                'model_push/eval_count': self.eval_count,
                'model_push/timestep': self.num_timesteps,
                'model_push/mean_reward': mean_reward,
                'model_push/artifact_name': artifact_name,
            }, step=self.num_timesteps)
            
        except Exception as e:
            print(f"[ERROR] Failed to push model to wandb: {e}")


def create_target_sequence(num_drones=4, scale=1):
    """Create target sequence - same as before"""
    if num_drones == 4:
        targets = np.array([
            [[ scale,  scale, 1], [-scale,  scale, 1], 
             [-scale, -scale, 1], [ scale, -scale, 1]],
            [[ scale,  scale, 1], [-scale,  scale, 1], 
             [-scale, -scale, 1], [ scale, -scale, 1]],
            [[ scale,  scale, 1], [-scale,  scale, 1], 
             [-scale, -scale, 1], [ scale, -scale, 1]],
            [[ scale,  scale, 1], [-scale,  scale, 1], 
             [-scale, -scale, 1], [ scale, -scale, 1]],
        ])
    else:
        targets = []
        n_phases = 4
        for phase in range(n_phases):
            phase_targets = []
            radius = scale * (1.0 + 0.2 * phase)
            height = 1 + 0.3 * phase
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones + phase * np.pi / 4
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                phase_targets.append([x, y, height])
            targets.append(phase_targets)
        targets = np.array(targets)
    
    return targets.astype(np.float32)


def run(output_folder, gui, record_video, plot, local, wandb_project, wandb_entity, 
        save_freq_evals, extractor_type, features_dim):
    """
    Modified run function with multi-agent architecture support
    """
    run_name = f"multiagent_{extractor_type}_swarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = {
        'algo': 'PPO',
        'architecture': 'MultiAgent',
        'extractor_type': extractor_type,
        'features_dim': features_dim,
        'num_drones': DEFAULT_DRONES,
        'obs_type': DEFAULT_OBS.name,
        'act_type': DEFAULT_ACT.name,
        'duration_sec': DEFAULT_DURATION_SEC,
        'total_timesteps': int(10e6),
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
        'save_freq_evals': save_freq_evals,
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

    dummy_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    freq = int(dummy_env.CTRL_FREQ)
    dummy_env.close()

    target_sequence = create_target_sequence(DEFAULT_DRONES, scale=1.0)
    steps_per_target = int(DEFAULT_DURATION_SEC * freq)
    
    print(f"[INFO] Created target sequence with shape: {target_sequence.shape}")
    print(f"[INFO] Steps per target: {steps_per_target}")
    print(f"[INFO] Using {extractor_type} architecture with {features_dim} features")

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
    
    eval_env = MultiTargetAviary(
        num_drones=DEFAULT_DRONES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        target_sequence=target_sequence,
        steps_per_target=steps_per_target,
        gui=False,
        record=False
    )

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # CREATE MODEL WITH CUSTOM MULTI-AGENT ARCHITECTURE
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
    
    # Log the model architecture
    print(f"\n[INFO] Model architecture:")
    print(f"Policy: {model.policy}")
    print(f"Feature extractor: {model.policy.features_extractor}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    extractor_params = sum(p.numel() for p in model.policy.features_extractor.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Feature extractor parameters: {extractor_params:,}")
    
    wandb.watch(model.policy, log='all', log_freq=1000, log_graph=True)

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
        save_freq_evals=save_freq_evals,
        push_to_wandb=True
    )
    
    standard_wandb_cb = WandbCallback(
        model_save_path=save_dir,
        verbose=1,
        model_save_freq=50000,
        gradient_save_freq=50000,
    )
    
    cb_list = CallbackList([enhanced_wandb_cb, detailed_eval_cb, standard_wandb_cb])

    wandb.log({
        'setup/num_drones': DEFAULT_DRONES,
        'setup/episode_length': len(target_sequence) * steps_per_target,
        'setup/phases': len(target_sequence),
        'setup/steps_per_phase': steps_per_target,
        'setup/control_frequency': freq,
        'setup/save_freq_evals': save_freq_evals,
        'setup/extractor_type': extractor_type,
        'setup/features_dim': features_dim,
        'setup/total_parameters': total_params,
        'setup/extractor_parameters': extractor_params,
    })

    print(f"[INFO] Starting training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=cb_list, 
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"[INFO] Training completed in {training_time:.2f} seconds")

    final_path = os.path.join(save_dir, 'final_model.zip')
    model.save(final_path)
    
    final_artifact = wandb.Artifact('final_model', type='model', metadata={
        'training_complete': True,
        'total_timesteps': config['total_timesteps'],
        'training_time_seconds': training_time,
        'final_model': True,
        'extractor_type': extractor_type,
        'features_dim': features_dim,
    })
    final_artifact.add_file(final_path)
    wandb.log_artifact(final_artifact, aliases=["final", "best"])

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

    summary_table = wandb.Table(columns=["Metric", "Value"])
    summary_table.add_data("Final Mean Reward", f"{mean_reward:.2f}")
    summary_table.add_data("Final Std Reward", f"{std_reward:.2f}")
    summary_table.add_data("Training Time (min)", f"{training_time/60:.2f}")
    summary_table.add_data("Total Timesteps", f"{config['total_timesteps']:,}")
    summary_table.add_data("Number of Drones", f"{DEFAULT_DRONES}")
    summary_table.add_data("Architecture Type", extractor_type)
    summary_table.add_data("Features Dimension", f"{features_dim}")
    summary_table.add_data("Total Parameters", f"{total_params:,}")
    wandb.log({"training_summary": summary_table})

    # Demonstration (same as before but with custom model)
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
        
        for i in range(max_steps + 100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            
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
            
            if i % 100 == 0:
                phase = info.get('phase', -1)
                targets_reached = np.sum(info.get('targets_reached', []))
                print(f"Step {i}, Phase {phase}, Targets reached: {targets_reached}, Reward: {reward:.2f}")
            
            if plot and DEFAULT_OBS == ObservationType.KIN:
                obs_arr = obs.squeeze() if obs.ndim > 1 else obs
                act_arr = action.squeeze() if action.ndim > 1 else action
                
                for d in range(DEFAULT_DRONES):
                    state = np.hstack([
                        obs_arr[d][0:3] if obs_arr.ndim > 1 else obs_arr[0:3],
                        np.zeros(4),
                        obs_arr[d][3:15] if obs_arr.ndim > 1 else obs_arr[3:15],
                        act_arr[d] if act_arr.ndim > 0 else act_arr
                    ])
                    logger.log(drone=d, timestamp=i/freq, state=state, control=np.zeros(12))
            
            if done:
                print(f"[INFO] Episode finished at step {i}")
                break
        
        print(f"[INFO] Total episode reward: {episode_reward:.2f}")
        
        wandb.log({
            'demo/total_reward': episode_reward,
            'demo/mean_distance': np.mean(demo_metrics['distances']) if demo_metrics['distances'] else 0,
            'demo/final_distance': demo_metrics['distances'][-1] if demo_metrics['distances'] else 0,
        })
        
        if demo_metrics['distances']:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            ax1.plot(demo_metrics['distances'])
            ax1.set_title('Distance to Targets During Demonstration')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Mean Distance')
            ax1.grid(True)
            
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
    parser = argparse.ArgumentParser(description='Multi-agent drone swarm training with PPO')
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
    parser.add_argument('--wandb_project', default='drone-swarm-multiagent', type=str, 
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str, 
                        help='Weights & Biases entity/username')
    parser.add_argument('--save_freq_evals', default=1, type=int,
                        help='Save and push model weights every N evaluations')
    
    # NEW: Multi-agent architecture arguments
    parser.add_argument('--extractor_type', default=DEFAULT_EXTRACTOR_TYPE, 
                        choices=['attention', 'meanpool', 'deepsets'], type=str,
                        help='Type of multi-agent feature extractor')
    parser.add_argument('--features_dim', default=DEFAULT_FEATURES_DIM, type=int,
                        help='Dimension of feature representation')
    
    args = parser.parse_args()

    run(
        args.output_folder,
        args.gui,
        args.record_video,
        args.plot,
        args.local,
        args.wandb_project,
        args.wandb_entity,
        args.save_freq_evals,
        args.extractor_type,    # NEW
        args.features_dim       # NEW
    )