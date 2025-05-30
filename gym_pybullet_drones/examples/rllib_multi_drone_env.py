#!/usr/bin/env python3
"""
rllib_multi_drone_env.py - RLlib Multi-Agent Environment Wrapper
Converts MultiTargetAviary to RLlib MultiAgentEnv format for much better performance
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv

# Import your existing environment
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class RLlibMultiDroneEnv(MultiAgentEnv):
    """
    RLlib wrapper for MultiTargetAviary environment.
    
    Key advantages over SB3 approach:
    - Native multi-agent support
    - Parameter sharing between agents
    - Distributed data collection
    - Much better performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration
        self.num_drones = config.get('num_drones', 4)
        self.obs_type = config.get('obs_type', ObservationType.KIN)
        self.act_type = config.get('act_type', ActionType.RPM)
        self.gui = config.get('gui', False)
        self.record = config.get('record', False)
        
        # Target sequence configuration
        target_sequence = config.get('target_sequence', None)
        if target_sequence is None:
            target_sequence = self._create_default_targets(self.num_drones, config.get('target_scale', 1.2))
        
        self.steps_per_target = config.get('steps_per_target', 100)
        self.tolerance = config.get('tolerance', 0.15)
        self.collision_distance = config.get('collision_distance', 0.05)
        
        # Paper-based reward hyperparameters
        reward_params = config.get('reward_params', {})
        self.lambda_1 = reward_params.get('lambda_1', 20.0)    # Progress reward
        self.lambda_2 = reward_params.get('lambda_2', 1.5)     # Perception reward  
        self.lambda_3 = reward_params.get('lambda_3', 0.4)     # Perception alignment
        self.lambda_4 = reward_params.get('lambda_4', 0.01)    # Action magnitude penalty
        self.lambda_5 = reward_params.get('lambda_5', 0.1)     # Action smoothness penalty
        self.crash_penalty = reward_params.get('crash_penalty', 15.0)
        self.bounds_penalty = reward_params.get('bounds_penalty', 8.0)
        
        # Create the underlying environment
        self.env = MultiTargetAviary(
            drone_model=config.get('drone_model', DroneModel.CF2X),
            num_drones=self.num_drones,
            neighbourhood_radius=config.get('neighbourhood_radius', np.inf),
            initial_xyzs=config.get('initial_xyzs', None),
            initial_rpys=config.get('initial_rpys', None),
            physics=config.get('physics', Physics.PYB),
            pyb_freq=config.get('pyb_freq', 240),
            ctrl_freq=config.get('ctrl_freq', 30),
            gui=self.gui,
            record=self.record,
            obs=self.obs_type,
            act=self.act_type,
            target_sequence=target_sequence,
            steps_per_target=self.steps_per_target,
            tolerance=self.tolerance,
            collision_distance=self.collision_distance,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            lambda_3=self.lambda_3,
            lambda_4=self.lambda_4,
            lambda_5=self.lambda_5,
            crash_penalty=self.crash_penalty,
            bounds_penalty=self.bounds_penalty,
        )
        
        # Define agent IDs
        self._agent_ids = set([f"drone_{i}" for i in range(self.num_drones)])
        
        # Extract individual agent observation and action spaces
        original_obs_space = self.env.observation_space
        original_action_space = self.env.action_space
        
        # Handle multi-drone observation space
        if len(original_obs_space.shape) == 2:  # (num_drones, features)
            self.observation_space = spaces.Box(
                low=original_obs_space.low[0],
                high=original_obs_space.high[0],
                dtype=original_obs_space.dtype
            )
        else:
            self.observation_space = original_obs_space
            
        # Handle multi-drone action space
        if len(original_action_space.shape) == 2:  # (num_drones, actions)
            self.action_space = spaces.Box(
                low=original_action_space.low[0],
                high=original_action_space.high[0],
                dtype=original_action_space.dtype
            )
        else:
            self.action_space = original_action_space
        
        # For RLlib compatibility
        self._spaces_in_preferred_format = True
        
        # Metrics tracking
        self.episode_stats = {
            'total_reward': 0.0,
            'episode_length': 0,
            'targets_reached': 0,
            'collisions': 0,
            'phase_completions': 0,
        }
        
        print(f"[RLlibMultiDroneEnv] Initialized with {self.num_drones} drones")
        print(f"[RLlibMultiDroneEnv] Observation space per agent: {self.observation_space.shape}")
        print(f"[RLlibMultiDroneEnv] Action space per agent: {self.action_space.shape}")
        print(f"[RLlibMultiDroneEnv] Action type: {self.act_type}")
        print(f"[RLlibMultiDroneEnv] Paper-based reward function enabled")
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment and return per-agent observations."""
        
        # Reset episode stats
        self.episode_stats = {
            'total_reward': 0.0,
            'episode_length': 0,
            'targets_reached': 0,
            'collisions': 0,
            'phase_completions': 0,
        }
        
        # Reset underlying environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Convert to per-agent format
        obs_dict = self._convert_obs_to_dict(obs)
        info_dict = self._convert_info_to_dict(info)
        
        return obs_dict, info_dict
    
    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Step environment with per-agent actions."""
        
        # Convert action dict to array format
        actions = self._convert_actions_to_array(action_dict)
        
        # Step underlying environment
        obs, reward, done, truncated, info = self.env.step(actions)
        
        # Update episode stats
        self.episode_stats['episode_length'] += 1
        self.episode_stats['total_reward'] += reward
        if 'targets_reached' in info:
            self.episode_stats['targets_reached'] = np.sum(info['targets_reached'])
        if 'collision_count' in info:
            self.episode_stats['collisions'] = info['collision_count']
        if 'phase' in info:
            self.episode_stats['phase_completions'] = info['phase']
        
        # Convert to per-agent format
        obs_dict = self._convert_obs_to_dict(obs)
        reward_dict = self._convert_reward_to_dict(reward)
        done_dict = self._convert_done_to_dict(done)
        truncated_dict = self._convert_truncated_to_dict(truncated)
        info_dict = self._convert_info_to_dict(info)
        
        # Add episode stats to info for logging
        info_dict['__common__'] = {
            **self.episode_stats,
            'current_phase': info.get('phase', 0),
            'mean_distance_to_targets': info.get('mean_distance_to_targets', 0.0),
            'formation_error': info.get('formation_error', 0.0),
        }
        
        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict
    
    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert multi-agent observation to dictionary format."""
        if obs.ndim == 2:  # (num_drones, features)
            return {f"drone_{i}": obs[i] for i in range(self.num_drones)}
        else:  # Single observation for all agents
            return {f"drone_{i}": obs for i in range(self.num_drones)}
    
    def _convert_actions_to_array(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert per-agent actions to array format."""
        if isinstance(action_dict, dict):
            # Ensure correct order
            actions = np.array([action_dict[f"drone_{i}"] for i in range(self.num_drones)])
        else:
            actions = action_dict
        return actions
    
    def _convert_reward_to_dict(self, reward: float) -> Dict[str, float]:
        """Convert single reward to per-agent rewards."""
        # Share reward among all agents (cooperative setting)
        return {f"drone_{i}": reward / self.num_drones for i in range(self.num_drones)}
    
    def _convert_done_to_dict(self, done: bool) -> Dict[str, bool]:
        """Convert single done signal to per-agent format."""
        done_dict = {f"drone_{i}": done for i in range(self.num_drones)}
        done_dict["__all__"] = done  # Global termination signal
        return done_dict
    
    def _convert_truncated_to_dict(self, truncated: bool) -> Dict[str, bool]:
        """Convert single truncated signal to per-agent format."""
        truncated_dict = {f"drone_{i}": truncated for i in range(self.num_drones)}
        truncated_dict["__all__"] = truncated  # Global truncation signal
        return truncated_dict
    
    def _convert_info_to_dict(self, info: Dict) -> Dict[str, Dict]:
        """Convert single info dict to per-agent format."""
        # Share info among all agents
        agent_info = {
            'phase': info.get('phase', 0),
            'targets_reached': info.get('targets_reached', np.zeros(self.num_drones, dtype=bool)),
            'distance_to_targets': info.get('distance_to_targets', np.zeros(self.num_drones)),
            'collision_count': info.get('collision_count', 0),
            'mean_distance': info.get('mean_distance_to_targets', 0.0),
            'formation_error': info.get('formation_error', 0.0),
        }
        
        info_dict = {f"drone_{i}": agent_info.copy() for i in range(self.num_drones)}
        return info_dict
    
    def _create_default_targets(self, num_drones: int, scale: float = 1.2) -> np.ndarray:
        """Create default target sequence matching training requirements."""
        if num_drones == 4:
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
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @property
    def get_agent_ids(self):
        """Return the set of agent IDs."""
        return self._agent_ids


# Factory function for easy environment creation
def make_rllib_multi_drone_env(config: Dict[str, Any]) -> RLlibMultiDroneEnv:
    """Factory function to create RLlib multi-drone environment."""
    return RLlibMultiDroneEnv(config)


# Test the environment
if __name__ == "__main__":
    print("Testing RLlib Multi-Drone Environment...")
    
    # Test configuration
    test_config = {
        'num_drones': 4,
        'obs_type': ObservationType.KIN,
        'act_type': ActionType.RPM,
        'gui': False,
        'steps_per_target': 50,
        'reward_params': {
            'lambda_1': 20.0,
            'lambda_2': 1.5,
            'lambda_3': 0.4,
            'lambda_4': 0.01,
            'lambda_5': 0.1,
            'crash_penalty': 15.0,
            'bounds_penalty': 8.0,
        }
    }
    
    # Create environment
    env = RLlibMultiDroneEnv(test_config)
    
    print(f"Agent IDs: {env.get_agent_ids}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test episode
    obs_dict, info_dict = env.reset()
    print(f"Reset observation keys: {list(obs_dict.keys())}")
    print(f"Observation shape per agent: {obs_dict['drone_0'].shape}")
    
    # Test step
    action_dict = {agent_id: env.action_space.sample() for agent_id in env.get_agent_ids}
    obs_dict, reward_dict, done_dict, truncated_dict, info_dict = env.step(action_dict)
    
    print(f"Step rewards: {reward_dict}")
    print(f"Step done: {done_dict['__all__']}")
    print(f"Episode stats: {env.episode_stats}")
    
    # Test multiple steps
    for i in range(10):
        action_dict = {agent_id: env.action_space.sample() for agent_id in env.get_agent_ids}
        obs_dict, reward_dict, done_dict, truncated_dict, info_dict = env.step(action_dict)
        
        if done_dict["__all__"] or truncated_dict["__all__"]:
            print(f"Episode finished at step {i+1}")
            break
    
    env.close()
    print("\n✅ RLlib Multi-Drone Environment test completed successfully!")
    print("Environment is ready for high-performance multi-agent training with RLlib.")