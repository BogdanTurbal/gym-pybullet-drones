# enhanced_multiagent_extractors.py
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.td3.policies import TD3Policy
import gymnasium as gym
from typing import Dict, Type, Union, Optional, List, Tuple


class IndividualDroneExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for individual drone with awareness of other drones.
    Each drone processes its own observation plus information about other drones.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, max_other_drones: int = 10):
        """
        Args:
            observation_space: Expected to be (own_obs_dim + other_drones_info_dim,)
            features_dim: Output feature dimension
            max_other_drones: Maximum number of other drones to consider
        """
        super().__init__(observation_space, features_dim)
        
        # Parse observation space - expect flattened format:
        # [own_state(12), own_target(6), other_drones_info(N * info_per_drone)]
        total_obs_dim = observation_space.shape[-1]
        
        # Own state dimensions (from BaseRLAviary)
        self.own_state_dim = 12  # pos(3) + rpy(3) + vel(3) + ang_vel(3)
        self.own_target_dim = 6   # target_pos(3) + relative_target(3)
        self.own_total_dim = self.own_state_dim + self.own_target_dim
        
        # Other drones info dimension
        self.other_drones_info_dim = total_obs_dim - self.own_total_dim
        self.info_per_other_drone = 7  # pos(3) + vel(3) + distance(1)
        self.max_other_drones = max_other_drones
        
        print(f"[IndividualDroneExtractor] Own state: {self.own_state_dim}")
        print(f"[IndividualDroneExtractor] Own target: {self.own_target_dim}")
        print(f"[IndividualDroneExtractor] Other drones info: {self.other_drones_info_dim}")
        print(f"[IndividualDroneExtractor] Info per other drone: {self.info_per_other_drone}")
        
        # Own state encoder
        self.own_state_encoder = nn.Sequential(
            nn.Linear(self.own_state_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
        )
        
        # Own target encoder
        self.own_target_encoder = nn.Sequential(
            nn.Linear(self.own_target_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        
        # Other drones encoder (processes each other drone)
        self.other_drone_encoder = nn.Sequential(
            nn.Linear(self.info_per_other_drone, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        
        # Attention mechanism for other drones
        self.attention_dim = 32
        self.attention_query = nn.Linear(64, self.attention_dim)  # From own state
        self.attention_key = nn.Linear(32, self.attention_dim)    # From other drones
        self.attention_value = nn.Linear(32, self.attention_dim)  # From other drones
        
        # Final aggregation
        combined_dim = 64 + 32 + self.attention_dim  # own_state + own_target + attended_others
        self.final_processor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, features_dim),
            nn.LeakyReLU(0.2),
        )
        
        self._features_dim = features_dim
        
        print(f"[IndividualDroneExtractor] Output features: {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split observation into components
        own_state = observations[:, :self.own_state_dim]
        own_target = observations[:, self.own_state_dim:self.own_state_dim + self.own_target_dim]
        other_drones_info = observations[:, self.own_state_dim + self.own_target_dim:]
        
        # Encode own state and target
        own_state_features = self.own_state_encoder(own_state)
        own_target_features = self.own_target_encoder(own_target)
        
        # Process other drones info
        if self.other_drones_info_dim > 0 and other_drones_info.shape[1] > 0:
            # Reshape other drones info to (batch_size, num_other_drones, info_per_drone)
            num_other_drones = self.other_drones_info_dim // self.info_per_other_drone
            
            if num_other_drones > 0:
                other_drones_reshaped = other_drones_info.view(batch_size, num_other_drones, self.info_per_other_drone)
                
                # Encode each other drone
                other_drones_flat = other_drones_reshaped.view(-1, self.info_per_other_drone)
                other_features_flat = self.other_drone_encoder(other_drones_flat)
                other_features = other_features_flat.view(batch_size, num_other_drones, -1)
                
                # Attention mechanism
                query = self.attention_query(own_state_features).unsqueeze(1)  # (batch, 1, attention_dim)
                keys = self.attention_key(other_features)  # (batch, num_other, attention_dim)
                values = self.attention_value(other_features)  # (batch, num_other, attention_dim)
                
                # Compute attention weights
                attention_scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, num_other)
                attention_weights = torch.softmax(attention_scores, dim=-1)
                
                # Apply attention
                attended_others = torch.bmm(attention_weights, values).squeeze(1)  # (batch, attention_dim)
            else:
                attended_others = torch.zeros(batch_size, self.attention_dim, device=observations.device)
        else:
            attended_others = torch.zeros(batch_size, self.attention_dim, device=observations.device)
        
        # Combine all features
        combined_features = torch.cat([own_state_features, own_target_features, attended_others], dim=1)
        final_features = self.final_processor(combined_features)
        
        return final_features


class MultiAgentWrapper:
    """
    Wrapper that converts multi-agent environment to work with parameter sharing.
    Each drone uses the same policy but gets individual observations and actions.
    """
    
    def __init__(self, base_env):
        self.base_env = base_env
        self.num_drones = base_env.NUM_DRONES
        
        # Create individual observation space (for one drone)
        self.individual_obs_space = self._create_individual_obs_space()
        
        # Create individual action space (for one drone)  
        self.individual_action_space = self._create_individual_action_space()
        
        print(f"[MultiAgentWrapper] Wrapped {self.num_drones} drones")
        print(f"[MultiAgentWrapper] Individual obs space: {self.individual_obs_space.shape}")
        print(f"[MultiAgentWrapper] Individual action space: {self.individual_action_space.shape}")
    
    def _create_individual_obs_space(self):
        """Create observation space for individual drone"""
        # Get sample observation to determine dimensions
        obs, _ = self.base_env.reset()
        
        if len(obs.shape) == 2:
            # Multi-agent format: (num_drones, obs_dim)
            single_drone_obs_dim = obs.shape[1]
        else:
            # Single agent format
            single_drone_obs_dim = obs.shape[0]
        
        # For individual drone, we need:
        # - Own observation (base features)
        # - Information about other drones
        
        # Parse the original observation to understand structure
        # From MultiTargetAviary: base_obs + target_obs (6 features)
        base_obs_dim = single_drone_obs_dim - 6  # Remove target features
        own_obs_dim = base_obs_dim + 6  # own base + own target
        
        # Other drones info: position(3) + velocity(3) + distance(1) = 7 per drone
        info_per_other_drone = 7
        other_drones_info_dim = (self.num_drones - 1) * info_per_other_drone
        
        total_individual_obs_dim = own_obs_dim + other_drones_info_dim
        
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_individual_obs_dim,), 
            dtype=np.float32
        )
    
    def _create_individual_action_space(self):
        """Create action space for individual drone"""
        # Action space is per drone, same as original but for one drone
        original_action_space = self.base_env.action_space
        
        if len(original_action_space.shape) == 2:
            # Multi-agent format: (num_drones, action_dim)
            action_dim = original_action_space.shape[1]
        else:
            # Single agent format
            action_dim = original_action_space.shape[0]
        
        return gym.spaces.Box(
            low=original_action_space.low[0] if len(original_action_space.low.shape) > 1 else original_action_space.low,
            high=original_action_space.high[0] if len(original_action_space.high.shape) > 1 else original_action_space.high,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def _convert_obs_to_individual(self, multi_obs):
        """Convert multi-agent observation to individual drone observations"""
        if len(multi_obs.shape) == 1:
            multi_obs = multi_obs.reshape(1, -1)
        
        batch_size, total_obs_dim = multi_obs.shape
        obs_per_drone = total_obs_dim // self.num_drones
        
        individual_observations = []
        
        for drone_idx in range(self.num_drones):
            # Get this drone's observation
            drone_obs = multi_obs[:, drone_idx * obs_per_drone:(drone_idx + 1) * obs_per_drone]
            
            # Split into base observation and target info
            base_obs_dim = obs_per_drone - 6  # Assuming 6 target features
            drone_base_obs = drone_obs[:, :base_obs_dim]
            drone_target_obs = drone_obs[:, base_obs_dim:]
            
            # Get information about other drones
            other_drones_info = []
            
            for other_idx in range(self.num_drones):
                if other_idx != drone_idx:
                    other_obs = multi_obs[:, other_idx * obs_per_drone:(other_idx + 1) * obs_per_drone]
                    other_base = other_obs[:, :base_obs_dim]
                    
                    # Extract position and velocity (first 6 features typically)
                    other_pos = other_base[:, 0:3]  # position
                    other_vel = other_base[:, 9:12] if other_base.shape[1] > 11 else other_base[:, 6:9]  # velocity
                    
                    # Calculate distance to this drone
                    drone_pos = drone_base_obs[:, 0:3]
                    distance = torch.norm(other_pos - drone_pos, dim=1, keepdim=True)
                    
                    # Combine: pos(3) + vel(3) + distance(1) = 7 features
                    other_info = torch.cat([other_pos, other_vel, distance], dim=1)
                    other_drones_info.append(other_info)
            
            # Sort other drones by distance (closest first)
            if other_drones_info:
                other_drones_tensor = torch.stack(other_drones_info, dim=1)  # (batch, num_others, 7)
                distances = other_drones_tensor[:, :, -1]  # Last feature is distance
                sorted_indices = torch.argsort(distances, dim=1)
                
                # Reorder by distance
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, sorted_indices.shape[1])
                sorted_others = other_drones_tensor[batch_indices, sorted_indices]
                
                # Flatten sorted other drones info
                other_drones_flat = sorted_others.view(batch_size, -1)
            else:
                other_drones_flat = torch.zeros(batch_size, 0)
            
            # Combine own observation with other drones info
            individual_obs = torch.cat([
                drone_base_obs,  # Own state
                drone_target_obs,  # Own target
                other_drones_flat  # Other drones info (sorted by distance)
            ], dim=1)
            
            individual_observations.append(individual_obs)
        
        return individual_observations
    
    def reset(self, **kwargs):
        """Reset environment and return individual observations"""
        multi_obs, info = self.base_env.reset(**kwargs)
        
        # Convert to torch tensor if needed
        if not isinstance(multi_obs, torch.Tensor):
            multi_obs = torch.tensor(multi_obs, dtype=torch.float32)
        
        individual_obs = self._convert_obs_to_individual(multi_obs)
        
        # Return observations for all drones (for training)
        return individual_obs, info
    
    def step(self, actions):
        """Execute actions and return individual observations"""
        # Convert individual actions back to multi-agent format
        if isinstance(actions, list):
            multi_actions = np.array(actions)
        else:
            multi_actions = actions
        
        # Execute in base environment
        multi_obs, reward, done, truncated, info = self.base_env.step(multi_actions)
        
        # Convert observations to individual format
        if not isinstance(multi_obs, torch.Tensor):
            multi_obs = torch.tensor(multi_obs, dtype=torch.float32)
        
        individual_obs = self._convert_obs_to_individual(multi_obs)
        
        return individual_obs, reward, done, truncated, info
    
    def __getattr__(self, name):
        """Delegate other attributes to base environment"""
        return getattr(self.base_env, name)


def create_individual_drone_model(
    env, 
    algorithm: str = "td3",
    features_dim: int = 256, 
    **kwargs
):
    """
    Create a model for individual drone control with parameter sharing.
    
    Args:
        env: The wrapped environment (should be MultiAgentWrapper)
        algorithm: Algorithm to use ("ppo", "td3", or "sac")
        features_dim: Dimension of the feature representation
        **kwargs: Additional arguments for the algorithm
    
    Returns:
        Model with individual drone feature extractor
    """
    
    # Use the individual observation and action spaces
    obs_space = env.individual_obs_space
    action_space = env.individual_action_space
    
    print(f"[create_individual_drone_model] Individual obs space: {obs_space.shape}")
    print(f"[create_individual_drone_model] Individual action space: {action_space.shape}")
    
    # Algorithm-specific configurations
    if algorithm.lower() == "td3":
        policy_kwargs = {
            "features_extractor_class": IndividualDroneExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": {
                "pi": [128, 128],
                "qf": [128, 128]
            },
            "share_features_extractor": False
        }
        
        default_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_delay": 2,
            "target_policy_noise": 0.1,
            "target_noise_clip": 0.5,
        }
        
        default_kwargs.update(kwargs)
        
        model = TD3(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **default_kwargs
        )
        
    elif algorithm.lower() == "sac":
        policy_kwargs = {
            "features_extractor_class": IndividualDroneExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": {
                "pi": [256, 256],
                "qf": [256, 256]
            },
            "share_features_extractor": False
        }
        
        default_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_entropy": "auto",
        }
        
        default_kwargs.update(kwargs)
        
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **default_kwargs
        )
        
    elif algorithm.lower() == "ppo":
        policy_kwargs = {
            "features_extractor_class": IndividualDroneExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        }
        
        default_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        }
        
        default_kwargs.update(kwargs)
        
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **default_kwargs
        )
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"[create_individual_drone_model] Created {algorithm.upper()} with individual drone architecture")
    print(f"[create_individual_drone_model] Features dimension: {features_dim}")
    
    return model


# Test the individual drone architecture
if __name__ == "__main__":
    print("Testing individual drone architecture...")
    
    # Test the feature extractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)  # 18 own + 7 others
    extractor = IndividualDroneExtractor(obs_space, features_dim=256)
    
    # Test forward pass
    dummy_obs = torch.randn(32, 25)
    output = extractor(dummy_obs)
    
    print(f"Input shape: {dummy_obs.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
    
    assert output.shape == (32, 256), f"Expected (32, 256), got {output.shape}"
    print("✓ Individual drone extractor works correctly!")
    
    print("\nIndividual drone architecture ready for training!")