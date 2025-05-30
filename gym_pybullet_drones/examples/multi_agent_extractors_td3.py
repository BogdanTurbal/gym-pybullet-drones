import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.td3.policies import TD3Policy
import gymnasium as gym
from typing import Dict, Type, Union, Optional, List, Tuple


class MultiAgentMatrixExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-agent observations in matrix format.
    Works with both PPO and TD3 policies.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Extract dimensions from observation space
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Enhanced agent encoder
        hidden_dim = 128
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        
        # Final aggregation network
        aggregation_input_dim = 64 * self.num_agents
        self.aggregator = nn.Sequential(
            nn.Linear(aggregation_input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )
        
        self._features_dim = features_dim
        
        print(f"[MultiAgentMatrixExtractor] Initialized for {self.num_agents} agents")
        print(f"[MultiAgentMatrixExtractor] Agent obs dim: {self.agent_obs_dim}")
        print(f"[MultiAgentMatrixExtractor] Output features: {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape to (batch_size * num_agents, agent_obs_dim)
        #shape_1 = observations.shape[1]
        obs_flat = observations.view(-1, self.agent_obs_dim)
        
        # Encode each agent's observation
        agent_features = self.agent_encoder(obs_flat)
        
        # Reshape back to (batch_size, num_agents, 64)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        
        # Flatten and aggregate
        aggregated = agent_features.view(batch_size, -1)
        final_features = self.aggregator(aggregated)
        
        return final_features


class MultiAgentSelfAttentionExtractor(BaseFeaturesExtractor):
    """
    Multi-agent feature extractor using self-attention.
    Works with both PPO and TD3 policies.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        
        # Multi-head self-attention
        self.attention_dim = 64
        self.num_heads = 4
        self.head_dim = self.attention_dim // self.num_heads
        
        self.query_proj = nn.Linear(64, self.attention_dim)
        self.key_proj = nn.Linear(64, self.attention_dim)
        self.value_proj = nn.Linear(64, self.attention_dim)
        self.output_proj = nn.Linear(self.attention_dim, 64)
        
        # Final aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(64 * self.num_agents, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )
        
        self._features_dim = features_dim
        
        print(f"[MultiAgentSelfAttentionExtractor] Initialized for {self.num_agents} agents")
        print(f"[MultiAgentSelfAttentionExtractor] Attention heads: {self.num_heads}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Encode agents
        obs_flat = observations.view(-1, self.agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        
        # Multi-head self-attention
        queries = self.query_proj(agent_features)
        keys = self.key_proj(agent_features)
        values = self.value_proj(agent_features)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, self.num_agents, self.attention_dim
        )
        attended_features = self.output_proj(attended_values)
        
        # Aggregate
        aggregated = attended_features.view(batch_size, -1)
        final_features = self.aggregator(aggregated)
        
        return final_features


class MultiAgentMeanPoolExtractor(BaseFeaturesExtractor):
    """
    Multi-agent feature extractor using mean pooling.
    Works with both PPO and TD3 policies.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        
        # Global processor (mean + max + individual)
        pooled_dim = 64 * 2  # mean + max pooling
        individual_dim = 64 * self.num_agents
        global_dim = pooled_dim + individual_dim
        
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )
        
        self._features_dim = features_dim
        
        print(f"[MultiAgentMeanPoolExtractor] Initialized for {self.num_agents} agents")
        print(f"[MultiAgentMeanPoolExtractor] Global feature dim: {global_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Encode agents
        obs_flat = observations.view(-1, self.agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        
        # Pooling
        mean_features = torch.mean(agent_features, dim=1)
        max_features = torch.max(agent_features, dim=1)[0]
        pooled_features = torch.cat([mean_features, max_features], dim=1)
        
        # Individual features
        individual_features = agent_features.view(batch_size, -1)
        
        # Combine and process
        combined_features = torch.cat([pooled_features, individual_features], dim=1)
        final_features = self.global_processor(combined_features)
        
        return final_features


def create_multiagent_model(
    env, 
    algorithm: str = "td3",
    extractor_type: str = "matrix", 
    features_dim: int = 256, 
    **kwargs
):
    """
    Create a model (PPO, TD3, or SAC) with custom multi-agent feature extractor.
    
    Args:
        env: The environment
        algorithm: Algorithm to use ("ppo", "td3", or "sac")
        extractor_type: Type of extractor ("matrix", "attention", "meanpool")
        features_dim: Dimension of the feature representation
        **kwargs: Additional arguments for the algorithm
    
    Returns:
        Model with custom feature extractor
    """
    
    # Choose the extractor class
    extractors = {
        "matrix": MultiAgentMatrixExtractor,
        "attention": MultiAgentSelfAttentionExtractor,
        "meanpool": MultiAgentMeanPoolExtractor
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    extractor_class = extractors[extractor_type]
    
    # Algorithm-specific configurations
    if algorithm.lower() == "td3":
        # TD3-specific policy kwargs
        policy_kwargs = {
            #"features_extractor_class": extractor_class,
            #"features_extractor_kwargs": {"features_dim": features_dim},
            # TD3 uses separate networks for actor and critic
            "net_arch": {
                "pi": [128, 128],  # Actor network
                "qf": [128, 128]   # Critic network
            },
            "share_features_extractor": True  # Don't share between actor and critic
        }
        
        # Default TD3 hyperparameters
        default_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 10000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "action_noise": None,  # Will be set based on action space
            "policy_delay": 2,
            "target_policy_noise": 0.1,
            "target_noise_clip": 0.5,
        }
        
        # Merge with user kwargs
        default_kwargs.update(kwargs)
        
        
        print(env.action_space)
        
        model = TD3(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            **default_kwargs
        )
        
    elif algorithm.lower() == "sac":
        # SAC-specific policy kwargs
        policy_kwargs = {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": {
                "pi": [256, 256],
                "qf": [256, 256]
            },
            "share_features_extractor": False
        }
        
        # Default SAC hyperparameters
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
        # PPO-specific policy kwargs
        # policy_kwargs = {
        #     "features_extractor_class": extractor_class,
        #     "features_extractor_kwargs": {"features_dim": features_dim},
        #     "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        # }
        
        # Default PPO hyperparameters
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
            #policy_kwargs=policy_kwargs,
            **default_kwargs
        )
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"[create_multiagent_model] Created {algorithm.upper()} with {extractor_type} extractor")
    print(f"[create_multiagent_model] Features dimension: {features_dim}")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing multi-agent feature extractors for multiple algorithms...")
    
    # Test observation space
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 80), dtype=np.float32)
    
    # Test each extractor
    extractors = [
        ("Matrix", MultiAgentMatrixExtractor),
        ("Attention", MultiAgentSelfAttentionExtractor),
        ("MeanPool", MultiAgentMeanPoolExtractor)
    ]
    
    for name, extractor_class in extractors:
        print(f"\n=== Testing {name} Extractor ===")
        extractor = extractor_class(obs_space, features_dim=256)
        
        # Test forward pass
        dummy_obs = torch.randn(32, 4, 80)
        output = extractor(dummy_obs)
        
        print(f"  Input shape: {dummy_obs.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
        
        # Verify output shape
        assert output.shape == (32, 256), f"Expected (32, 256), got {output.shape}"
        print(f"  ✓ Output shape correct")
    
    print(f"\n{'='*60}")
    print("All extractors tested successfully!")
    print("Ready for use with TD3, SAC, and PPO algorithms.")