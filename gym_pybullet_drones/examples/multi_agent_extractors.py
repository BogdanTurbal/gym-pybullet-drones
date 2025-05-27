import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gymnasium as gym


class MultiAgentMatrixExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-agent observations in matrix format.
    UPDATED to work with ActionType.RPM (larger observation space).
    
    BaseRLAviary provides:
    - 12 kinematic features per drone
    - Action buffer (ctrl_freq//2 timesteps * 4 RPM values)
    MultiTargetAviary adds:
    - 8 target-related features per drone
    
    Total features per drone ≈ 12 + (15 * 4) + 8 = 80 for ctrl_freq=30
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Extract dimensions from observation space
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Enhanced agent encoder to handle larger observation space
        # More capacity for the larger feature space with RPM actions
        hidden_dim = min(256, max(128, self.agent_obs_dim // 2))  # Adaptive hidden size
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),  # Add dropout for regularization with larger networks
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final aggregation network
        aggregation_input_dim = 64 * self.num_agents
        self.aggregator = nn.Sequential(
            nn.Linear(aggregation_input_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
        
        print(f"[MultiAgentMatrixExtractor] Initialized for {self.num_agents} agents")
        print(f"[MultiAgentMatrixExtractor] Agent obs dim: {self.agent_obs_dim}")
        print(f"[MultiAgentMatrixExtractor] Hidden dim: {hidden_dim}")
        print(f"[MultiAgentMatrixExtractor] Expected breakdown for RPM actions:")
        print(f"  - Base kinematic: 12 features")
        print(f"  - Action buffer: ~{self.agent_obs_dim - 8 - 12} features (4 RPM * buffer_size)")
        print(f"  - Target info: 8 features")
        print(f"[MultiAgentMatrixExtractor] Output features: {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape to (batch_size * num_agents, agent_obs_dim)
        obs_flat = observations.view(-1, self.agent_obs_dim)
        
        # Encode each agent's observation
        agent_features = self.agent_encoder(obs_flat)  # (batch_size * num_agents, 64)
        
        # Reshape back to (batch_size, num_agents, 64)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        
        # Flatten and aggregate
        aggregated = agent_features.view(batch_size, -1)  # (batch_size, num_agents * 64)
        final_features = self.aggregator(aggregated)
        
        return final_features


class MultiAgentSelfAttentionExtractor(BaseFeaturesExtractor):
    """
    Multi-agent feature extractor using self-attention for matrix observations.
    UPDATED to work with ActionType.RPM (larger observation space).
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Enhanced agent encoder for larger observation space
        hidden_dim = min(256, max(128, self.agent_obs_dim // 2))
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Multi-head self-attention mechanism
        self.attention_dim = 64
        self.num_heads = 4  # Multi-head attention
        self.head_dim = self.attention_dim // self.num_heads
        
        self.query_proj = nn.Linear(64, self.attention_dim)
        self.key_proj = nn.Linear(64, self.attention_dim)
        self.value_proj = nn.Linear(64, self.attention_dim)
        self.output_proj = nn.Linear(self.attention_dim, 64)
        
        # Final aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(64 * self.num_agents, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
        
        print(f"[MultiAgentSelfAttentionExtractor] Initialized for {self.num_agents} agents")
        print(f"[MultiAgentSelfAttentionExtractor] Agent obs dim: {self.agent_obs_dim}")
        print(f"[MultiAgentSelfAttentionExtractor] Hidden dim: {hidden_dim}")
        print(f"[MultiAgentSelfAttentionExtractor] Attention heads: {self.num_heads}")
        print(f"[MultiAgentSelfAttentionExtractor] Attention dim: {self.attention_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape to (batch_size * num_agents, agent_obs_dim)
        obs_flat = observations.view(-1, self.agent_obs_dim)
        
        # Encode each agent's observation
        agent_features = self.agent_encoder(obs_flat)  # (batch_size * num_agents, 64)
        
        # Reshape back to (batch_size, num_agents, 64)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        
        # Multi-head self-attention
        queries = self.query_proj(agent_features)  # (batch_size, num_agents, attention_dim)
        keys = self.key_proj(agent_features)       # (batch_size, num_agents, attention_dim)
        values = self.value_proj(agent_features)   # (batch_size, num_agents, attention_dim)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, num_agents, num_agents)
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_values = torch.matmul(attention_weights, values)  # (batch_size, num_heads, num_agents, head_dim)
        
        # Concatenate heads and reshape
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, self.num_agents, self.attention_dim
        )
        
        # Output projection
        attended_features = self.output_proj(attended_values)  # (batch_size, num_agents, 64)
        
        # Flatten and aggregate
        aggregated = attended_features.view(batch_size, -1)  # (batch_size, num_agents * 64)
        final_features = self.aggregator(aggregated)
        
        return final_features


class MultiAgentMeanPoolExtractor(BaseFeaturesExtractor):
    """
    Multi-agent feature extractor that encodes each agent individually
    and then pools their representations.
    UPDATED to work with ActionType.RPM (larger observation space).
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Enhanced individual agent encoder for larger observation space
        hidden_dim = min(256, max(128, self.agent_obs_dim // 2))
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Global features (mean + max + individual agent features)
        # Enhanced pooling with both mean and max pooling
        pooled_dim = 64 * 2  # mean + max pooling
        individual_dim = 64 * self.num_agents
        global_dim = pooled_dim + individual_dim
        
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
        
        print(f"[MultiAgentMeanPoolExtractor] Initialized for {self.num_agents} agents")
        print(f"[MultiAgentMeanPoolExtractor] Agent obs dim: {self.agent_obs_dim}")
        print(f"[MultiAgentMeanPoolExtractor] Hidden dim: {hidden_dim}")
        print(f"[MultiAgentMeanPoolExtractor] Global feature dim: {global_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape and encode each agent
        obs_flat = observations.view(-1, self.agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)  # (batch_size * num_agents, 64)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)  # (batch_size, num_agents, 64)
        
        # Global pooled representations
        mean_features = torch.mean(agent_features, dim=1)  # (batch_size, 64)
        max_features = torch.max(agent_features, dim=1)[0]  # (batch_size, 64)
        pooled_features = torch.cat([mean_features, max_features], dim=1)  # (batch_size, 128)
        
        # Flatten individual agent features
        individual_features = agent_features.view(batch_size, -1)  # (batch_size, num_agents * 64)
        
        # Combine pooled and individual information
        combined_features = torch.cat([pooled_features, individual_features], dim=1)
        
        # Final processing
        final_features = self.global_processor(combined_features)
        
        return final_features


def create_multiagent_ppo_model(env, extractor_type="matrix", features_dim=256, **ppo_kwargs):
    """
    Create a PPO model with custom multi-agent feature extractor for matrix observations.
    UPDATED to work with ActionType.RPM (larger observation space).
    
    Args:
        env: The environment
        extractor_type: Type of extractor ("matrix", "attention", "meanpool")
        features_dim: Dimension of the feature representation
        **ppo_kwargs: Additional arguments for PPO
    
    Returns:
        PPO model with custom feature extractor
    """
    
    # Choose the extractor class
    if extractor_type == "matrix":
        extractor_class = MultiAgentMatrixExtractor
    elif extractor_type == "attention":
        extractor_class = MultiAgentSelfAttentionExtractor
    elif extractor_type == "meanpool":
        extractor_class = MultiAgentMeanPoolExtractor
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    # Create policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": features_dim},
        # Enhanced network architecture for larger observation space
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],  # Larger policy and value networks
    }
    
    # Merge with any additional policy kwargs
    if "policy_kwargs" in ppo_kwargs:
        policy_kwargs.update(ppo_kwargs.pop("policy_kwargs"))
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs
    )
    
    print(f"[create_multiagent_ppo_model] Created PPO with {extractor_type} extractor")
    print(f"[create_multiagent_ppo_model] Enhanced architecture for RPM actions")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing multi-agent feature extractors for RPM actions...")
    
    # Test with dummy observation space matching RPM MultiTargetAviary
    # For ctrl_freq=30 and ActionType.RPM:
    # - BaseRLAviary: 12 kinematic + (15 * 4) action buffer = 72
    # - MultiTargetAviary: +8 target features = 80 total
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 80), dtype=np.float32)
    
    # Test each extractor
    extractors = [
        ("Matrix", MultiAgentMatrixExtractor),
        ("Attention", MultiAgentSelfAttentionExtractor),
        ("MeanPool", MultiAgentMeanPoolExtractor)
    ]
    
    for name, extractor_class in extractors:
        print(f"\n=== Testing {name} Extractor (RPM Actions) ===")
        extractor = extractor_class(obs_space, features_dim=256)
        
        # Test forward pass
        dummy_obs = torch.randn(32, 4, 80)  # batch_size=32, 4 agents, 80 features each
        output = extractor(dummy_obs)
        
        print(f"  Input shape: {dummy_obs.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
        
        # Verify output shape
        assert output.shape == (32, 256), f"Expected (32, 256), got {output.shape}"
        print(f"  ✓ Output shape correct")
        
        # Test with different batch sizes
        for batch_size in [1, 16, 64]:
            test_obs = torch.randn(batch_size, 4, 80)
            test_output = extractor(test_obs)
            assert test_output.shape == (batch_size, 256), f"Failed for batch size {batch_size}"
        print(f"  ✓ Multiple batch sizes work")
    
    print(f"\n{'='*60}")
    print("All extractors tested successfully for RPM actions!")
    print("Ready to use with RPM MultiTargetAviary environment.")
    print("\nObservation space breakdown for ActionType.RPM:")
    print("  - Kinematic features: 12")
    print("  - Action buffer: (ctrl_freq//2) * 4 RPM values")
    print("  - Target features: 8")
    print("  - Total: ~80 features per drone (for ctrl_freq=30)")
    print("The extractors automatically adapt to the actual observation size.")