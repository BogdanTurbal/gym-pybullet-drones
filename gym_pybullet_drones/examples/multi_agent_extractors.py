import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gymnasium as gym


class MultiAgentSelfAttentionExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-agent observations that uses self-attention
    to allow agents to attend to each other's states.
    
    Designed for observations of shape (num_agents, agent_obs_dim)
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Extract dimensions from observation space
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Agent encoder - processes individual agent observations
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Self-attention mechanism
        self.attention_dim = 64
        self.query_proj = nn.Linear(64, self.attention_dim)
        self.key_proj = nn.Linear(64, self.attention_dim)
        self.value_proj = nn.Linear(64, self.attention_dim)
        
        # Final aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(64 * self.num_agents, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape to (batch_size * num_agents, agent_obs_dim)
        obs_flat = observations.view(-1, self.agent_obs_dim)
        
        # Encode each agent's observation
        agent_features = self.agent_encoder(obs_flat)  # (batch_size * num_agents, 64)
        
        # Reshape back to (batch_size, num_agents, 64)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        
        # Self-attention
        queries = self.query_proj(agent_features)  # (batch_size, num_agents, attention_dim)
        keys = self.key_proj(agent_features)       # (batch_size, num_agents, attention_dim)
        values = self.value_proj(agent_features)   # (batch_size, num_agents, attention_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_agents, num_agents)
        attention_scores = attention_scores / np.sqrt(self.attention_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_features = torch.matmul(attention_weights, values)  # (batch_size, num_agents, attention_dim)
        
        # Flatten and aggregate
        aggregated = attended_features.view(batch_size, -1)  # (batch_size, num_agents * attention_dim)
        final_features = self.aggregator(aggregated)
        
        return final_features


class MultiAgentMeanPoolExtractor(BaseFeaturesExtractor):
    """
    Simpler multi-agent feature extractor that encodes each agent individually
    and then pools their representations.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Individual agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Global features (mean + individual agent features)
        # Mean pooled features + concatenated individual features
        global_dim = 64 + (64 * self.num_agents)  # pooled + individual
        
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape and encode each agent
        obs_flat = observations.view(-1, self.agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)  # (batch_size * num_agents, 64)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)  # (batch_size, num_agents, 64)
        
        # Global pooled representation
        mean_features = torch.mean(agent_features, dim=1)  # (batch_size, 64)
        
        # Flatten individual agent features
        individual_features = agent_features.view(batch_size, -1)  # (batch_size, num_agents * 64)
        
        # Combine global and individual information
        combined_features = torch.cat([mean_features, individual_features], dim=1)
        
        # Final processing
        final_features = self.global_processor(combined_features)
        
        return final_features


class MultiAgentDeepsetsExtractor(BaseFeaturesExtractor):
    """
    Multi-agent feature extractor using Deep Sets architecture.
    Permutation invariant to agent ordering.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) == 2:
            self.num_agents, self.agent_obs_dim = observation_space.shape
        else:
            raise ValueError(f"Expected 2D observation space, got shape {observation_space.shape}")
        
        super().__init__(observation_space, features_dim)
        
        # Phi network - processes individual agents
        self.phi_network = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Rho network - processes aggregated representation
        self.rho_network = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Process each agent through phi network
        obs_flat = observations.view(-1, self.agent_obs_dim)
        phi_outputs = self.phi_network(obs_flat)  # (batch_size * num_agents, 64)
        phi_outputs = phi_outputs.view(batch_size, self.num_agents, -1)  # (batch_size, num_agents, 64)
        
        # Permutation-invariant aggregation (sum)
        aggregated = torch.sum(phi_outputs, dim=1)  # (batch_size, 64)
        
        # Process through rho network
        final_features = self.rho_network(aggregated)
        
        return final_features


def create_multiagent_ppo_model(env, extractor_type="attention", features_dim=256, **ppo_kwargs):
    """
    Create a PPO model with custom multi-agent feature extractor.
    
    Args:
        env: The environment
        extractor_type: Type of extractor ("attention", "meanpool", "deepsets")
        features_dim: Dimension of the feature representation
        **ppo_kwargs: Additional arguments for PPO
    
    Returns:
        PPO model with custom feature extractor
    """
    
    # Choose the extractor class
    if extractor_type == "attention":
        extractor_class = MultiAgentSelfAttentionExtractor
    elif extractor_type == "meanpool":
        extractor_class = MultiAgentMeanPoolExtractor
    elif extractor_type == "deepsets":
        extractor_class = MultiAgentDeepsetsExtractor
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    # Create policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": features_dim},
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
    
    return model


# Example usage in your training script
def example_usage():
    """
    Example of how to integrate this into your training script
    """
    # Your environment creation code...
    # train_env = make_vec_env(make_env, n_envs=NUM_VEC, seed=0)
    
    # Create model with custom multi-agent architecture
    model = create_multiagent_ppo_model(
        env=train_env,
        extractor_type="attention",  # or "meanpool" or "deepsets"
        features_dim=256,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )
    
    print(f"Model policy architecture:")
    print(model.policy)
    
    # Train as normal
    model.learn(total_timesteps=1000000)
    
    return model


if __name__ == "__main__":
    # You can test the extractors here
    print("Multi-agent feature extractors created successfully!")
    
    # Test with dummy observation space
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 72), dtype=np.float32)
    
    # Test each extractor
    extractors = [
        ("Attention", MultiAgentSelfAttentionExtractor),
        ("MeanPool", MultiAgentMeanPoolExtractor), 
        ("DeepSets", MultiAgentDeepsetsExtractor)
    ]
    
    for name, extractor_class in extractors:
        extractor = extractor_class(obs_space, features_dim=256)
        
        # Test forward pass
        dummy_obs = torch.randn(32, 4, 72)  # batch_size=32, 4 agents, 72 features each
        output = extractor(dummy_obs)
        
        print(f"{name} Extractor:")
        print(f"  Input shape: {dummy_obs.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
        print()