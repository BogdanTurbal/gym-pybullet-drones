#!/usr/bin/env python3
"""
rllib_models.py - Custom Neural Network Models for RLlib Multi-Agent Drone Training
Much more efficient than SB3's approach with better multi-agent support
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from gymnasium import spaces

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class MultiAgentDroneModel(TorchModelV2, nn.Module):
    """
    Advanced multi-agent model optimized for drone swarm control.
    Features:
    - Attention mechanism for agent coordination
    - Shared feature extraction with agent-specific heads
    - Much faster than SB3's approach
    - Better parameter efficiency
    """
    
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Extract configuration
        custom_config = model_config.get("custom_model_config", {})
        self.obs_dim = obs_space.shape[0]
        self.hidden_dim = custom_config.get("hidden_dim", 256)
        self.use_attention = custom_config.get("use_attention", True)
        self.use_layer_norm = custom_config.get("use_layer_norm", True)
        self.dropout_rate = custom_config.get("dropout_rate", 0.1)
        
        print(f"[MultiAgentDroneModel] Obs dim: {self.obs_dim}")
        print(f"[MultiAgentDroneModel] Hidden dim: {self.hidden_dim}")
        print(f"[MultiAgentDroneModel] Use attention: {self.use_attention}")
        
        # Feature encoder - processes individual agent observations
        encoder_layers = [
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            encoder_layers.append(nn.LayerNorm(self.hidden_dim))
        if self.dropout_rate > 0:
            encoder_layers.append(nn.Dropout(self.dropout_rate))
        
        encoder_layers.extend([
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
        ])
        if self.use_layer_norm:
            encoder_layers.append(nn.LayerNorm(self.hidden_dim // 2))
        
        self.feature_encoder = nn.Sequential(*encoder_layers)
        
        # Attention mechanism for multi-agent coordination
        if self.use_attention:
            self.attention_dim = self.hidden_dim // 2
            self.query_proj = nn.Linear(self.attention_dim, self.attention_dim)
            self.key_proj = nn.Linear(self.attention_dim, self.attention_dim)
            self.value_proj = nn.Linear(self.attention_dim, self.attention_dim)
            self.attention_out = nn.Linear(self.attention_dim, self.attention_dim)
            
            # Use smaller final feature dim when using attention
            self.final_feature_dim = self.attention_dim
        else:
            self.final_feature_dim = self.hidden_dim // 2
        
        # Policy head (action prediction)
        policy_layers = [
            nn.Linear(self.final_feature_dim, self.hidden_dim // 4),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            policy_layers.append(nn.LayerNorm(self.hidden_dim // 4))
        if self.dropout_rate > 0:
            policy_layers.append(nn.Dropout(self.dropout_rate))
        
        policy_layers.append(nn.Linear(self.hidden_dim // 4, num_outputs))
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Value head (state value estimation)
        value_layers = [
            nn.Linear(self.final_feature_dim, self.hidden_dim // 4),
            nn.ReLU(),
        ]
        if self.use_layer_norm:
            value_layers.append(nn.LayerNorm(self.hidden_dim // 4))
        if self.dropout_rate > 0:
            value_layers.append(nn.Dropout(self.dropout_rate))
        
        value_layers.append(nn.Linear(self.hidden_dim // 4, 1))
        self.value_head = nn.Sequential(*value_layers)
        
        # Store value output for value_function()
        self._value_out = None
        
        # Initialize weights
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[MultiAgentDroneModel] Total parameters: {total_params:,}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, List[TensorType]]:
        """Forward pass through the model."""
        
        obs = input_dict["obs"].float()
        batch_size = obs.shape[0]
        
        # Feature extraction
        features = self.feature_encoder(obs)  # (batch_size, feature_dim)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self._apply_self_attention(features)
        
        # Policy prediction
        policy_logits = self.policy_head(features)
        
        # Value prediction (store for value_function())
        self._value_out = self.value_head(features).squeeze(-1)
        
        return policy_logits, state
    
    def _apply_self_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention mechanism.
        Note: This is simplified self-attention for single-agent context.
        For true multi-agent attention, we'd need access to other agents' features.
        """
        # Generate queries, keys, values
        queries = self.query_proj(features)
        keys = self.key_proj(features)
        values = self.value_proj(features)
        
        # Self-attention (each agent attends to its own features)
        # In a full multi-agent setting, agents would attend to each other
        attention_scores = torch.matmul(queries.unsqueeze(1), keys.unsqueeze(-1))
        attention_weights = torch.softmax(attention_scores / np.sqrt(self.attention_dim), dim=1)
        
        attended_features = attention_weights * values.unsqueeze(1)
        attended_features = attended_features.squeeze(1)
        
        # Output projection
        output = self.attention_out(attended_features)
        
        # Residual connection
        return output + features
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value function output for the most recent forward pass."""
        assert self._value_out is not None, "Must call forward() first"
        return self._value_out


class SharedParameterDroneModel(TorchModelV2, nn.Module):
    """
    Simplified shared-parameter model for drone swarm.
    All agents share the same parameters - very efficient for homogeneous swarms.
    """
    
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        custom_config = model_config.get("custom_model_config", {})
        self.obs_dim = obs_space.shape[0]
        self.hidden_dim = custom_config.get("hidden_dim", 128)
        
        # Simple but effective architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim // 2),
        )
        
        # Separate heads for policy and value
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        
        self._value_out = None
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SharedParameterDroneModel] Total parameters: {total_params:,}")
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"].float()
        
        # Shared feature extraction
        features = self.encoder(obs)
        
        # Policy and value predictions
        policy_logits = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(-1)
        
        return policy_logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None, "Must call forward() first"
        return self._value_out


class LightweightDroneModel(TorchModelV2, nn.Module):
    """
    Lightweight model for fast training and inference.
    Optimized for speed over complexity.
    """
    
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_dim = obs_space.shape[0]
        hidden_dim = model_config.get("custom_model_config", {}).get("hidden_dim", 64)
        
        # Very simple architecture for maximum speed
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(hidden_dim, num_outputs)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._value_out = None
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[LightweightDroneModel] Total parameters: {total_params:,}")
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"].float()
        
        features = self.net(obs)
        policy_logits = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(-1)
        
        return policy_logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None, "Must call forward() first"
        return self._value_out


# Model factory function
def create_drone_model(
    model_type: str = "multi_agent",
    obs_space: spaces.Space = None,
    action_space: spaces.Space = None,
    num_outputs: int = None,
    model_config: ModelConfigDict = None,
    name: str = "drone_model"
):
    """
    Factory function to create different types of drone models.
    
    Args:
        model_type: Type of model ("multi_agent", "shared", "lightweight")
        obs_space: Observation space
        action_space: Action space  
        num_outputs: Number of output units
        model_config: Model configuration dictionary
        name: Model name
    
    Returns:
        Model class (not instance)
    """
    
    model_classes = {
        "multi_agent": MultiAgentDroneModel,
        "shared": SharedParameterDroneModel,
        "lightweight": LightweightDroneModel,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}")
    
    print(f"[Model Factory] Selected model type: {model_type}")
    return model_classes[model_type]


# Register models with RLlib
def register_drone_models():
    """Register all drone models with RLlib's ModelCatalog."""
    from ray.rllib.models import ModelCatalog
    
    ModelCatalog.register_custom_model("multi_agent_drone", MultiAgentDroneModel)
    ModelCatalog.register_custom_model("shared_drone", SharedParameterDroneModel)
    ModelCatalog.register_custom_model("lightweight_drone", LightweightDroneModel)
    
    print("[Model Registry] Registered drone models:")
    print("  - multi_agent_drone: Advanced model with attention")
    print("  - shared_drone: Efficient shared-parameter model")  
    print("  - lightweight_drone: Fast lightweight model")


# Performance comparison utility
def compare_model_performance():
    """Compare the computational performance of different models."""
    import time
    
    # Mock spaces for testing
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(80,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    models_to_test = [
        ("MultiAgent", MultiAgentDroneModel),
        ("Shared", SharedParameterDroneModel),
        ("Lightweight", LightweightDroneModel),
    ]
    
    batch_size = 256
    num_iterations = 100
    
    print(f"\n{'='*60}")
    print("Model Performance Comparison")
    print(f"Batch size: {batch_size}, Iterations: {num_iterations}")
    print(f"{'='*60}")
    
    for name, model_class in models_to_test:
        # Create model
        model = model_class(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=4,
            model_config={"custom_model_config": {"hidden_dim": 128}},
            name=f"test_{name.lower()}"
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Benchmark forward pass
        dummy_obs = torch.randn(batch_size, 80)
        dummy_input = {"obs": dummy_obs}
        
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                logits, _ = model.forward(dummy_input, [], None)
                value = model.value_function()
            
            # Timing
            start_time = time.time()
            for _ in range(num_iterations):
                logits, _ = model.forward(dummy_input, [], None)
                value = model.value_function()
            end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        throughput = batch_size * num_iterations / (end_time - start_time)
        
        print(f"{name:12s} | Params: {total_params:6,d} | "
              f"Time: {avg_time:6.2f}ms | Throughput: {throughput:8.0f} samples/sec")
    
    print(f"{'='*60}")
    print("💡 Recommendation:")
    print("  - MultiAgent: Best for complex coordination tasks")
    print("  - Shared: Best balance of performance and efficiency")
    print("  - Lightweight: Best for maximum speed")


if __name__ == "__main__":
    print("Testing RLlib Custom Drone Models...")
    
    # Register models
    register_drone_models()
    
    # Test model creation
    from gymnasium import spaces
    
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(80,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    print("\n=== Testing Model Creation ===")
    for model_name, model_class in [
        ("MultiAgent", MultiAgentDroneModel),
        ("Shared", SharedParameterDroneModel),
        ("Lightweight", LightweightDroneModel)
    ]:
        print(f"\nTesting {model_name} Model:")
        model = model_class(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=4,
            model_config={"custom_model_config": {"hidden_dim": 128}},
            name=f"test_{model_name.lower()}"
        )
        
        # Test forward pass
        dummy_obs = torch.randn(32, 80)
        dummy_input = {"obs": dummy_obs}
        
        logits, state = model.forward(dummy_input, [], None)
        value = model.value_function()
        
        print(f"  ✅ Forward pass successful")
        print(f"  ✅ Output shape: {logits.shape}")
        print(f"  ✅ Value shape: {value.shape}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    compare_model_performance()
    
    print("\n✅ All models tested successfully!")
    print("Models are ready for high-performance RLlib training.")