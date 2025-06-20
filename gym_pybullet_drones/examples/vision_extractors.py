"""
vision_extractors.py - Custom feature extractors for vision-based drone control

This module provides specialized neural network architectures for processing
Dict observation spaces with separate kinematic and depth vision data.
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiModalDroneExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations with kinematic and depth data.
    
    Processes kinematic data with MLP and depth maps with CNN,
    then combines the features for better decision-making.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Need to pass an empty Dummy space to the parent as we'll handle the Dict
        super().__init__(observation_space, features_dim)
        
        # Extract the structure from the first drone (all drones have the same structure)
        if '0' in observation_space.spaces:
            # Multi-drone case
            self.num_drones = len(observation_space.spaces)
            first_drone_space = observation_space.spaces['0']
        else:
            # Single drone case
            self.num_drones = 1
            first_drone_space = observation_space
        
        # Check if we have a Dict for each drone
        if not isinstance(first_drone_space, gym.spaces.Dict):
            raise ValueError("Expected Dict observation space for each drone")
        
        # Extract dimensions from the spaces
        # Kinematic dimensions
        if 'kinematics' in first_drone_space.spaces:
            kin_space = first_drone_space.spaces['kinematics']
            if isinstance(kin_space, gym.spaces.Box):
                self.kin_dim = kin_space.shape[0]
            else:
                raise ValueError("Expected Box space for kinematics")
        else:
            raise ValueError("Missing 'kinematics' key in observation space")
        
        # Depth map dimensions
        if 'depth' in first_drone_space.spaces:
            depth_space = first_drone_space.spaces['depth']
            if isinstance(depth_space, gym.spaces.Box):
                self.depth_height, self.depth_width = depth_space.shape
            else:
                raise ValueError("Expected Box space for depth map")
        else:
            raise ValueError("Missing 'depth' key in observation space")
        
        print(f"[MultiModalDroneExtractor] Num drones: {self.num_drones}")
        print(f"[MultiModalDroneExtractor] Kinematics dim: {self.kin_dim}")
        print(f"[MultiModalDroneExtractor] Depth map: {self.depth_width}x{self.depth_height}")
        
        # CNN for depth maps
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        
        # Calculate CNN output dimension based on input dimensions and strides
        # Height and width are divided by 2 three times due to stride=2
        cnn_height = self.depth_height // 8
        cnn_width = self.depth_width // 8
        cnn_output_dim = 64 * cnn_height * cnn_width
        
        # MLP for kinematic features
        self.kin_encoder = nn.Sequential(
            nn.Linear(self.kin_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        
        # Combiner network
        self.combiner = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )
        
        # Set output dimension
        self._features_dim = features_dim
    
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Process Dict observations with separate CNNs and MLPs.
        
        Args:
            observations: Dict with structure {drone_id: {kinematics: tensor, depth: tensor}}
        
        Returns:
            torch.Tensor: Extracted features
        """
        batch_size = next(iter(next(iter(observations.values())).values())).shape[0]
        
        # Process each drone separately
        all_features = []
        
        for i in range(self.num_drones):
            drone_key = str(i)
            
            if drone_key not in observations:
                raise ValueError(f"Missing key {drone_key} in observations dictionary")
            
            drone_obs = observations[drone_key]
            
            # Extract modalities
            kin_obs = drone_obs['kinematics']
            depth_obs = drone_obs['depth']
            
            # Process kinematic features
            kin_features = self.kin_encoder(kin_obs)
            
            # Process depth maps with CNN
            # Reshape to (batch, channels, height, width)
            depth_obs = depth_obs.view(batch_size, 1, self.depth_height, self.depth_width)
            cnn_features = self.cnn(depth_obs)
            
            # Combine features
            combined = torch.cat([kin_features, cnn_features], dim=1)
            drone_features = self.combiner(combined)
            
            all_features.append(drone_features)
        
        # Handle multi-drone case
        if self.num_drones > 1:
            # Stack and mean-pool features from all drones
            stacked_features = torch.stack(all_features, dim=1)
            return torch.mean(stacked_features, dim=1)
        else:
            # Single drone case
            return all_features[0]


class AttentionDroneExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with self-attention mechanism for Dict observations.
    
    Uses attention to focus on important parts of the depth maps and
    combines them with kinematic data.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract the structure (same as above)
        if '0' in observation_space.spaces:
            self.num_drones = len(observation_space.spaces)
            first_drone_space = observation_space.spaces['0']
        else:
            self.num_drones = 1
            first_drone_space = observation_space
        
        if not isinstance(first_drone_space, gym.spaces.Dict):
            raise ValueError("Expected Dict observation space for each drone")
        
        # Extract dimensions
        if 'kinematics' in first_drone_space.spaces:
            kin_space = first_drone_space.spaces['kinematics']
            self.kin_dim = kin_space.shape[0]
        else:
            raise ValueError("Missing 'kinematics' key in observation space")
        
        if 'depth' in first_drone_space.spaces:
            depth_space = first_drone_space.spaces['depth']
            self.depth_height, self.depth_width = depth_space.shape
        else:
            raise ValueError("Missing 'depth' key in observation space")
        
        print(f"[AttentionDroneExtractor] Num drones: {self.num_drones}")
        print(f"[AttentionDroneExtractor] Kinematics dim: {self.kin_dim}")
        print(f"[AttentionDroneExtractor] Depth map: {self.depth_width}x{self.depth_height}")
        
        # CNN for depth maps with intermediate features for attention
        self.cnn_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        
        # Calculate feature map size after two stride-2 convs
        feature_h, feature_w = self.depth_height // 4, self.depth_width // 4
        self.feature_size = feature_h * feature_w
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=32,  # Channel dimension from CNN
            num_heads=4,
            batch_first=True
        )
        
        # Post-attention processing
        self.post_attention = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        
        # Calculate CNN output dimension
        cnn_output_height = feature_h // 2  # One more stride-2 conv
        cnn_output_width = feature_w // 2
        cnn_output_dim = 64 * cnn_output_height * cnn_output_width
        
        # MLP for kinematic features
        self.kin_encoder = nn.Sequential(
            nn.Linear(self.kin_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        
        # Final combiner
        self.combiner = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = next(iter(next(iter(observations.values())).values())).shape[0]
        
        all_features = []
        
        for i in range(self.num_drones):
            drone_key = str(i)
            
            if drone_key not in observations:
                raise ValueError(f"Missing key {drone_key} in observations dictionary")
            
            drone_obs = observations[drone_key]
            
            # Extract modalities
            kin_obs = drone_obs['kinematics']
            depth_obs = drone_obs['depth'].view(batch_size, 1, self.depth_height, self.depth_width)
            
            # Process kinematic features
            kin_features = self.kin_encoder(kin_obs)
            
            # Get CNN feature maps for attention
            feature_maps = self.cnn_features(depth_obs)  # (batch, 32, H/4, W/4)
            
            # Reshape for attention: (batch, seq_len, embed_dim)
            # Where seq_len = H*W (spatial locations) and embed_dim = channels
            b, c, h, w = feature_maps.shape
            feature_maps_flat = feature_maps.view(b, c, h*w).permute(0, 2, 1)  # (batch, H*W, 32)
            
            # Apply self-attention
            attended_features, _ = self.attention(
                feature_maps_flat, feature_maps_flat, feature_maps_flat
            )
            
            # Reshape back to spatial
            attended_spatial = attended_features.permute(0, 2, 1).view(b, c, h, w)
            
            # Apply post-attention processing
            cnn_features = self.post_attention(attended_spatial)
            
            # Combine with kinematic features
            combined = torch.cat([kin_features, cnn_features], dim=1)
            drone_features = self.combiner(combined)
            
            all_features.append(drone_features)
        
        # Handle multi-drone case
        if self.num_drones > 1:
            stacked_features = torch.stack(all_features, dim=1)
            return torch.mean(stacked_features, dim=1)
        else:
            return all_features[0]


# Simple extractor for a single drone with dict observations
class SimpleDroneExtractor(BaseFeaturesExtractor):
    """
    Simplified feature extractor for Dict observations.
    
    This extractor is more straightforward and uses simpler networks,
    making it faster to train but potentially less powerful.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract the structure
        if '0' in observation_space.spaces:
            self.num_drones = len(observation_space.spaces)
            first_drone_space = observation_space.spaces['0']
        else:
            self.num_drones = 1
            first_drone_space = observation_space
        
        # Extract dimensions
        if 'kinematics' in first_drone_space.spaces:
            kin_space = first_drone_space.spaces['kinematics']
            self.kin_dim = kin_space.shape[0]
        else:
            raise ValueError("Missing 'kinematics' key in observation space")
        
        if 'depth' in first_drone_space.spaces:
            depth_space = first_drone_space.spaces['depth']
            self.depth_height, self.depth_width = depth_space.shape
            self.depth_size = self.depth_height * self.depth_width
        else:
            raise ValueError("Missing 'depth' key in observation space")
        
        print(f"[SimpleDroneExtractor] Num drones: {self.num_drones}")
        print(f"[SimpleDroneExtractor] Kinematics dim: {self.kin_dim}")
        print(f"[SimpleDroneExtractor] Depth map: {self.depth_width}x{self.depth_height}")
        
        # Simple CNN for depth
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),  # Larger kernel, more stride
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output dimension
        # With 64x48 input, after stride 4 and stride 2, we get roughly 8x6 feature maps
        h_out = (self.depth_height + 2*2 - 8) // 4 + 1  # First conv
        w_out = (self.depth_width + 2*2 - 8) // 4 + 1
        
        h_out = (h_out + 2*1 - 4) // 2 + 1  # Second conv
        w_out = (w_out + 2*1 - 4) // 2 + 1
        
        cnn_output_dim = 32 * h_out * w_out
        
        # Simple MLP for kinematic data
        self.kin_net = nn.Sequential(
            nn.Linear(self.kin_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = next(iter(next(iter(observations.values())).values())).shape[0]
        
        all_features = []
        
        for i in range(self.num_drones):
            drone_key = str(i)
            
            if drone_key not in observations:
                raise ValueError(f"Missing key {drone_key} in observations dictionary")
            
            drone_obs = observations[drone_key]
            
            # Extract modalities
            kin_obs = drone_obs['kinematics']
            depth_obs = drone_obs['depth'].view(batch_size, 1, self.depth_height, self.depth_width)
            
            # Process each modality
            kin_features = self.kin_net(kin_obs)
            cnn_features = self.cnn(depth_obs)
            
            # Combine
            combined = torch.cat([kin_features, cnn_features], dim=1)
            drone_features = self.combiner(combined)
            
            all_features.append(drone_features)
        
        # Handle multi-drone case
        if self.num_drones > 1:
            stacked_features = torch.stack(all_features, dim=1)
            return torch.mean(stacked_features, dim=1)
        else:
            return all_features[0]


# Example usage and testing
if __name__ == "__main__":
    # Create a test observation space with dict structure
    num_drones = 1
    kin_dim = 24  # Kinematic features
    img_width, img_height = 64, 48
    
    # Create a Dict observation space
    obs_dict = {}
    
    for i in range(num_drones):
        drone_spaces = {
            "kinematics": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(kin_dim,), dtype=np.float32),
            "depth": gym.spaces.Box(low=0.01, high=1000.0, shape=(img_height, img_width), dtype=np.float32),
        }
        obs_dict[str(i)] = gym.spaces.Dict(drone_spaces)
    
    obs_space = gym.spaces.Dict(obs_dict)
    
    # Create extractors
    cnn_extractor = MultiModalDroneExtractor(obs_space, features_dim=256)
    attn_extractor = AttentionDroneExtractor(obs_space, features_dim=256)
    simple_extractor = SimpleDroneExtractor(obs_space, features_dim=256)
    
    # Create a dummy batch of observations
    batch_size = 4
    dummy_obs = {}
    
    for i in range(num_drones):
        dummy_obs[str(i)] = {
            "kinematics": torch.rand(batch_size, kin_dim),
            "depth": torch.rand(batch_size, img_height, img_width),
        }
    
    # Test forward pass
    cnn_output = cnn_extractor(dummy_obs)
    attn_output = attn_extractor(dummy_obs)
    simple_output = simple_extractor(dummy_obs)
    
    print(f"CNN Extractor output shape: {cnn_output.shape}")
    print(f"Attention Extractor output shape: {attn_output.shape}")
    print(f"Simple Extractor output shape: {simple_output.shape}")