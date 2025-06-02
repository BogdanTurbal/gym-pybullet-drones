import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3 import PPO, TD3, SAC, DDPG
# from stable_baselines3.common.policies import BasePolicy # Not directly used here
# from stable_baselines3.td3.policies import TD3Policy # Not directly used here
import gymnasium as gym # Changed from gym to gymnasium
from typing import Dict, Type, Union, Optional, List, Tuple

import torch as th
from gymnasium import spaces
from torch import nn


class NatureCNNSmall(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        #print(observation_space.shape)
        n_input_channels = observation_space.shape[-1] * observation_space.shape[-2]
        # for i in range(100):
        #   print(n_input_channels)

        self.cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_channels, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, features_dim),
            nn.LeakyReLU(0.1),
        )

        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        #self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print('-------')
        # print(observations.shape)
        cnn_res = self.cnn(observations)
        # print(cnn_res.shape)
        return cnn_res#self.linear(cnn_res)



class KinDepthExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for combined KINematic and DEPTH observations.
    Handles Dict observation space: {"kin": Box, "depth": Box}.
    Designed for NUM_DRONES=1, where tensors are (batch_size, *feature_shape).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        extractors = {}
        total_concat_size = 0

        # KIN feature extractor (MLP)
        kin_space = observation_space.spaces["kin"]
        kin_input_dim = kin_space.shape[0] - 2 # Assumes kin_space is (features,)
        # Simple MLP for KIN, can be made more complex
        kin_feature_dim = 32 # Output dim for KIN features
        extractors["kin"] = nn.Sequential(
            nn.Linear(kin_input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            #nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(64, kin_feature_dim),
            nn.LeakyReLU(0.1),
            #nn.ReLU(),
        )
        total_concat_size += kin_feature_dim
#python train_multi_td3.py --algorithm ppo --num_drones 1 --features_dim 32 --ctrl_freq 24 --extractor_type kin_depth_auto
        # DEPTH feature extractor (CNN)
        # SB3's NatureCNN is a common choice for image data
        # It expects input shape (batch_size, C, H, W)
        depth_space = observation_space.spaces["depth"] # Shape (H, W, C), dtype=float32

        # Create a NatureCNN compatible space: (C, H, W)
        # Keep the float32 dtype from the original depth space
        # cnn_input_space = gym.spaces.Box(
        #     low=np.min(depth_space.low), # Use actual low/high from depth_space
        #     high=np.max(depth_space.high),
        #     shape=(depth_space.shape[2], depth_space.shape[0], depth_space.shape[1]), # (C, H, W)
        #     dtype=depth_space.dtype # This will be float32
        # )
        # Since the input is float32 and not necessarily normalized 0-1 pixels,
        # tell NatureCNN not to expect standard normalized image input if its checks are strict.
        # However, the main issue is the assertion `is_image_space` which checks for uint8 by default.
        # We might need a simpler CNN if NatureCNN is too restrictive with its input space definition for float data.

        # Let's try to satisfy `is_image_space` by using uint8 for the dummy space passed to NatureCNN,
        # but ensure our actual float data is handled. This is a bit of a workaround for the assertion.
        # The data passed to forward() will still be float.
        # A better long-term solution might be a custom CNN for float depth maps.

        temp_cnn_input_space_for_assertion = gym.spaces.Box(
            low=0, high=255, # Standard image range for assertion
            shape=(depth_space.shape[2], depth_space.shape[0], depth_space.shape[1]), # (C, H, W)
            dtype=np.uint8 # Standard image dtype for assertion
        )
        extractors["depth"] = NatureCNNSmall(temp_cnn_input_space_for_assertion, features_dim=32)
        total_concat_size += 32
        # extractors["depth"] = NatureCNN(cnn_input_space, features_dim=256) # Output 256 features from CNN
        # total_concat_size += 256 # NatureCNN's default output if features_dim not forced on it effectively

        self.extractors = nn.ModuleDict(extractors)

        # Final MLP to combine features from all modalities
        self.fc = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.LeakyReLU(0.1)
            #nn.Tanh()
            #nn.ReLU()
        )
        
        self._features_dim = features_dim # Ensure this is set

        print(f"[KinDepthExtractor] Initialized. KIN input: {kin_input_dim}, Depth input: {depth_space.shape}")
        print(f"[KinDepthExtractor] KIN features: {kin_feature_dim}, Depth CNN features: 256 (from NatureCNN)")
        print(f"[KinDepthExtractor] Combined features before final FC: {total_concat_size}")
        print(f"[KinDepthExtractor] Output features_dim: {features_dim}")


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []

        # KIN features
        kin_obs = observations["kin"][:, 2:]
        #return self.extractors["kin"](kin_obs)
        # If kin_obs is (batch_size, num_drones, kin_dim) for multi-drone, flatten num_drones for MLP
        # Assuming for now it's (batch_size, kin_dim) if NUM_DRONES=1 (handled by VecEnv)
        encoded_tensor_list.append(self.extractors["kin"](kin_obs))

        # DEPTH features
        depth_obs = observations["depth"] # Expected (batch_size, H, W, C) from env
        # Permute to (batch_size, C, H, W) for NatureCNN
        # print('======')
        # print(depth_obs.shape)
        depth_obs_permuted = depth_obs.permute(0, 3, 1, 2)
        # print(depth_obs_permuted.shape)
        encoded_tensor_list.append(self.extractors["depth"](depth_obs_permuted))
        
        # Concatenate features and pass through final FC layer
        concatenated_features = torch.cat(encoded_tensor_list, dim=1)
        return self.fc(concatenated_features) #


# --- Existing Extractors (MultiAgentMatrixExtractor, etc.) ---
# These are designed for Box observation space of shape (NUM_AGENTS, AGENT_OBS_DIM)
# They will not work directly with the new Dict KIN_DEPTH observation space.
# Keep them for other use cases or if you switch obs type.

class MultiAgentMatrixExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-agent observations in matrix format.
    Works with both PPO and TD3 policies.
    Input observation_space: Box(num_agents, agent_obs_dim)
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 2:
             raise ValueError(f"MultiAgentMatrixExtractor expects Box obs space of shape (num_agents, agent_obs_dim), got {observation_space}")
        
        self.num_agents, self.agent_obs_dim = observation_space.shape
        super().__init__(observation_space, features_dim)
        
        hidden_dim = 32
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 32), nn.LeakyReLU(0.2),
        )
        aggregation_input_dim = 32 * self.num_agents
        self.aggregator = nn.Sequential(
            nn.Linear(aggregation_input_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, features_dim), nn.LeakyReLU(0.2),
        )
        self._features_dim = features_dim
        print(f"[MultiAgentMatrixExtractor] Initialized for {self.num_agents} agents, agent_obs_dim: {self.agent_obs_dim}, output: {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations expected shape: (batch_size, num_agents, agent_obs_dim)
        batch_size = observations.shape[0]
        obs_flat = observations.reshape(-1, self.agent_obs_dim) # (batch_size * num_agents, agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)
        agent_features = agent_features.view(batch_size, self.num_agents, -1) # (batch_size, num_agents, 64)
        aggregated = agent_features.reshape(batch_size, -1) # (batch_size, num_agents * 64)
        final_features = self.aggregator(aggregated)
        return final_features

# MultiAgentSelfAttentionExtractor and MultiAgentMeanPoolExtractor remain unchanged
# They also expect Box(num_agents, agent_obs_dim)
class MultiAgentSelfAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 2:
             raise ValueError(f"MultiAgentSelfAttentionExtractor expects Box obs space of shape (num_agents, agent_obs_dim), got {observation_space}")
        self.num_agents, self.agent_obs_dim = observation_space.shape
        super().__init__(observation_space, features_dim)
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
        )
        self.attention_dim = 64; self.num_heads = 4; self.head_dim = self.attention_dim // self.num_heads
        self.query_proj = nn.Linear(64, self.attention_dim); self.key_proj = nn.Linear(64, self.attention_dim)
        self.value_proj = nn.Linear(64, self.attention_dim); self.output_proj = nn.Linear(self.attention_dim, 64)
        self.aggregator = nn.Sequential(
            nn.Linear(64 * self.num_agents, features_dim), nn.ReLU(), nn.LayerNorm(features_dim),
        )
        self._features_dim = features_dim
        print(f"[MultiAgentSelfAttentionExtractor] Initialized for {self.num_agents} agents")
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_flat = observations.view(-1, self.agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        queries = self.query_proj(agent_features); keys = self.key_proj(agent_features); values = self.value_proj(agent_features)
        queries = queries.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, self.num_agents, self.attention_dim)
        attended_features = self.output_proj(attended_values)
        aggregated = attended_features.view(batch_size, -1)
        final_features = self.aggregator(aggregated)
        return final_features

class MultiAgentMeanPoolExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 2:
             raise ValueError(f"MultiAgentMeanPoolExtractor expects Box obs space of shape (num_agents, agent_obs_dim), got {observation_space}")
        self.num_agents, self.agent_obs_dim = observation_space.shape
        super().__init__(observation_space, features_dim)
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_obs_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
        )
        pooled_dim = 64 * 2; individual_dim = 64 * self.num_agents; global_dim = pooled_dim + individual_dim
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim, features_dim), nn.ReLU(), nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim), nn.ReLU(), nn.LayerNorm(features_dim),
        )
        self._features_dim = features_dim
        print(f"[MultiAgentMeanPoolExtractor] Initialized for {self.num_agents} agents")
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_flat = observations.view(-1, self.agent_obs_dim)
        agent_features = self.agent_encoder(obs_flat)
        agent_features = agent_features.view(batch_size, self.num_agents, -1)
        mean_features = torch.mean(agent_features, dim=1); max_features = torch.max(agent_features, dim=1)[0]
        pooled_features = torch.cat([mean_features, max_features], dim=1)
        individual_features = agent_features.view(batch_size, -1)
        combined_features = torch.cat([pooled_features, individual_features], dim=1)
        final_features = self.global_processor(combined_features)
        return final_features

def create_multiagent_model(
    env, 
    algorithm: str = "td3",
    extractor_type: str = "matrix", # This might be "kin_depth" now too
    features_dim: int = 256, 
    **kwargs
):
    """
    Create a model (PPO, TD3, SAC, or DDPG) with custom feature extractor.
    Handles Dict observation space for KIN_DEPTH.
    """
    
    policy_kwargs = {}
    chosen_extractor_class = None

    if isinstance(env.observation_space, gym.spaces.Dict):
        # This is likely KIN_DEPTH observation space
        if "kin" in env.observation_space.spaces and "depth" in env.observation_space.spaces:
            chosen_extractor_class = KinDepthExtractor
            print(f"[create_multiagent_model] Using KinDepthExtractor for Dict observation space.")
            # For Dict space, SB3 policy needs to know the extractor.
            policy_kwargs["features_extractor_class"] = chosen_extractor_class
            policy_kwargs["features_extractor_kwargs"] = {"features_dim": features_dim}
        else:
            raise ValueError("Dict observation space missing 'kin' or 'depth' keys.")
    
    elif isinstance(env.observation_space, gym.spaces.Box):
        # This is for KIN or RGB (if flat Box) observation types
        # Uses the original multi-agent extractors
        extractors = {
            "matrix": MultiAgentMatrixExtractor,
            "attention": MultiAgentSelfAttentionExtractor,
            "meanpool": MultiAgentMeanPoolExtractor
        }
        if extractor_type not in extractors:
            raise ValueError(f"Unknown Box extractor type: {extractor_type}")
        chosen_extractor_class = extractors[extractor_type]
        print(f"[create_multiagent_model] Using {extractor_type} for Box observation space.")
        policy_kwargs["features_extractor_class"] = chosen_extractor_class
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": features_dim}
    else:
        raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")


    # Algorithm-specific configurations
    if algorithm.lower() == "td3":
        # TD3 uses separate networks for actor and critic, features_extractor is shared by default.
        # If using Dict obs, CombinedExtractor (auto by SB3 if policy_kwargs has feat_ext) or our custom one
        # needs to be compatible.
        # MlpPolicy for TD3 does not directly take net_arch for pi/qf like PPO's ActorCriticPolicy.
        # Instead, it defines its structure. For custom features, the features_extractor handles it.
        # The default MlpPolicy of TD3 might not have `share_features_extractor`
        # Let's ensure net_arch is set correctly if policy_kwargs are used.
        
        # Default TD3 net_arch if not specified in kwargs, used by MlpPolicy internally.
        # This will be used *after* the feature_dim output from the extractor.
        net_arch_pi = kwargs.pop("net_arch_pi", [256, 256]) # Actor network
        net_arch_qf = kwargs.pop("net_arch_qf", [256, 256]) # Critic network
        
        policy_kwargs.update({
            # "net_arch": is not directly taken by MlpPolicy constructor in the same way as ActorCriticPolicy.
            # TD3 MlpPolicy defines actor/critic internally.
            # The features_extractor's output (features_dim) becomes input to these internal MLPs.
            # If you need to customize actor/critic structure *after* extractor, you might need custom TD3Policy.
            # For now, let's assume the default MlpPolicy structure post-extraction is fine.
            # If using Dict observations, SB3 will typically wrap this in a CombinedExtractor if you specify features_extractor_class.
        })
        
        default_kwargs = {
            "learning_rate": 1e-4, "buffer_size": 1_000_000, "learning_starts": 10000,
            "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
            "gradient_steps": 1, "action_noise": None, "policy_delay": 2,
            "target_policy_noise": 0.1, "target_noise_clip": 0.5,
        }
        default_kwargs.update(kwargs)
        
        model = TD3("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **default_kwargs)
        
    elif algorithm.lower() == "ddpg":
        # DDPG is simpler than TD3 - single critic, no policy delay, no target noise
        # Similar structure to TD3 but fewer hyperparameters
        
        policy_kwargs.update({
            # DDPG MlpPolicy also defines actor/critic internally
            # The features_extractor's output becomes input to these networks
        })
        
        default_kwargs = {
            "learning_rate": 1e-4, "buffer_size": 1_000_000, "learning_starts": 10000,
            "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
            "gradient_steps": 1, "action_noise": None,
        }
        default_kwargs.update(kwargs)
        
        model = DDPG("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **default_kwargs)
        
    elif algorithm.lower() == "sac":
        # SAC also has MlpPolicy.
        policy_kwargs.update({
            # Similar to TD3, net_arch applies after extractor.
        })
        default_kwargs = {
            "learning_rate": 3e-4, "buffer_size": 1_000_000, "learning_starts": 10000,
            "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1,
            "gradient_steps": 1, "ent_coef": "auto", "target_entropy": "auto",
        }
        default_kwargs.update(kwargs)
        model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **default_kwargs)
        
    elif algorithm.lower() == "ppo":
        # PPO uses ActorCriticPolicy, which can take net_arch directly.
        # The features_extractor output goes into this net_arch.
        # net_arch_config = kwargs.pop("net_arch", [dict(pi=[64, 64], vf=[64, 64])]) # PPO net_arch format
        # policy_kwargs.update({
        #     "net_arch":  ,
        # })
        
        default_kwargs = {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        }
        default_kwargs.update(kwargs)
        #policy_kwargs=policy_kwargs MultiInputPolicy
        model = PPO("MlpPolicy", env, device='cpu', **default_kwargs) #, device='mps') # $MultiInputPolicy policy_kwargs=policy_kwargs
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"[create_multiagent_model] Created {algorithm.upper()} with extractor: {chosen_extractor_class.__name__ if chosen_extractor_class else 'Default SB3 Extractor'}")
    print(f"[create_multiagent_model] Features dimension for policy input: {features_dim}")
    
    return model

# Example usage and testing (can be removed or adapted)
if __name__ == "__main__":
    print("Testing KinDepthExtractor...")
    # Create a dummy Dict observation space for a single agent
    dummy_kin_dim = 12 + (5 * 4) # 12 base kin + 5 buffer * 4 action_dim
    dummy_img_h, dummy_img_w = 48, 64 
    
    # For NUM_DRONES = 1
    obs_space_single_drone = gym.spaces.Dict({
        "kin": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dummy_kin_dim,), dtype=np.float32),
        "depth": gym.spaces.Box(low=0, high=1, shape=(dummy_img_h, dummy_img_w, 1), dtype=np.float32)
    })

    extractor_single = KinDepthExtractor(obs_space_single_drone, features_dim=64)
    
    dummy_obs_single = {
        "kin": torch.randn(32, dummy_kin_dim), # Batch of 32
        "depth": torch.randn(32, dummy_img_h, dummy_img_w, 1)
    }
    output_single = extractor_single(dummy_obs_single)
    print(f"  KinDepthExtractor (single drone) Input: kin {dummy_obs_single['kin'].shape}, depth {dummy_obs_single['depth'].shape}")
    print(f"  KinDepthExtractor (single drone) Output shape: {output_single.shape}")
    #assert output_single.shape == (32, 256), f"Expected (32, 256), got {output_single.shape}"

    print("\nTesting MultiAgentMatrixExtractor (as an example of Box extractor)...")
    # For NUM_DRONES > 1, Box space
    num_test_agents = 2
    agent_obs_feature_dim = 30 # Example per-agent feature dim
    obs_space_multi_box = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(num_test_agents, agent_obs_feature_dim), dtype=np.float32
    )
    extractor_multi_matrix = MultiAgentMatrixExtractor(obs_space_multi_box, features_dim=128)
    dummy_obs_multi_box = torch.randn(16, num_test_agents, agent_obs_feature_dim) # Batch of 16
    output_multi_matrix = extractor_multi_matrix(dummy_obs_multi_box)
    print(f"  MultiAgentMatrixExtractor Input: {dummy_obs_multi_box.shape}")
    print(f"  MultiAgentMatrixExtractor Output shape: {output_multi_matrix.shape}")
    assert output_multi_matrix.shape == (16, 128)

    print(f"\n{'='*60}")
    print("Extractor tests completed.")