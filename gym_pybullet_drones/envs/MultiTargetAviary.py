# # MultiTargetAviary.py - UPDATED for ActionType.RPM (4 actions per drone)
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
# from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# class MultiTargetAviary(BaseRLAviary):
#     """
#     Multi-agent RL environment where drones must visit a sequence of targets
#     while maintaining formation and avoiding collisions.
    
#     UPDATED to work with ActionType.RPM (4 propeller controls per drone):
#     - Uses standard 12 kinematic features + action buffer from BaseRLAviary
#     - Action buffer now contains 4 RPM values per drone per timestep
#     - Adds target information by overriding observation methods
#     - Implements all required abstract methods
#     """

#     def __init__(
#         self,
#         drone_model: DroneModel = DroneModel.CF2X,
#         num_drones: int = 4,
#         neighbourhood_radius: float = np.inf,
#         initial_xyzs=None,
#         initial_rpys=None,
#         physics: Physics = Physics.PYB,
#         pyb_freq: int = 240,
#         ctrl_freq: int = 30,
#         gui=False,
#         record=False,
#         obs: ObservationType = ObservationType.KIN,
#         act: ActionType = ActionType.RPM,  # CHANGED: Now uses RPM (4 values per drone)
#         target_sequence: np.ndarray = None,
#         steps_per_target: int = 100,
#         tolerance: float = 0.15,
#         collision_distance: float = 0.3,
#     ):
#         # Create default target sequence if none provided
#         if target_sequence is None:
#             target_sequence = self._create_default_targets(num_drones)
        
#         self.target_sequence = target_sequence
#         self.steps_per_target = steps_per_target
#         self.tolerance = tolerance
#         self.collision_distance = collision_distance
        
#         self.gui = gui
#         self.record = record
#         self.OBS_TYPE = obs
#         self.ACT_TYPE = act
#         self.NUM_DRONES = num_drones
#         self.DRONE_MODEL = drone_model
        
#         # Episode state
#         self.current_phase = 0
#         self.step_in_phase = 0
#         self.total_steps = 0
#         self.max_episode_steps = len(target_sequence) * steps_per_target
        
#         # Performance tracking
#         self.targets_reached = np.zeros(num_drones, dtype=bool)
#         self.collision_count = 0
        
#         # Set episode length in seconds
#         self.EPISODE_LEN_SEC = (len(target_sequence) * steps_per_target) / ctrl_freq
        
#         super().__init__(
#             drone_model=drone_model,
#             num_drones=num_drones,
#             neighbourhood_radius=neighbourhood_radius,
#             initial_xyzs=initial_xyzs,
#             initial_rpys=initial_rpys,
#             physics=physics,
#             pyb_freq=pyb_freq,
#             ctrl_freq=ctrl_freq,
#             gui=gui,
#             record=record,
#             obs=obs,
#             act=act
#         )
        
#         # Calculate actual observation dimensions after BaseRLAviary initialization
#         sample_obs = super()._computeObs()
#         self.base_obs_dim = sample_obs.shape[1]  # Features per drone from BaseRLAviary
        
#         print(f"[MultiTargetAviary] Initialized with {len(target_sequence)} target phases")
#         print(f"[MultiTargetAviary] Targets shape: {target_sequence.shape}")
#         print(f"[MultiTargetAviary] Episode length: {self.EPISODE_LEN_SEC:.1f} seconds")
#         print(f"[MultiTargetAviary] Action type: {act} (4 RPM values per drone)")
#         print(f"[MultiTargetAviary] Base observation dim: {self.base_obs_dim} features per drone")
#         print(f"[MultiTargetAviary] Action buffer size: {self.ACTION_BUFFER_SIZE}")

#     def _create_default_targets(self, num_drones):
#         """Create a default sequence of targets"""
#         if num_drones == 4:
#             targets = np.array([
#                 # Phase 0: Square formation
#                 [[ 1.0,  1.0, 1.5], [-1.0,  1.0, 1.5], [-1.0, -1.0, 1.5], [ 1.0, -1.0, 1.5]],
#                 # Phase 1: Rotate positions
#                 [[-1.0,  1.0, 1.5], [-1.0, -1.0, 1.5], [ 1.0, -1.0, 1.5], [ 1.0,  1.0, 1.5]],
#                 # Phase 2: Diamond formation
#                 [[ 0.0,  1.5, 1.8], [-1.5,  0.0, 1.8], [ 0.0, -1.5, 1.8], [ 1.5,  0.0, 1.8]],
#                 # Phase 3: Center formation
#                 [[ 0.5,  0.5, 1.5], [-0.5,  0.5, 1.5], [-0.5, -0.5, 1.5], [ 0.5, -0.5, 1.5]]
#             ])
#         else:
#             # Create circular formations for other numbers of drones
#             targets = []
#             n_phases = 4
#             for phase in range(n_phases):
#                 phase_targets = []
#                 radius = 1.0 + 0.3 * phase
#                 height = 1.5 + 0.2 * phase
#                 for i in range(num_drones):
#                     angle = 2 * np.pi * i / num_drones + phase * np.pi / 4
#                     x = radius * np.cos(angle)
#                     y = radius * np.sin(angle)
#                     phase_targets.append([x, y, height])
#                 targets.append(phase_targets)
#             targets = np.array(targets)
        
#         return targets.astype(np.float32)

#     def _observationSpace(self):
#         """
#         Override BaseRLAviary observation space to add target information.
        
#         BaseRLAviary provides (for ActionType.RPM):
#         - 12 kinematic features per drone: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
#         - Action buffer: ACTION_BUFFER_SIZE * 4 RPM values (since RPM actions have 4 values)
        
#         We add:
#         - 3 target position features per drone
#         - 3 relative target position features per drone  
#         - 1 distance to target per drone
#         - 1 phase progress per drone
#         """
#         if self.OBS_TYPE == ObservationType.RGB:
#             # For RGB, use parent implementation
#             return super()._observationSpace()
            
#         elif self.OBS_TYPE == ObservationType.KIN:
#             # Get base observation space from parent
#             base_obs_space = super()._observationSpace()
#             base_shape = base_obs_space.shape
            
#             # Add target information: 8 features per drone
#             target_features_per_drone = 8
            
#             # New observation shape: base_features + target_features per drone
#             new_shape = (base_shape[0], base_shape[1] + target_features_per_drone)
            
#             # Create bounds for additional features
#             lo = -np.inf
#             hi = np.inf
            
#             # Extend bounds for target features
#             base_low = base_obs_space.low
#             base_high = base_obs_space.high
            
#             target_low = np.full((self.NUM_DRONES, target_features_per_drone), lo, dtype=np.float32)
#             target_high = np.full((self.NUM_DRONES, target_features_per_drone), hi, dtype=np.float32)
            
#             # Set specific bounds for some features
#             target_low[:, 6] = 0.0  # Distance is always positive
#             target_high[:, 6] = 10.0  # Reasonable max distance
#             target_low[:, 7] = 0.0  # Progress is 0-1
#             target_high[:, 7] = 1.0
            
#             new_low = np.hstack([base_low, target_low])
#             new_high = np.hstack([base_high, target_high])
            
#             return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
#         else:
#             raise NotImplementedError(f"Observation type {self.OBS_TYPE} not implemented")

#     def _computeObs(self):
#         """
#         Override BaseRLAviary observation computation to add target information.
        
#         For ActionType.RPM, BaseRLAviary provides:
#         - 12 kinematic features + (ACTION_BUFFER_SIZE * 4) action history features
        
#         We add 8 target-related features per drone.
#         """
#         if self.OBS_TYPE == ObservationType.RGB:
#             # For RGB, use parent implementation
#             return super()._computeObs()
            
#         elif self.OBS_TYPE == ObservationType.KIN:
#             # Get base observations from parent (includes 12 kinematic + 4*ACTION_BUFFER_SIZE action buffer)
#             base_obs = super()._computeObs()  # Shape: (NUM_DRONES, base_features)
            
#             # Compute target information
#             current_targets = self.get_current_targets()
#             phase_progress = self.current_phase / max(1, len(self.target_sequence) - 1)
            
#             target_obs = np.zeros((self.NUM_DRONES, 8), dtype=np.float32)
            
#             for i in range(self.NUM_DRONES):
#                 # Get current position from base observation (first 3 features are x, y, z)
#                 my_position = base_obs[i, 0:3]
#                 my_target = current_targets[i]
#                 relative_target = my_target - my_position
#                 target_distance = np.linalg.norm(relative_target)
                
#                 target_obs[i, :] = np.array([
#                     my_target[0], my_target[1], my_target[2],      # Target position (3)
#                     relative_target[0], relative_target[1], relative_target[2],  # Relative target (3)
#                     target_distance,                               # Distance (1)
#                     phase_progress                                 # Phase progress (1)
#                 ])
            
#             # Concatenate base observations with target observations
#             full_obs = np.hstack([base_obs, target_obs])
            
#             return full_obs
#         else:
#             raise NotImplementedError(f"Observation type {self.OBS_TYPE} not implemented")

#     def reset(self, seed=None, options=None):
#         """Reset environment to initial state"""
#         self.current_phase = 0
#         self.step_in_phase = 0
#         self.total_steps = 0
#         self.targets_reached = np.zeros(self.NUM_DRONES, dtype=bool)
#         self.collision_count = 0
        
#         obs, info = super().reset(seed=seed, options=options)
        
#         # Add target information to info
#         info.update({
#             'current_targets': self.get_current_targets(),
#             'phase': self.current_phase,
#             'step_in_phase': self.step_in_phase
#         })
        
#         return obs, info

#     def get_current_targets(self):
#         """Get current target positions for all drones"""
#         if self.current_phase < len(self.target_sequence):
#             return self.target_sequence[self.current_phase]
#         else:
#             return self.target_sequence[-1]

#     def step(self, action):
#         """Execute one simulation step"""
#         # Take physics step using parent class
#         obs, _, terminated, truncated, info = super().step(action)
        
#         # Get current drone positions (from first 3 features of each drone's observation)
#         positions = obs[:, 0:3]  # Shape: (NUM_DRONES, 3)
#         current_targets = self.get_current_targets()
        
#         # Compute reward
#         reward = self._compute_swarm_reward(positions, current_targets)
        
#         # Update episode state
#         self.step_in_phase += 1
#         self.total_steps += 1
        
#         # Check if we should advance to next phase
#         if self.step_in_phase >= self.steps_per_target:
#             self.current_phase += 1
#             self.step_in_phase = 0
#             self.targets_reached.fill(False)
#             if self.gui:
#                 print(f"Advanced to phase {self.current_phase}")

#         # Check termination conditions
#         done = self._computeTerminated()
#         truncated, unnatural = self._computeTruncated()
#         if truncated and unnatural:
#             #print(f"Truncated due to unnatural conditions at step {self.total_steps}")
#             reward -= 20.0  # Penalize for unnatural truncation
        
#         # Update info with additional metrics
#         distances_to_targets = np.linalg.norm(positions - current_targets, axis=1)
        
#         info.update({
#             'current_targets': current_targets,
#             'phase': self.current_phase,
#             'step_in_phase': self.step_in_phase,
#             'targets_reached': self.targets_reached.copy(),
#             'collision_count': self.collision_count,
#             'distance_to_targets': distances_to_targets,
#             'mean_distance_to_targets': np.mean(distances_to_targets),
#             'formation_error': np.std(distances_to_targets),
#         })
        
#         return obs, reward, done, truncated, info

#     def _compute_swarm_reward(self, positions, targets):
#         """Compute reward based on target proximity and formation keeping"""
#         reward = 0.0
        
#         # 1. Distance-to-target reward (primary objective)
#         distances = np.linalg.norm(positions - targets, axis=1)
#         #print(targets)
#         # print(f"Positions:\n{positions}")
#         # print(f"Targets:\n{targets}")
#         # print("------")
#         # distances = np.linalg.norm(positions[:, 2] - targets[:, 2], axis=1)
#         ret = np.sum(-distances)#np.sum(np.exp(-2.0 * distances))#max(0, 2 - np.linalg.norm(targets[:, 2]-positions[:, 2]))#.sum()
#         #ret = max(0, 2 - np.linalg.norm(targets[:, 2]-positions[:, 2]))
#         reward += ret
        
#         # # Exponential reward for being close to targets
#         # target_rewards = np.exp(-2.0 * distances)
#         # reward += np.sum(target_rewards)
        
#         # Bonus for reaching targets
#         newly_reached = (distances < self.tolerance) & (~self.targets_reached)
#         if np.any(newly_reached):
#             reward += np.sum(newly_reached) * 10.0
#             self.targets_reached |= newly_reached
        
#         # # 2. Collision avoidance penalty
#         # collision_penalty = 0.0
#         # for i in range(self.NUM_DRONES):
#         #     for j in range(i + 1, self.NUM_DRONES):
#         #         distance = np.linalg.norm(positions[i] - positions[j])
#         #         if distance < self.collision_distance:
#         #             collision_penalty -= 2.0 * np.exp(-(distance / 0.1))
#         #             self.collision_count += 1
        
#         # reward += collision_penalty
        
#         # 3. Phase completion bonus
#         if np.all(self.targets_reached):
#             reward += 20.0
        
#         return reward

#     # =====================================================================
#     # REQUIRED ABSTRACT METHODS FROM BaseRLAviary/BaseAviary
#     # =====================================================================

#     def _computeReward(self):
#         """Computes the current reward value."""
#         positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
#         current_targets = self.get_current_targets()
#         return self._compute_swarm_reward(positions, current_targets)

#     def _computeTerminated(self):
#         """Computes the current terminated value."""
#         # Episode ends when we've completed all phases
#         if self.current_phase >= len(self.target_sequence):
#             return True
            
#         # Episode ends if all drones reach targets in final phase
#         if self.current_phase == len(self.target_sequence) - 1:
#             positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
#             current_targets = self.get_current_targets()
#             distances = np.linalg.norm(positions - current_targets, axis=1)
#             if np.all(distances < self.tolerance):
#                 return True
                
#         return False

#     def _computeTruncated(self):
#         """Computes the current truncated value."""
#         # Episode truncated if we've exceeded maximum steps
#         if self.total_steps >= self.max_episode_steps:
#             return True, False
            
#         # Episode truncated if any drone goes too far out of bounds
#         for i in range(self.NUM_DRONES):
#             state = self._getDroneStateVector(i)
#             #print(f"Drone {i} state: {state}")
#             position = state[0:3]
            
#             # Check position bounds
#             if (abs(position[0]) > 5.0 or abs(position[1]) > 5.0 or 
#                 position[2] > 4.0 or position[2] < 0.05):
#                 return True, True
                
#             # # Check orientation bounds (too tilted)
#             # if abs(state[7]) > 0.8 or abs(state[8]) > 0.8:  # ~45 degrees
#             #     return True
                
#         return False, False

#     def _computeInfo(self):
#         """Computes the current info dict(s)."""
#         positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
#         current_targets = self.get_current_targets()
#         distances_to_targets = np.linalg.norm(positions - current_targets, axis=1)
        
#         return {
#             'current_targets': current_targets,
#             'phase': self.current_phase,
#             'step_in_phase': self.step_in_phase,
#             'targets_reached': self.targets_reached.copy(),
#             'collision_count': self.collision_count,
#             'distance_to_targets': distances_to_targets,
#             'mean_distance_to_targets': np.mean(distances_to_targets),
#             'formation_error': np.std(distances_to_targets),
#             'episode_progress': self.total_steps / self.max_episode_steps
#         }


# # Test the environment with RPM actions
# if __name__ == "__main__":
#     print("Testing MultiTargetAviary with ActionType.RPM...")
    
#     env = MultiTargetAviary(
#         num_drones=4,
#         obs=ObservationType.KIN,
#         act=ActionType.RPM,  # Using RPM actions (4 values per drone)
#         gui=False
#     )
    
#     print(f"Action space: {env.action_space}")
#     print(f"Action space shape: {env.action_space.shape}")  # Should be (4, 4)
#     print(f"Observation space: {env.observation_space}")
#     print(f"Observation space shape: {env.observation_space.shape}")
    
#     obs, info = env.reset()
#     print(f"Observation shape: {obs.shape}")
    
#     # Test action space
#     action = env.action_space.sample()
#     print(f"Sample action shape: {action.shape}")  # Should be (4, 4)
#     print(f"Sample action:\n{action}")
    
#     # Test a few steps
#     for i in range(3):
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         print(f"\nStep {i+1}:")
#         print(f"  Reward: {reward:.3f}")
#         print(f"  Action shape: {action.shape}")
#         print(f"  Mean distance to targets: {info['mean_distance_to_targets']:.3f}")
        
#         if done or truncated:
#             break
    
#     env.close()
#     print("\nRPM action test completed successfully!")



# MultiTargetAviary.py - UPDATED for ActionType.RPM (4 actions per drone) with DELTA DISTANCE REWARDS
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiTargetAviary(BaseRLAviary):
    """
    Multi-agent RL environment where drones must visit a sequence of targets
    while maintaining formation and avoiding collisions.
    
    UPDATED to work with ActionType.RPM (4 propeller controls per drone):
    - Uses standard 12 kinematic features + action buffer from BaseRLAviary
    - Action buffer now contains 4 RPM values per drone per timestep
    - Adds target information by overriding observation methods
    - Implements all required abstract methods
    - USES DELTA DISTANCE REWARDS: reward based on progress toward targets
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 4,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,  # CHANGED: Now uses RPM (4 values per drone)
        target_sequence: np.ndarray = None,
        steps_per_target: int = 100,
        tolerance: float = 0.15,
        collision_distance: float = 0.3,
        progress_scale: float = 100.0,  # NEW: Scale factor for progress rewards
    ):
        # Create default target sequence if none provided
        if target_sequence is None:
            target_sequence = self._create_default_targets(num_drones)
        
        self.target_sequence = target_sequence
        self.steps_per_target = steps_per_target
        self.tolerance = tolerance
        self.collision_distance = collision_distance
        self.progress_scale = progress_scale  # NEW: Scale for delta rewards
        
        self.gui = gui
        self.record = record
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.NUM_DRONES = num_drones
        self.DRONE_MODEL = drone_model
        
        # Episode state
        self.current_phase = 0
        self.step_in_phase = 0
        self.total_steps = 0
        self.max_episode_steps = len(target_sequence) * steps_per_target
        
        # Performance tracking
        self.targets_reached = np.zeros(num_drones, dtype=bool)
        self.collision_count = 0
        
        # NEW: Track previous distances for delta rewards
        self.previous_distances = np.zeros(num_drones, dtype=np.float32)
        self.first_step = True  # Flag to handle first step (no previous distance)
        
        # Set episode length in seconds
        self.EPISODE_LEN_SEC = (len(target_sequence) * steps_per_target) / ctrl_freq
        
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )
        
        # Calculate actual observation dimensions after BaseRLAviary initialization
        sample_obs = super()._computeObs()
        self.base_obs_dim = sample_obs.shape[1]  # Features per drone from BaseRLAviary
        
        print(f"[MultiTargetAviary] Initialized with {len(target_sequence)} target phases")
        print(f"[MultiTargetAviary] Targets shape: {target_sequence.shape}")
        print(f"[MultiTargetAviary] Episode length: {self.EPISODE_LEN_SEC:.1f} seconds")
        print(f"[MultiTargetAviary] Action type: {act} (4 RPM values per drone)")
        print(f"[MultiTargetAviary] Base observation dim: {self.base_obs_dim} features per drone")
        print(f"[MultiTargetAviary] Action buffer size: {self.ACTION_BUFFER_SIZE}")
        print(f"[MultiTargetAviary] Using DELTA distance rewards with scale: {progress_scale}")

    def _create_default_targets(self, num_drones):
        """Create a default sequence of targets"""
        if num_drones == 4:
            targets = np.array([
                # Phase 0: Square formation
                [[ 1.0,  1.0, 1.5], [-1.0,  1.0, 1.5], [-1.0, -1.0, 1.5], [ 1.0, -1.0, 1.5]],
                # Phase 1: Rotate positions
                [[-1.0,  1.0, 1.5], [-1.0, -1.0, 1.5], [ 1.0, -1.0, 1.5], [ 1.0,  1.0, 1.5]],
                # Phase 2: Diamond formation
                [[ 0.0,  1.5, 1.8], [-1.5,  0.0, 1.8], [ 0.0, -1.5, 1.8], [ 1.5,  0.0, 1.8]],
                # Phase 3: Center formation
                [[ 0.5,  0.5, 1.5], [-0.5,  0.5, 1.5], [-0.5, -0.5, 1.5], [ 0.5, -0.5, 1.5]]
            ])
        else:
            # Create circular formations for other numbers of drones
            targets = []
            n_phases = 4
            for phase in range(n_phases):
                phase_targets = []
                radius = 1.0 + 0.3 * phase
                height = 1.5 + 0.2 * phase
                for i in range(num_drones):
                    angle = 2 * np.pi * i / num_drones + phase * np.pi / 4
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    phase_targets.append([x, y, height])
                targets.append(phase_targets)
            targets = np.array(targets)
        
        return targets.astype(np.float32)

    def _observationSpace(self):
        """
        Override BaseRLAviary observation space to add target information.
        
        BaseRLAviary provides (for ActionType.RPM):
        - 12 kinematic features per drone: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        - Action buffer: ACTION_BUFFER_SIZE * 4 RPM values (since RPM actions have 4 values)
        
        We add:
        - 3 target position features per drone
        - 3 relative target position features per drone  
        - 1 distance to target per drone
        - 1 phase progress per drone
        """
        if self.OBS_TYPE == ObservationType.RGB:
            # For RGB, use parent implementation
            return super()._observationSpace()
            
        elif self.OBS_TYPE == ObservationType.KIN:
            # Get base observation space from parent
            base_obs_space = super()._observationSpace()
            base_shape = base_obs_space.shape
            
            # Add target information: 8 features per drone
            target_features_per_drone = 8
            
            # New observation shape: base_features + target_features per drone
            new_shape = (base_shape[0], base_shape[1] + target_features_per_drone)
            
            # Create bounds for additional features
            lo = -np.inf
            hi = np.inf
            
            # Extend bounds for target features
            base_low = base_obs_space.low
            base_high = base_obs_space.high
            
            target_low = np.full((self.NUM_DRONES, target_features_per_drone), lo, dtype=np.float32)
            target_high = np.full((self.NUM_DRONES, target_features_per_drone), hi, dtype=np.float32)
            
            # # Set specific bounds for some features
            # target_low[:, 6] = 0.0  # Distance is always positive
            # target_high[:, 6] = 10.0  # Reasonable max distance
            # target_low[:, 7] = 0.0  # Progress is 0-1
            # target_high[:, 7] = 1.0
            
            new_low = np.hstack([base_low, target_low])
            new_high = np.hstack([base_high, target_high])
            
            return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
        else:
            raise NotImplementedError(f"Observation type {self.OBS_TYPE} not implemented")

    def _computeObs(self):
        """
        Override BaseRLAviary observation computation to add target information.
        
        For ActionType.RPM, BaseRLAviary provides:
        - 12 kinematic features + (ACTION_BUFFER_SIZE * 4) action history features
        
        We add 8 target-related features per drone.
        """
        if self.OBS_TYPE == ObservationType.RGB:
            # For RGB, use parent implementation
            return super()._computeObs()
            
        elif self.OBS_TYPE == ObservationType.KIN:
            # Get base observations from parent (includes 12 kinematic + 4*ACTION_BUFFER_SIZE action buffer)
            base_obs = super()._computeObs()  # Shape: (NUM_DRONES, base_features)
            
            # Compute target information
            current_targets = self.get_current_targets()
            phase_progress = self.current_phase / max(1, len(self.target_sequence) - 1)
            
            target_obs = np.zeros((self.NUM_DRONES, 8), dtype=np.float32)
            
            for i in range(self.NUM_DRONES):
                # Get current position from base observation (first 3 features are x, y, z)
                my_position = base_obs[i, 0:3]
                my_target = current_targets[i]
                relative_target = my_target - my_position
                target_distance = np.linalg.norm(relative_target)
                
                target_obs[i, :] = np.array([
                    my_target[0], my_target[1], my_target[2],      # Target position (3)
                    relative_target[0], relative_target[1], relative_target[2],  # Relative target (3)
                    target_distance,                               # Distance (1)
                    phase_progress                                 # Phase progress (1)
                ])
            
            # Concatenate base observations with target observations
            full_obs = np.hstack([base_obs, target_obs])
            
            return full_obs
        else:
            raise NotImplementedError(f"Observation type {self.OBS_TYPE} not implemented")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        self.current_phase = 0
        self.step_in_phase = 0
        self.total_steps = 0
        self.targets_reached = np.zeros(self.NUM_DRONES, dtype=bool)
        self.collision_count = 0
        
        # NEW: Reset distance tracking for delta rewards
        self.previous_distances = np.zeros(self.NUM_DRONES, dtype=np.float32)
        self.first_step = True
        
        obs, info = super().reset(seed=seed, options=options)
        
        # NEW: Initialize previous distances with current distances
        positions = obs[:, 0:3]  # Get positions from observation
        current_targets = self.get_current_targets()
        self.previous_distances = np.linalg.norm(positions - current_targets, axis=1)
        
        # Add target information to info
        info.update({
            'current_targets': self.get_current_targets(),
            'phase': self.current_phase,
            'step_in_phase': self.step_in_phase
        })
        
        return obs, info

    def get_current_targets(self):
        """Get current target positions for all drones"""
        if self.current_phase < len(self.target_sequence):
            return self.target_sequence[self.current_phase]
        else:
            return self.target_sequence[-1]

    def step(self, action):
        """Execute one simulation step"""
        # Take physics step using parent class
        obs, _, terminated, truncated, info = super().step(action)
        
        # Get current drone positions (from first 3 features of each drone's observation)
        positions = obs[:, 0:3]  # Shape: (NUM_DRONES, 3)
        current_targets = self.get_current_targets()
        
        # Compute reward using delta distances
        reward = self._compute_swarm_reward(positions, current_targets)
        
        # Update episode state
        self.step_in_phase += 1
        self.total_steps += 1
        
        # Check if we should advance to next phase
        if self.step_in_phase >= self.steps_per_target:
            self.current_phase += 1
            self.step_in_phase = 0
            self.targets_reached.fill(False)
            # NEW: Reset previous distances when switching phases
            current_distances = np.linalg.norm(positions - self.get_current_targets(), axis=1)
            self.previous_distances = current_distances.copy()
            if self.gui:
                print(f"Advanced to phase {self.current_phase}")

        # Check termination conditions
        done = self._computeTerminated()
        truncated, unnatural = self._computeTruncated()
        if truncated and unnatural:
            #print(f"Truncated due to unnatural conditions at step {self.total_steps}")
            reward -= 20.0  # Penalize for unnatural truncation
        
        # Update info with additional metrics
        distances_to_targets = np.linalg.norm(positions - current_targets, axis=1)
        
        info.update({
            'current_targets': current_targets,
            'phase': self.current_phase,
            'step_in_phase': self.step_in_phase,
            'targets_reached': self.targets_reached.copy(),
            'collision_count': self.collision_count,
            'distance_to_targets': distances_to_targets,
            'mean_distance_to_targets': np.mean(distances_to_targets),
            'formation_error': np.std(distances_to_targets),
        })
        
        return obs, reward, done, truncated, info

    def _compute_swarm_reward(self, positions, targets):
        """
        Compute reward based on DELTA distances (progress toward targets).
        Reward = progress_scale * (previous_distance - current_distance)
        Positive reward for moving closer, negative for moving farther.
        """
        reward = 0.0
        
        # Calculate current distances
        current_distances = np.linalg.norm(positions - targets, axis=1)
        
        if self.first_step:
            # On first step, no delta reward (no previous distance to compare)
            # Could give small reward based on current distance to encourage initial movement
            reward += np.sum(np.exp(-0.5 * current_distances))  # Small initial reward
            self.first_step = False
        else:
            # MAIN DELTA REWARD: reward progress toward targets
            distance_deltas = self.previous_distances - current_distances
            progress_reward = self.progress_scale * np.sum(distance_deltas)
            reward += progress_reward
            
            # Optional: Add small bonus for being very close to targets
            #close_bonus = np.sum(np.exp(-5.0 * current_distances))
            #reward += close_bonus
        
        # Update previous distances for next step
        self.previous_distances = current_distances.copy()
        
        # Bonus for reaching targets
        newly_reached = (current_distances < self.tolerance) & (~self.targets_reached)
        if np.any(newly_reached):
            reward += np.sum(newly_reached) * 100.0  # Large bonus for target completion
            self.targets_reached |= newly_reached
        
        # Optional collision avoidance (uncomment if desired)
        # collision_penalty = 0.0
        # for i in range(self.NUM_DRONES):
        #     for j in range(i + 1, self.NUM_DRONES):
        #         distance = np.linalg.norm(positions[i] - positions[j])
        #         if distance < self.collision_distance:
        #             collision_penalty -= 5.0
        #             self.collision_count += 1
        # reward += collision_penalty
        
        # Phase completion bonus
        if np.all(self.targets_reached):
            reward += 1000.0
        
        return reward

    # =====================================================================
    # REQUIRED ABSTRACT METHODS FROM BaseRLAviary/BaseAviary
    # =====================================================================

    def _computeReward(self):
        """Computes the current reward value."""
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        current_targets = self.get_current_targets()
        return self._compute_swarm_reward(positions, current_targets)

    def _computeTerminated(self):
        """Computes the current terminated value."""
        # Episode ends when we've completed all phases
        if self.current_phase >= len(self.target_sequence):
            return True
            
        # Episode ends if all drones reach targets in final phase
        if self.current_phase == len(self.target_sequence) - 1:
            positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
            current_targets = self.get_current_targets()
            distances = np.linalg.norm(positions - current_targets, axis=1)
            if np.all(distances < self.tolerance):
                return True
                
        return False

    def _computeTruncated(self):
        """Computes the current truncated value."""
        # Episode truncated if we've exceeded maximum steps
        if self.total_steps >= self.max_episode_steps:
            return True, False
            
        # Episode truncated if any drone goes too far out of bounds
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            #print(f"Drone {i} state: {state}")
            position = state[0:3]
            
            # Check position bounds
            if (#abs(position[0]) > 5.0 or abs(position[1]) > 5.0 or 
                position[2] > 4.0 or position[2] < 0.05):
                return True, True
                
            # # Check orientation bounds (too tilted)
            # if abs(state[7]) > 0.8 or abs(state[8]) > 0.8:  # ~45 degrees
            #     return True
                
        return False, False

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        current_targets = self.get_current_targets()
        distances_to_targets = np.linalg.norm(positions - current_targets, axis=1)
        
        return {
            'current_targets': current_targets,
            'phase': self.current_phase,
            'step_in_phase': self.step_in_phase,
            'targets_reached': self.targets_reached.copy(),
            'collision_count': self.collision_count,
            'distance_to_targets': distances_to_targets,
            'mean_distance_to_targets': np.mean(distances_to_targets),
            'formation_error': np.std(distances_to_targets),
            'episode_progress': self.total_steps / self.max_episode_steps
        }


# Test the environment with RPM actions and delta rewards
if __name__ == "__main__":
    print("Testing MultiTargetAviary with ActionType.RPM and DELTA distance rewards...")
    
    env = MultiTargetAviary(
        num_drones=4,
        obs=ObservationType.KIN,
        act=ActionType.RPM,  # Using RPM actions (4 values per drone)
        gui=False,
        progress_scale=100.0  # NEW: Scale factor for delta rewards
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")  # Should be (4, 4)
    print(f"Observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Test action space
    action = env.action_space.sample()
    print(f"Sample action shape: {action.shape}")  # Should be (4, 4)
    print(f"Sample action:\n{action}")
    
    # Test a few steps to see delta rewards in action
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Mean distance to targets: {info['mean_distance_to_targets']:.3f}")
        print(f"  Previous distances: {env.previous_distances}")
        
        if done or truncated:
            break
    
    env.close()
    print("\nDelta distance reward test completed successfully!")