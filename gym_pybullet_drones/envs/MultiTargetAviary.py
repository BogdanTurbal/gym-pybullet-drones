# MultiTargetAviary.py - ADAPTIVE DIFFICULTY SINGLE DRONE VERSION
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiTargetAviary(BaseRLAviary):
    """
    Adaptive difficulty single-drone RL environment.
    
    Features:
    - Adaptive target distance based on success rate
    - Simple reward: +200 for target, -200 for failure
    - Episode terminates on target reach or failure
    - Targets spawned uniformly in sphere around start position
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        episode_length_sec: float = 3.0,
        target_radius_start: float = 0.1,
        target_radius_max: float = 1.0,
        target_radius_increment: float = 0.1,
        target_tolerance: float = 0.01,
        success_threshold: float = 0.9,
        evaluation_window: int = 100,
        collision_distance: float = 0.05,
        # Soft reward parameters
        lambda_distance: float = 10.0,    # Distance improvement reward
        lambda_angle: float = 1.0,        # Target orientation reward
        # Keep compatibility params (not used)
        lambda_1: float = 0.0,    
        lambda_2: float = 0.0,     
        lambda_3: float = 0.0,    
        lambda_4: float = 0.0,    
        lambda_5: float = 0.0,    
        crash_penalty: float = 200.0,
        bounds_penalty: float = 200.0,
    ):
        
        self.episode_length_sec = episode_length_sec
        self.target_radius_start = target_radius_start
        self.target_radius_max = target_radius_max
        self.target_radius_increment = target_radius_increment
        self.target_tolerance = target_tolerance
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.collision_distance = collision_distance
        self.crash_penalty = crash_penalty
        self.bounds_penalty = bounds_penalty
        
        # Soft reward parameters
        self.lambda_distance = lambda_distance
        self.lambda_angle = lambda_angle
        
        self.gui = gui
        self.record = record
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.NUM_DRONES = num_drones
        self.DRONE_MODEL = drone_model
        
        # Adaptive difficulty system
        self.current_target_radius = target_radius_start
        self.episode_results = deque(maxlen=evaluation_window)  # Store success/failure
        self.total_episodes = 0
        
        # Episode state
        self.total_steps = 0
        self.max_episode_steps = int(episode_length_sec * ctrl_freq)
        self.current_targets = None
        self.start_positions = None
        
        # Soft reward tracking
        self.previous_distances = None
        self.first_step = True
        
        # Set episode length
        self.EPISODE_LEN_SEC = episode_length_sec
        
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=self._get_initial_positions(),
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
        
        print(f"[MultiTargetAviary] Initialized ADAPTIVE DIFFICULTY system with SOFT REWARDS")
        print(f"[MultiTargetAviary] Start radius: {target_radius_start:.2f}")
        print(f"[MultiTargetAviary] Max radius: {target_radius_max:.2f}")
        print(f"[MultiTargetAviary] Episode length: {episode_length_sec:.1f} seconds")
        print(f"[MultiTargetAviary] Success threshold: {success_threshold:.1f}")
        print(f"[MultiTargetAviary] Evaluation window: {evaluation_window} episodes")
        print(f"[MultiTargetAviary] Distance reward λ: {lambda_distance:.1f}")
        print(f"[MultiTargetAviary] Angle reward λ: {lambda_angle:.1f}")

    def _get_initial_positions(self):
        """Get initial spawn positions for drones"""
        positions = []
        for i in range(self.NUM_DRONES):
            if i == 0:
                # First drone at (1, 1, 1)
                positions.append([1.0, 1.0, 1.0])
            else:
                # Other drones offset by i * 0.25
                offset = i * 0.25
                positions.append([1.0 + offset, 1.0 + offset, 1.0])
        return np.array(positions, dtype=np.float32)

    def _generate_random_targets(self):
        """Generate random targets in sphere around start positions"""
        targets = []
        for i in range(self.NUM_DRONES):
            # Generate random point in sphere
            # Use rejection sampling for uniform distribution in sphere
            while True:
                # Generate random point in cube [-1, 1]^3
                x, y, z = np.random.uniform(-1, 1, 3)
                
                # Check if point is inside unit sphere
                if x*x + y*y + z*z <= 1.0:
                    # Scale by current radius and offset by start position
                    target = self.start_positions[i] + self.current_target_radius * np.array([x, y, z])
                    targets.append(target)
                    break
        
        return np.array(targets, dtype=np.float32)

    def _update_adaptive_difficulty(self):
        """Update target radius based on success rate"""
        if len(self.episode_results) >= self.evaluation_window:
            success_rate = np.mean(self.episode_results)
            
            if success_rate >= self.success_threshold:
                old_radius = self.current_target_radius
                self.current_target_radius = min(
                    self.target_radius_max,
                    self.current_target_radius + self.target_radius_increment
                )
                
                if self.current_target_radius > old_radius:
                    print(f"[AdaptiveDifficulty] Success rate: {success_rate:.3f} >= {self.success_threshold:.3f}")
                    print(f"[AdaptiveDifficulty] Increased radius: {old_radius:.2f} -> {self.current_target_radius:.2f}")
                    
                    # Clear episode results to re-evaluate at new difficulty
                    self.episode_results.clear()

    def _observationSpace(self):
        """Override BaseRLAviary observation space to add target information."""
        if self.OBS_TYPE == ObservationType.RGB:
            return super()._observationSpace()
            
        elif self.OBS_TYPE == ObservationType.KIN:
            # Get base observation space from parent
            base_obs_space = super()._observationSpace()
            base_shape = base_obs_space.shape
            
            # Add target information: 6 features per drone
            target_features_per_drone = 6
            
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
            
            new_low = np.hstack([base_low, target_low])
            new_high = np.hstack([base_high, target_high])
            
            return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
        else:
            raise NotImplementedError(f"Observation type {self.OBS_TYPE} not implemented")

    def _computeObs(self):
        """Override BaseRLAviary observation computation to add target information."""
        if self.OBS_TYPE == ObservationType.RGB:
            return super()._computeObs()
            
        elif self.OBS_TYPE == ObservationType.KIN:
            # Get base observations from parent
            base_obs = super()._computeObs()  # Shape: (NUM_DRONES, base_features)
            
            # Compute target information
            target_obs = np.zeros((self.NUM_DRONES, 6), dtype=np.float32)
            
            for i in range(self.NUM_DRONES):
                # Get current position from base observation
                my_position = base_obs[i, 0:3]
                my_target = self.current_targets[i]
                relative_target = my_target - my_position
                
                target_obs[i, :] = np.array([
                    my_target[0], my_target[1], my_target[2],      # Target position (3)
                    relative_target[0], relative_target[1], relative_target[2],  # Relative target (3)
                ])
            
            # Concatenate base observations with target observations
            full_obs = np.hstack([base_obs, target_obs])
            
            return full_obs
        else:
            raise NotImplementedError(f"Observation type {self.OBS_TYPE} not implemented")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        #print(self.total_steps)
        self.total_steps = 0
        self.first_step = True
        
        # Store start positions
        self.start_positions = self._get_initial_positions()
        
        # Generate new random targets
        self.current_targets = self._generate_random_targets()
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize previous distances for soft rewards
        positions = obs[:, 0:3]
        self.previous_distances = np.linalg.norm(positions - self.current_targets, axis=1)
        
        # Add target information to info
        info.update({
            'current_targets': self.current_targets.copy(),
            'target_radius': self.current_target_radius,
            'total_episodes': self.total_episodes,
            'success_rate_last_100': np.mean(self.episode_results) if len(self.episode_results) > 0 else 0.0,
        })
        
        return obs, info

    def step(self, action):
        """Execute one simulation step"""
        # Take physics step using parent class
        obs, _, terminated, truncated, info = super().step(action)
        # print('---')
        # print(self.total_steps)
        # print(truncated)
        
        # Get current drone positions
        positions = obs[:, 0:3]  # Shape: (NUM_DRONES, 3)
        
        # Initialize reward
        reward = 0.0
        episode_done = False
        episode_success = False
        
        # Calculate distances to targets
        distances_to_targets = np.linalg.norm(positions - self.current_targets, axis=1)
        
        # === SOFT REWARDS ===
        
        # 1. Distance reward (progress toward target)
        if not self.first_step and self.previous_distances is not None:
            distance_improvements = self.previous_distances - distances_to_targets
            distance_reward = self.lambda_distance * np.sum(distance_improvements)
            reward += distance_reward
        
        # 2. Angle reward (orientation toward target)
        angle_reward = 0.0
        for i in range(self.NUM_DRONES):
            # Get drone state for orientation
            state = self._getDroneStateVector(i)
            drone_pos = state[0:3]
            drone_yaw = state[9]  # Yaw angle from RPY
            
            # Calculate target direction in XY plane
            target_direction = self.current_targets[i][:2] - drone_pos[:2]  # Only X,Y
            target_direction_norm = np.linalg.norm(target_direction)
            
            if target_direction_norm > 0.01:  # Avoid division by zero
                # Normalize target direction
                target_direction = target_direction / target_direction_norm
                
                # Calculate drone's forward direction from yaw
                drone_forward = np.array([np.cos(drone_yaw), np.sin(drone_yaw)])
                
                # Calculate alignment (cosine of angle between directions)
                alignment = np.dot(drone_forward, target_direction)
                alignment = np.clip(alignment, -1, 1)  # Ensure valid range
                
                # Convert to positive reward (alignment ranges from -1 to 1)
                # Scale to 0-1 range: (alignment + 1) / 2
                normalized_alignment = (alignment + 1) / 2
                angle_reward += self.lambda_angle * normalized_alignment
        
        reward += angle_reward
        
        # print(truncated)
        # print(episode_done)
        
        # === BINARY REWARDS ===
        
        # Check if any drone reached its target
        targets_reached = distances_to_targets < self.target_tolerance
        
        if np.any(targets_reached):
            # Target reached - big positive reward and end episode
            reward += 200.0
            episode_done = True
            episode_success = True
            if self.gui:
                print(f"[SUCCESS] Target reached! Distance: {np.min(distances_to_targets):.3f}")
        
        # Check for crashes and out of bounds
        for i in range(self.NUM_DRONES):
            pos = positions[i]
            
            # Check if drone crashed (too low)
            if pos[2] < 0.1:
                reward -= self.crash_penalty
                episode_done = True
                if self.gui:
                    print(f"[CRASH] Drone {i} crashed (z={pos[2]:.3f})")
            
            # Check if drone went out of bounds
            if pos[2] > 3.0 or np.linalg.norm(pos[:2]) > 5.0:
                reward -= self.bounds_penalty
                episode_done = True
                if self.gui:
                    print(f"[OUT_OF_BOUNDS] Drone {i} out of bounds")
        
        # Check for inter-drone collisions
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < self.collision_distance:
                    reward -= self.crash_penalty
                    episode_done = True
                    if self.gui:
                        print(f"[COLLISION] Drones {i} and {j} collided")
        
        # Update episode state
        self.total_steps += 1
        
        # print(truncated)
        # print(episode_done)
        
        # Check if episode time limit reached
        if self.total_steps >= self.max_episode_steps:
            reward -= self.bounds_penalty  # Penalty for timeout
            episode_done = True
            if self.gui:
                print(f"[TIMEOUT] Episode timed out after {self.total_steps} steps")
        
        # Handle episode completion
        if episode_done:
            self.total_episodes += 1
            self.episode_results.append(1.0 if episode_success else 0.0)
            
            # Update adaptive difficulty
            self._update_adaptive_difficulty()
        
        # Update tracking for next step
        self.previous_distances = distances_to_targets.copy()
        self.first_step = False
        
        # print(truncated)
        # print(episode_done)
        
        # Update info with metrics
        info.update({
            'current_targets': self.current_targets.copy(),
            'target_radius': self.current_target_radius,
            'total_episodes': self.total_episodes,
            'success_rate_last_100': np.mean(self.episode_results) if len(self.episode_results) > 0 else 0.0,
            'distance_to_targets': distances_to_targets.copy(),
            'min_distance_to_target': np.min(distances_to_targets),
            'episode_success': episode_success,
            'targets_reached': targets_reached.copy(),
        })
        
        truncated = truncated[0]
        
        return obs, reward, episode_done, truncated, info

    # =====================================================================
    # REQUIRED ABSTRACT METHODS FROM BaseRLAviary/BaseAviary
    # =====================================================================

    def _computeReward(self):
        """Computes the current reward value."""
        # This is overridden in step() method
        return 0.0

    def _computeTerminated(self):
        """Computes the current terminated value."""
        # This is handled in step() method
        return False

    def _computeTruncated(self):
        """Computes the current truncated value."""
        # Episode truncated if we've exceeded maximum steps
        if self.total_steps >= self.max_episode_steps:
            return True, False
        return False, False

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        if self.current_targets is None:
            return {}
            
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        distances_to_targets = np.linalg.norm(positions - self.current_targets, axis=1)
        
        return {
            'current_targets': self.current_targets.copy(),
            'target_radius': self.current_target_radius,
            'total_episodes': self.total_episodes,
            'success_rate_last_100': np.mean(self.episode_results) if len(self.episode_results) > 0 else 0.0,
            'distance_to_targets': distances_to_targets.copy(),
            'min_distance_to_target': np.min(distances_to_targets),
        }


# Test the environment with adaptive difficulty and soft rewards
if __name__ == "__main__":
    print("Testing MultiTargetAviary with ADAPTIVE DIFFICULTY and SOFT REWARDS...")
    
    env = MultiTargetAviary(
        num_drones=1,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        episode_length_sec=3.0,
        target_radius_start=0.1,
        target_radius_max=1.0,
        target_radius_increment=0.1,
        success_threshold=0.9,
        evaluation_window=10,  # Smaller for testing
        lambda_distance=10.0,  # Distance improvement reward
        lambda_angle=1.0,      # Target orientation reward
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial target radius: {info['target_radius']:.2f}")
    
    # Test a few episodes
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        print(f"Target radius: {info['target_radius']:.2f}")
        print(f"Target position: {info['current_targets'][0]}")
        
        for step in range(env.max_episode_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                success = info.get('episode_success', False)
                print(f"Episode ended at step {step + 1}: {'SUCCESS' if success else 'FAILURE'}")
                print(f"Episode reward: {episode_reward:.1f}")
                print(f"Success rate: {info['success_rate_last_100']:.3f}")
                break
    
    env.close()
    print("\nAdaptive difficulty test completed!")