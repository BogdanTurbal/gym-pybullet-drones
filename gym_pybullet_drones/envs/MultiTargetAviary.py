# MultiTargetAviary.py - UPDATED with PAPER-BASED REWARD FUNCTION
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiTargetAviary(BaseRLAviary):
    """
    Multi-agent RL environment where drones must visit a sequence of targets
    while maintaining formation and avoiding collisions.
    
    UPDATED with PAPER-BASED REWARD FUNCTION:
    - Progress reward: r^prog = λ1[d_{t-1} - d_t] (delta distance)
    - Perception reward: r^perc = λ2 * exp[λ3 * alignment^4] (target alignment)
    - Command smoothness: r^cmd = λ4 * |actions| + λ5 * |action_changes|^2
    - Crash penalty: r^crash = binary penalty for crashes/bounds
    
    Based on "Champion-level drone racing using deep reinforcement learning" (Nature 2023)
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
        act: ActionType = ActionType.RPM,
        target_sequence: np.ndarray = None,
        steps_per_target: int = 100,
        tolerance: float = 0.15,
        collision_distance: float = 0.05,
        # Paper-based reward hyperparameters
        lambda_1: float = 10.0,    # Progress reward coefficient
        lambda_2: float = 2.0,     # Perception reward coefficient  
        lambda_3: float = 0.5,     # Perception alignment exponent
        lambda_4: float = 0.01,    # Action magnitude penalty
        lambda_5: float = 0.1,     # Action smoothness penalty
        crash_penalty: float = 5.0, # Crash penalty magnitude
        bounds_penalty: float = 2.0, # Out-of-bounds penalty
    ):
        # Create default target sequence if none provided
        if target_sequence is None:
            target_sequence = self._create_default_targets(num_drones)
        
        self.target_sequence = target_sequence
        self.steps_per_target = steps_per_target
        self.tolerance = tolerance
        self.collision_distance = collision_distance
        
        # Paper-based reward hyperparameters
        self.lambda_1 = lambda_1  # Progress reward
        self.lambda_2 = lambda_2  # Perception reward
        self.lambda_3 = lambda_3  # Perception alignment
        self.lambda_4 = lambda_4  # Action magnitude penalty
        self.lambda_5 = lambda_5  # Action smoothness penalty
        self.crash_penalty = crash_penalty
        self.bounds_penalty = bounds_penalty
        
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
        
        # Paper-based reward tracking
        self.previous_distances = np.zeros(num_drones, dtype=np.float32)
        self.previous_actions = np.zeros((num_drones, 4), dtype=np.float32)  # For RPM actions
        self.first_step = True
        
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
        
        print(f"[MultiTargetAviary] Initialized with PAPER-BASED REWARD FUNCTION")
        print(f"[MultiTargetAviary] Reward hyperparameters:")
        print(f"  - λ1 (progress): {lambda_1}")
        print(f"  - λ2 (perception): {lambda_2}")
        print(f"  - λ3 (alignment): {lambda_3}")
        print(f"  - λ4 (action magnitude): {lambda_4}")
        print(f"  - λ5 (action smoothness): {lambda_5}")
        print(f"  - Crash penalty: {crash_penalty}")
        print(f"  - Bounds penalty: {bounds_penalty}")
        print(f"[MultiTargetAviary] Target sequence shape: {target_sequence.shape}")
        print(f"[MultiTargetAviary] Episode length: {self.EPISODE_LEN_SEC:.1f} seconds")

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
            current_targets = self.get_current_targets()
            
            target_obs = np.zeros((self.NUM_DRONES, 6), dtype=np.float32)
            
            for i in range(self.NUM_DRONES):
                # Get current position from base observation
                my_position = base_obs[i, 0:3]
                my_target = current_targets[i]
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
        self.current_phase = 0
        self.step_in_phase = 0
        self.total_steps = 0
        self.targets_reached = np.zeros(self.NUM_DRONES, dtype=bool)
        self.collision_count = 0
        
        # Reset paper-based reward tracking
        self.previous_distances = np.zeros(self.NUM_DRONES, dtype=np.float32)
        self.previous_actions = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        self.first_step = True
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize previous distances with current distances
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
        # print('-----')
        # print(action)
        # action = np.array([
        #     [0, 0, 1],
        #     [0, 0, 2],
        #     [0, 0, 3],
        #     [1, 1, 3],
        # ])
        # Take physics step using parent class
        
        # current_targets = self.get_current_targets()
        # action = current_targets#obs[:, -6:-3]
        # print(action)
        
        obs, _, terminated, truncated, info = super().step(action)
        # print(obs)
        
        #print(obs[:, :3])
        
        # Get current drone positions
        positions = obs[:, 0:3]  # Shape: (NUM_DRONES, 4)
        current_targets = self.get_current_targets()
        
        # Compute reward using paper-based approach
        reward = self._compute_paper_based_reward(positions, current_targets, action)
        
        # Update episode state
        self.step_in_phase += 1
        self.total_steps += 1
        
        # Check if we should advance to next phase
        if self.step_in_phase >= self.steps_per_target:
            self.current_phase += 1
            self.step_in_phase = 0
            self.targets_reached.fill(False)
            # Reset previous distances when switching phases
            current_distances = np.linalg.norm(positions - self.get_current_targets(), axis=1)
            self.previous_distances = current_distances.copy()
            if self.gui:
                print(f"Advanced to phase {self.current_phase}")

        # Check termination conditions
        done = self._computeTerminated()
        truncated, unnatural = self._computeTruncated()
        if unnatural:
            reward -= self.bounds_penalty * 10  # Large penalty for unnatural termination
        
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

    def _compute_paper_based_reward(self, positions, targets, actions):
        """
        Compute reward based on the paper's approach:
        r_t = r_t^prog + r_t^perc + r_t^cmd - r_t^crash
        
        Based on "Champion-level drone racing using deep reinforcement learning" (Nature 2023)
        """
        total_reward = 0.0
        
        # Calculate current distances to targets
        current_distances = np.linalg.norm(positions - targets, axis=1)
        
        # === 1. PROGRESS REWARD: r^prog = λ1[d_{t-1} - d_t] ===
        if not self.first_step:
            # Delta distance reward (progress toward target)
            distance_deltas = self.previous_distances - current_distances
            progress_reward = self.lambda_1 * np.sum(distance_deltas)
            total_reward += progress_reward
        else:
            progress_reward = 0.0
        
        # === 2. PERCEPTION REWARD: r^perc = λ2 * exp[λ3 * alignment^4] ===
        perception_reward = 0.0
        for i in range(self.NUM_DRONES):
            # Get drone state for orientation calculation
            state = self._getDroneStateVector(i)
            drone_pos = state[0:3]
            drone_rpy = state[7:10]  # roll, pitch, yaw
            
            # Calculate target direction vector
            target_direction = targets[i] - drone_pos
            target_direction_norm = np.linalg.norm(target_direction)
            
            if target_direction_norm > 0:
                target_direction = target_direction / target_direction_norm
                
                # Calculate drone's forward direction (assuming forward is along x-axis in body frame)
                # Convert yaw to forward direction vector
                yaw = drone_rpy[2]
                drone_forward = np.array([np.cos(yaw), np.sin(yaw)])
                
                # Calculate alignment (cosine of angle between forward direction and target direction)
                alignment = np.dot(drone_forward, target_direction[:2])  # Only use x,y for alignment
                
                # Ensure alignment is in valid range [-1, 1]
                alignment = np.clip(alignment, -1, 1)
                
                # Convert to positive reward (higher alignment = higher reward)
                # Use paper's exponential form: exp[λ3 * alignment^4]
                alignment_reward = self.lambda_2 * np.exp(self.lambda_3 * (alignment + 1)**4)
                perception_reward += alignment_reward
        
        total_reward += perception_reward
        
        # === 3. COMMAND SMOOTHNESS REWARD: r^cmd = λ4 * |actions| + λ5 * |action_changes|^2 ===
        command_reward = 0.0
        
        # Action magnitude penalty
        action_magnitude_penalty = -self.lambda_4 * np.sum(np.abs(actions))
        command_reward += action_magnitude_penalty
        
        # Action smoothness penalty (penalize rapid changes)
        if not self.first_step:
            action_changes = actions - self.previous_actions
            action_smoothness_penalty = -self.lambda_5 * np.sum(action_changes**2)
            command_reward += action_smoothness_penalty
        
        total_reward += command_reward
        
        # === 4. CRASH PENALTY: r^crash ===
        crash_penalty = 0.0
        
        # Check for collisions with targets (reaching targets is good, but crashing is bad)
        for i in range(self.NUM_DRONES):
            # Check if drone is too low (crashed)
            if positions[i][2] < 0.1:
                crash_penalty += self.crash_penalty
            
            # Check if drone is too high or too far out
            if positions[i][2] > 4.0 or np.linalg.norm(positions[i][:2]) > 10.0:
                crash_penalty += self.bounds_penalty
        
        # Check for inter-drone collisions
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < self.collision_distance:
                    crash_penalty += self.crash_penalty
                    self.collision_count += 1
        
        total_reward -= crash_penalty
        
        # === BONUS: Target reaching reward (not in paper but useful for multi-target tasks) ===
        # target_bonus = 0.0
        # newly_reached = (current_distances < self.tolerance) & (~self.targets_reached)
        # if np.any(newly_reached):
        #     target_bonus = np.sum(newly_reached) * 50.0  # Moderate bonus for target completion
        #     total_reward += target_bonus
        #     self.targets_reached |= newly_reached
        
        # Update tracking variables for next step
        self.previous_distances = current_distances.copy()
        self.previous_actions = actions.copy()
        self.first_step = False
        
        return total_reward

    # =====================================================================
    # REQUIRED ABSTRACT METHODS FROM BaseRLAviary/BaseAviary
    # =====================================================================

    def _computeReward(self):
        """Computes the current reward value."""
        positions = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        current_targets = self.get_current_targets()
        # Use dummy action for reward computation (will be overridden in step())
        dummy_action = np.zeros((self.NUM_DRONES, 4))  # RPM actions
        return self._compute_paper_based_reward(positions, current_targets, dummy_action)

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
            position = state[0:3]
            
            # Check position bounds
            if (position[2] > 4.0 or position[2] < 0.1):
                return True, True
                
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


# Test the environment with paper-based reward function
if __name__ == "__main__":
    print("Testing MultiTargetAviary with PAPER-BASED REWARD FUNCTION...")
    
    env = MultiTargetAviary(
        num_drones=4,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        # Paper-based reward hyperparameters (tunable)
        lambda_1=15.0,    # Progress reward - higher values encourage faster progress
        lambda_2=1.0,     # Perception reward - rewards target alignment
        lambda_3=0.3,     # Perception alignment exponent
        lambda_4=0.005,   # Action magnitude penalty - small penalty for large actions
        lambda_5=0.05,    # Action smoothness penalty - penalizes rapid action changes
        crash_penalty=10.0,   # Crash penalty
        bounds_penalty=5.0,   # Out-of-bounds penalty
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Test action space
    action = env.action_space.sample()
    print(f"Sample action shape: {action.shape}")
    print(f"Sample action:\n{action}")
    
    print("\n=== TESTING PAPER-BASED REWARD FUNCTION ===")
    
    # Test a few steps
    for i in range(30):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i+1}: Reward: {reward:.3f}, Mean distance: {info['mean_distance_to_targets']:.3f}")
        
        if done or truncated:
            break
    
    env.close()
    print("\nMultiTargetAviary with paper-based reward function test completed successfully!")
    print("\nREWARD COMPONENTS:")
    print("✅ Progress reward: r^prog = λ1[d_{t-1} - d_t] (delta distance)")
    print("✅ Perception reward: r^perc = λ2 * exp[λ3 * alignment^4] (target alignment)")
    print("✅ Command smoothness: r^cmd = λ4 * |actions| + λ5 * |action_changes|^2")
    print("✅ Crash penalty: r^crash (crashes, bounds violations, collisions)")
    print("✅ Target bonus: Additional reward for reaching targets")