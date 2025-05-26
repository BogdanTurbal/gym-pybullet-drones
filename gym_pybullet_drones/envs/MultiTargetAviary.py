# MultiTargetAviary.py - Improved Version
import numpy as np
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiTargetAviary(MultiHoverAviary):
    """
    Multi-agent RL environment where drones must visit a sequence of targets
    while maintaining formation and avoiding collisions.
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
        act: ActionType = ActionType.ONE_D_RPM,
        target_sequence: np.ndarray = None,
        steps_per_target: int = 300,  # 5 seconds at 30Hz
        tolerance: float = 0.15,      # distance tolerance to "reach" target
        collision_distance: float = 0.3,  # minimum safe distance between drones
    ):
        """
        Parameters
        ----------
        target_sequence : ndarray
            Shape (n_phases, num_drones, 3) - sequence of target positions
        steps_per_target : int
            Number of simulation steps to spend at each target
        tolerance : float
            Distance within which a drone is considered to have "reached" target
        collision_distance : float
            Minimum distance to maintain between drones
        """
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
        
        # Default target sequence if none provided
        if target_sequence is None:
            target_sequence = self._create_default_targets()
        
        self.target_sequence = target_sequence
        self.steps_per_target = steps_per_target
        self.tolerance = tolerance
        self.collision_distance = collision_distance
        
        self.gui = gui
        self.NUM_DRONES = num_drones
        self.DRONE_MODEL = drone_model
        self.PYB_FREQ = pyb_freq
        
        # Episode state
        self.current_phase = 0
        self.step_in_phase = 0
        self.total_steps = 0
        self.max_episode_steps = len(target_sequence) * steps_per_target
        
        # Performance tracking
        self.targets_reached = np.zeros(num_drones, dtype=bool)
        self.collision_count = 0
        
        print(f"[MultiTargetAviary] Initialized with {len(target_sequence)} target phases")
        print(f"[MultiTargetAviary] Targets shape: {target_sequence.shape}")

    def _create_default_targets(self):
        """Create a default sequence of targets in a square formation"""
        # Create 4 phases with square formation targets
        targets = np.array([
            # Phase 0: Square corners
            [[ 1.0,  1.0, 0.5], [-1.0,  1.0, 0.5], [-1.0, -1.0, 0.5], [ 1.0, -1.0, 0.5]],
            [[ 1.0,  1.0, 0.5], [-1.0,  1.0, 0.5], [-1.0, -1.0, 0.5], [ 1.0, -1.0, 0.5]],
            [[ 1.0,  1.0, 0.5], [-1.0,  1.0, 0.5], [-1.0, -1.0, 0.5], [ 1.0, -1.0, 0.5]],
            [[ 1.0,  1.0, 0.5], [-1.0,  1.0, 0.5], [-1.0, -1.0, 0.5], [ 1.0, -1.0, 0.5]]
            # Phase 1: Rotate positions
            # [[-1.0,  1.0, 0.5], [-1.0, -1.0, 0.5], [ 1.0, -1.0, 0.5], [ 1.0,  1.0, 0.5]],
            # # Phase 2: Diamond formation
            # [[ 0.0,  0.5, 0.5], [-0.5,  0.0, 0.5], [ 0.0, -0.5, 0.5], [ 0.5,  0.0, 0.5]],
            # # Phase 3: Return to center
            # [[ 0.5,  0.5, 0.5], [-0.5,  0.5, 0.5], [-0.5, -0.5, 0.5], [ 0.5, -0.5, 0.5]]
        ])
        return targets[:, :self.NUM_DRONES, :]  # Adjust for actual number of drones

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        self.current_phase = 0
        self.step_in_phase = 0
        self.total_steps = 0
        self.targets_reached = np.zeros(self.NUM_DRONES, dtype=bool)
        self.collision_count = 0
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Add target information to info
        info['current_targets'] = self.get_current_targets()
        info['phase'] = self.current_phase
        
        return obs, info

    def get_current_targets(self):
        """Get current target positions for all drones"""
        if self.current_phase < len(self.target_sequence):
            return self.target_sequence[self.current_phase]
        else:
            # Return last targets if we've exceeded the sequence
            return self.target_sequence[-1]

    def step(self, action):
        """Execute one simulation step"""
        # Take physics step using parent class
        obs, _, terminated, truncated, info = super().step(action)
        
        # Get current drone positions
        positions = np.array([self._getDroneStateVector(i)[:3] for i in range(self.NUM_DRONES)])
        current_targets = self.get_current_targets()
        
        # Compute reward
        reward = self._compute_swarm_reward(positions, current_targets)
        
        # Update episode state
        self.step_in_phase += 1
        self.total_steps += 1
        
        # Check if we should advance to next phase
        if self.step_in_phase >= self.steps_per_target:
            self.current_phase += 1
            self.step_in_phase = 0
            self.targets_reached.fill(False)  # Reset for new targets
            if self.gui:
                print(f"Advanced to phase {self.current_phase}")

        # Check termination conditions
        done = self._is_episode_done()
        
        # Update info
        info.update({
            'current_targets': current_targets,
            'phase': self.current_phase,
            'step_in_phase': self.step_in_phase,
            'targets_reached': self.targets_reached.copy(),
            'collision_count': self.collision_count
        })
        
        return obs, reward, done, done, info

    def _compute_swarm_reward(self, positions, targets):
        """Compute reward based on target proximity and collision avoidance"""
        reward = 0.0
        
        # 1. Distance-to-target reward (primary objective)
        distances = np.linalg.norm(positions - targets, axis=1)
        
        # Exponential reward for being close to targets
        target_rewards = np.exp(-2.0 * distances)# * 2  # Max reward ~1.0 per drone
        reward += np.sum(target_rewards)
        
        # Bonus for reaching targets
        # newly_reached = (distances < self.tolerance) & (~self.targets_reached)
        # if np.any(newly_reached):
        #     reward += np.sum(newly_reached) * 2.0  # Bonus for reaching new targets
        #     self.targets_reached |= newly_reached
        
        # 2. Collision avoidance penalty
        # collision_penalty = 0.0
        # for i in range(self.NUM_DRONES):
        #     for j in range(i + 1, self.NUM_DRONES):
        #         distance = np.linalg.norm(positions[i] - positions[j])
        #         if distance < self.collision_distance:
        #             # Exponential penalty for being too close
        #             collision_penalty += -2.0 * np.exp(-(distance / 0.1))
        #             self.collision_count += 1
        
        # reward += collision_penalty
        
        # 3. Formation keeping bonus (stay reasonably close to each other)
        # center = np.mean(positions, axis=0)
        # formation_distances = np.linalg.norm(positions - center, axis=1)
        # if np.max(formation_distances) < 2.0:  # All drones within 2m of center
        #     reward += 0.5
        
        # 4. Stability bonus (penalize excessive tilting/rotation)
        # tilt_penalty = 0.0
        # for i in range(self.NUM_DRONES):
        #     state = self._getDroneStateVector(i)
        #     roll, pitch = state[7], state[8]  # Roll and pitch angles
        #     if abs(roll) > 0.3 or abs(pitch) > 0.3:  # > ~17 degrees
        #         tilt_penalty -= 0.5
        
        # reward += tilt_penalty
        
        return reward

    def _is_episode_done(self):
        """Check if episode should terminate"""
        # Episode ends when we've completed all phases
        if self.current_phase >= len(self.target_sequence):
            return True
            
        # Episode ends if we've exceeded maximum steps
        if self.total_steps >= self.max_episode_steps:
            return True
            
        # Episode ends if any drone goes too far out of bounds
        # positions = np.array([self._getDroneStateVector(i)[:3] for i in range(self.NUM_DRONES)])
        # if np.any(np.abs(positions[:, :2]) > 5.0) or np.any(positions[:, 2] > 3.0) or np.any(positions[:, 2] < 0.1):
        #     return True
            
        # # Episode ends if any drone is tilted too much
        # for i in range(self.NUM_DRONES):
        #     state = self._getDroneStateVector(i)
        #     if abs(state[7]) > 0.8 or abs(state[8]) > 0.8:  # ~45 degrees
        #         return True
                
        return False

    def _computeReward(self):
        """Override parent class reward computation"""
        # This method is called by parent class step(), but we handle reward in our step()
        return 0.0

    def _computeTerminated(self):
        """Override parent class termination logic"""
        return False  # We handle termination in our step() method

    def _computeTruncated(self):
        """Override parent class truncation logic"""
        return False  # We handle truncation in our step() method