# MultiTargetAviary.py - ADAPTIVE DIFFICULTY SINGLE DRONE VERSION
import numpy as np
import gymnasium as gym # Changed from gym to gymnasium
from gymnasium import spaces # Changed from gym to gymnasium
from collections import deque
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
# Assuming enums are in this path or defined elsewhere accessible
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
import pybullet as p

# If enums.py is not provided or KIN_DEPTH is not there, define locally for now
# This should ideally be in your global enums.py
try:
    from gym_pybullet_drones.utils.enums import ObservationType
    if not hasattr(ObservationType, 'KIN_DEPTH'): # pragma: no cover
        from enum import Enum
        class ExtendedObservationType(Enum): 
            KIN = "kin"
            RGB = "rgb"
            KIN_DEPTH = "kin_depth"
        ObservationType = ExtendedObservationType
except ImportError: # pragma: no cover
    from enum import Enum
    class ObservationType(Enum):
        KIN = "kin"
        RGB = "rgb"
        KIN_DEPTH = "kin_depth" # New observation type

class MultiTargetAviary(BaseRLAviary):
    """
    Adaptive difficulty single-drone RL environment with KIN+Depth observations.
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
        obs: ObservationType = ObservationType.KIN_DEPTH, 
        act: ActionType = ActionType.RPM,
        episode_length_sec: float = 3.0,
        target_radius_start: float = 0.1,
        target_radius_max: float = 1.0, 
        target_radius_increment: float = 0.1,
        target_tolerance: float = 0.05, 
        success_threshold: float = 0.9,
        evaluation_window: int = 100,
        collision_distance: float = 0.1,
        lambda_distance: float = 10.0,
        lambda_angle: float = 1.0,
        crash_penalty: float = 200.0,
        bounds_penalty: float = 200.0,
        individual_target_reward: float = 400.0,
        add_obstacles: bool = True,      # New parameter to enable/disable obstacles
        obs_prob: float = 0.5,            # New parameter for obstacle density
        obstacle_size: float = 0.1        # New parameter for obstacle size
    ):
        self.gui = gui
        
        self.episode_length_sec = episode_length_sec # Store for calculating max_episode_steps
        self.target_radius_start = target_radius_start
        self.target_radius_max = target_radius_max
        self.target_radius_increment = target_radius_increment
        self.target_tolerance = target_tolerance
        self.success_threshold = success_threshold
        self.evaluation_window = evaluation_window
        self.collision_distance = collision_distance
        self.crash_penalty = crash_penalty
        self.bounds_penalty = bounds_penalty
        self.individual_target_reward = individual_target_reward
        
        # New parameters for obstacles
        self.add_obstacles = add_obstacles
        self.obs_prob = obs_prob
        self.obstacle_size = obstacle_size
        self.obstacles_info = []  # To store information about generated obstacles
        
        self.lambda_distance = lambda_distance
        self.lambda_angle = lambda_angle
        
        self.NUM_DRONES = num_drones 
        self.DRONE_MODEL = drone_model
        
        self.current_target_radius = target_radius_start
        self.episode_results = deque(maxlen=evaluation_window)
        self.total_episodes = 0
        
        self.total_steps = 0
        self.current_targets = None 
        self.start_positions = None 
        
        self.previous_distances = None
        self.first_step = True
        # Initialize targets_reached_flags early, using num_drones passed to __init__
        self.targets_reached_flags = np.zeros(num_drones, dtype=bool)

        if initial_xyzs is None:
            current_initial_xyzs = self._get_initial_positions()
        else:
            current_initial_xyzs = initial_xyzs

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=current_initial_xyzs, 
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq, # This sets self.CTRL_FREQ in BaseAviary
            gui=gui,
            record=record,
            obs=obs, 
            act=act
        )
        
        # Now that super().__init__() has run, self.CTRL_FREQ is set by BaseAviary.
        # We can safely calculate self.max_episode_steps.
        self.max_episode_steps = int(self.episode_length_sec * self.CTRL_FREQ)

        if self.OBS_TYPE == ObservationType.KIN_DEPTH or self.OBS_TYPE == ObservationType.KIN:
            if self.NUM_DRONES == 1:
                 self.base_kin_obs_dim = self.observation_space["kin"].shape[0] if self.OBS_TYPE == ObservationType.KIN_DEPTH else self.observation_space.shape[0]
            else: 
                 self.base_kin_obs_dim = self.observation_space["kin"].shape[1] if self.OBS_TYPE == ObservationType.KIN_DEPTH else self.observation_space.shape[1]
        
        print(f"[MultiTargetAviary] Initialized with {self.OBS_TYPE.value} observations.")
        print(f"[MultiTargetAviary] Max episode steps: {self.max_episode_steps} (Duration: {self.episode_length_sec}s, Ctrl Freq: {self.CTRL_FREQ}Hz)")
        print(f"[MultiTargetAviary] Current target radius: {self.current_target_radius:.2f}m")
        if self.add_obstacles:
            print(f"[MultiTargetAviary] Obstacles enabled with probability: {self.obs_prob:.2f}, size: {self.obstacle_size:.2f}m")

    def _get_initial_positions(self):
        positions = []
        for i in range(self.NUM_DRONES): 
            if self.NUM_DRONES == 1:
                 positions.append([0.0, 0.0, 1.0]) 
            else:
                 offset = i * 0.5 
                 positions.append([offset, offset, 1.0])
        return np.array(positions, dtype=np.float32) if self.NUM_DRONES > 0 else None

    def _generate_random_targets(self):
        targets = []
        # Use self.INIT_XYZS from BaseAviary as the definitive start positions after reset/init
        current_start_positions = self.INIT_XYZS 
        if current_start_positions is None: # Should not happen after BaseAviary._housekeeping
            print("[WARNING] self.INIT_XYZS is None in _generate_random_targets. Falling back.")
            current_start_positions = self._get_initial_positions()


        for i in range(self.NUM_DRONES):
            # print('=' * 100)
            # print("NUM_DRONES: ", self.NUM_DRONES)
            while True:
                x, y, z = np.random.uniform(-1, 1, 3)
                ln_sq = x*x + y*y + z*z 
                min_l_sq = (0.05 / self.current_target_radius) ** 2 #if self.current_target_radius > 0.001 else 0 # Avoid division by zero
                
                target_z_coord = current_start_positions[i, 2] + z * self.current_target_radius
                if min_l_sq < ln_sq <= 1.0 and target_z_coord >= 0.1: # Target must be above ground
                    target = current_start_positions[i, :] + self.current_target_radius * np.array([x, y, z])
                    targets.append(target)
                    
                    # print('fuck ' * 20)
                    # obstacle_id = p.loadURDF("duck_vhacd.urdf",
                    #                     target,
                    #                     p.getQuaternionFromEuler([0, 0, 0]),
                    #                     globalScaling=1,
                    #                     physicsClientId=self.CLIENT)
                    
                    # p.changeDynamics(obstacle_id, 
                    #             linkIndex=-1,  # -1 refers to the base link
                    #             mass=0,  # Setting mass to 0 makes it static
                    #             physicsClientId=self.CLIENT)
                    break
        return np.array(targets, dtype=np.float32)

    def _update_adaptive_difficulty(self):
        if len(self.episode_results) >= self.evaluation_window:
            success_rate = np.mean(self.episode_results)
            if success_rate >= self.success_threshold:
                old_radius = self.current_target_radius
                self.current_target_radius = min(
                    self.target_radius_max,
                    self.current_target_radius + self.target_radius_increment
                )
                if self.current_target_radius > old_radius + 1e-5: # Check for actual change
                    print(f"[AdaptiveDifficulty] Success rate: {success_rate:.3f}, Increased radius: {old_radius:.2f} -> {self.current_target_radius:.2f}")
                    self.episode_results.clear()

    def _addObstacles(self):
        """Add obstacles between the drone and its target"""
        # If obstacles are not enabled, don't add any
        # print('ok' * 200)
        # print(self.add_obstacles, self.current_targets)
        if not self.add_obstacles or self.current_targets is None:
            return
        
        # Clear previous obstacles info
        self.obstacles_info = []
        
        # For each drone, add obstacles on the path to its target
        for i in range(self.NUM_DRONES):
            # Get the starting position and target for this drone
            start_pos = self.INIT_XYZS[i]
            target_pos = self.current_targets[i]
            
            # Calculate the direction vector from start to target
            direction = target_pos - start_pos
            distance = np.linalg.norm(direction)
            
            # Determine the number of obstacles based on obs_prob and distance
            # Higher obs_prob means more obstacles
            import random
            random_value = random.uniform(0, distance)
            # print('-=-=' * 30)
            # print(random_value, distance, int(random_value * 4))
            num_obstacles = int(random_value * 4)  # Scale factor of 5 is arbitrary, adjust as needed
            
            # Place obstacles along the path
            for j in range(num_obstacles):
                # Uniform distribution along the path
                u = np.random.uniform(0.2, 0.8)  # Avoid placing too close to start or target
                
                # Base position along the path
                base_pos = start_pos + u * direction
                
                # Add Gaussian noise with std=0.2
                noise = np.random.normal(0, 0.2, size=3)
                
                # Ensure the obstacle is not too close to the ground
                obstacle_pos = base_pos + noise
                obstacle_pos[2] = max(obstacle_pos[2], 0.1 + self.obstacle_size)  # Keep above ground
                
                # Choose random obstacle type: 0 for sphere, 1 for cube, 2 for quadrocopter
                obstacle_type = np.random.randint(0, 3)  # Now includes quadrocopter
                
                rand_scale = max(0.1, 1 + np.random.normal(0, 1.0))
                
                if obstacle_type == 0:  # Sphere
                    obstacle_id = p.loadURDF("sphere2.urdf",
                                        obstacle_pos,
                                        p.getQuaternionFromEuler([0, 0, 0]),
                                        globalScaling=rand_scale * self.obstacle_size,
                                        physicsClientId=self.CLIENT)
                    self.obstacles_info.append({
                        'id': obstacle_id,
                        'type': 'sphere',
                        'position': obstacle_pos.tolist(),
                        'size': self.obstacle_size * rand_scale
                    })
                elif obstacle_type == 1:  # Cube
                    obstacle_id = p.loadURDF("cube.urdf",
                                        obstacle_pos,
                                        p.getQuaternionFromEuler([0, 0, 0]),
                                        globalScaling=rand_scale * self.obstacle_size * 0.5,
                                        physicsClientId=self.CLIENT)
                    self.obstacles_info.append({
                        'id': obstacle_id,
                        'type': 'cube',
                        'position': obstacle_pos.tolist(),
                        'size': self.obstacle_size * rand_scale * 0.5
                    })
                else:  # Quadrocopter (obstacle_type == 2)
                    # Random orientation for the quadrocopter obstacle
                    random_yaw = np.random.uniform(0, 2 * np.pi)
                    random_pitch = np.random.uniform(-0.3, 0.3)  # Slight tilt
                    random_roll = np.random.uniform(-0.3, 0.3)   # Slight tilt
                    
                    # Scale factor for quadrocopter (typically larger than other obstacles)
                    quad_scale = rand_scale * self.obstacle_size * 2.0  # Make it more visible
                    
                    try:
                        # Try different possible quadrocopter URDF files
                        urdf_options = [
                            "/Users/bohdan.turbal/Desktop/dipl_nw_pp/gym-pybullet-drones/gym_pybullet_drones/assets/cf2p.urdf",  # Crazyflie 2.X (most common in gym_pybullet_drones)
                            "/Users/bohdan.turbal/Desktop/dipl_nw_pp/gym-pybullet-drones/gym_pybullet_drones/assets/racer.urdf",  # Crazyflie 2.P
                            "/Users/bohdan.turbal/Desktop/dipl_nw_pp/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf",    # HummingBird
                            # "race.urdf"   # Racing drone
                        ]
                        
                        # Try to load one of the available drone URDFs
                        obstacle_id = None
                        quad_urdf_used = None
                        
                        for urdf_file in urdf_options:
                            try:
                                obstacle_id = p.loadURDF(urdf_file,
                                                    obstacle_pos,
                                                    p.getQuaternionFromEuler([random_roll, random_pitch, random_yaw]),
                                                    globalScaling=quad_scale,
                                                    physicsClientId=self.CLIENT)
                                quad_urdf_used = urdf_file
                                break  # Successfully loaded, exit the loop
                            except Exception as e:
                                continue  # Try next URDF file
                        
                        if obstacle_id is not None:
                            # Successfully loaded a quadrocopter obstacle
                            self.obstacles_info.append({
                                'id': obstacle_id,
                                'type': 'quadrocopter',
                                'position': obstacle_pos.tolist(),
                                'size': rand_scale * self.obstacle_size * 2.0,  # Larger collision radius for quads
                                'urdf_file': quad_urdf_used,
                                'orientation': [random_roll, random_pitch, random_yaw]
                            })
                        else:
                            # Fallback to cube if no quadrocopter URDF is available
                            print("[WARNING] No quadrocopter URDF available, falling back to cube")
                            obstacle_id = p.loadURDF("cube_small.urdf",
                                                obstacle_pos,
                                                p.getQuaternionFromEuler([0, 0, 0]),
                                                globalScaling=rand_scale * self.obstacle_size * 0.5,
                                                physicsClientId=self.CLIENT)
                            self.obstacles_info.append({
                                'id': obstacle_id,
                                'type': 'cube',
                                'position': obstacle_pos.tolist(),
                                'size': self.obstacle_size * rand_scale * 0.5
                            })
                            
                    except Exception as e:
                        print(f"[ERROR] Failed to load quadrocopter obstacle: {e}")
                        # Fallback to sphere
                        obstacle_id = p.loadURDF("sphere2.urdf",
                                            obstacle_pos,
                                            p.getQuaternionFromEuler([0, 0, 0]),
                                            globalScaling=rand_scale * self.obstacle_size,
                                            physicsClientId=self.CLIENT)
                        self.obstacles_info.append({
                            'id': obstacle_id,
                            'type': 'sphere',
                            'position': obstacle_pos.tolist(),
                            'size': self.obstacle_size * rand_scale
                        })
                
                # Make the obstacle static (no physics simulation)
                if obstacle_id is not None:
                    p.changeDynamics(obstacle_id, 
                                linkIndex=-1,  # -1 refers to the base link
                                mass=0,  # Setting mass to 0 makes it static
                                physicsClientId=self.CLIENT)
            
            if self.gui:
                quad_count = sum(1 for obs in self.obstacles_info if obs['type'] == 'quadrocopter')
                total_count = len(self.obstacles_info)
                print(f"[MultiTargetAviary] Added {total_count} obstacles between drone {i} and target ({quad_count} quadrocopters)")

    def _observationSpace(self):
        base_obs_space = super()._observationSpace() 
        if self.OBS_TYPE == ObservationType.KIN_DEPTH:
            original_kin_space = base_obs_space.spaces["kin"]
            target_features_per_drone_kin = 6
            if self.NUM_DRONES == 1:
                new_kin_dim = original_kin_space.shape[0] + target_features_per_drone_kin
                new_kin_low = np.concatenate([original_kin_space.low, np.full(target_features_per_drone_kin, -np.inf)])
                new_kin_high = np.concatenate([original_kin_space.high, np.full(target_features_per_drone_kin, np.inf)])
                augmented_kin_space = spaces.Box(low=new_kin_low, high=new_kin_high, dtype=np.float32)
            else: 
                new_kin_dim_per_agent = original_kin_space.shape[1] + target_features_per_drone_kin
                original_low_per_agent = original_kin_space.low[0,:] 
                original_high_per_agent = original_kin_space.high[0,:]
                new_low_per_agent = np.concatenate([original_low_per_agent, np.full(target_features_per_drone_kin, -np.inf)])
                new_high_per_agent = np.concatenate([original_high_per_agent, np.full(target_features_per_drone_kin, np.inf)])
                augmented_kin_space = spaces.Box(
                    low=np.tile(new_low_per_agent, (self.NUM_DRONES,1)),
                    high=np.tile(new_high_per_agent, (self.NUM_DRONES,1)),
                    dtype=np.float32)
            return spaces.Dict({"kin": augmented_kin_space, "depth": base_obs_space.spaces["depth"]})
        elif self.OBS_TYPE == ObservationType.KIN:
            original_kin_space = base_obs_space
            target_features_kin = 6 
            if self.NUM_DRONES == 1:
                new_dim = original_kin_space.shape[0] + target_features_kin
                new_low = np.concatenate([original_kin_space.low, np.full(target_features_kin, -np.inf)])
                new_high = np.concatenate([original_kin_space.high, np.full(target_features_kin, np.inf)])
                return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
            else: 
                new_dim_per_agent = original_kin_space.shape[1] + target_features_kin
                original_low_per_agent = original_kin_space.low[0,:]
                original_high_per_agent = original_kin_space.high[0,:]
                new_low_per_agent = np.concatenate([original_low_per_agent, np.full(target_features_kin, -np.inf)])
                new_high_per_agent = np.concatenate([original_high_per_agent, np.full(target_features_kin, np.inf)])
                return spaces.Box(low=np.tile(new_low_per_agent, (self.NUM_DRONES,1)), high=np.tile(new_high_per_agent, (self.NUM_DRONES,1)), dtype=np.float32)
        else: 
            return base_obs_space

    def _computeObs(self):
        base_obs = super()._computeObs()
        if self.current_targets is None: 
             self.current_targets = self._generate_random_targets()

        if self.OBS_TYPE == ObservationType.KIN_DEPTH:
            current_kin_obs = base_obs["kin"] 
            target_info_list = []
            for i in range(self.NUM_DRONES):
                drone_pos_idx_end = 3
                my_position = current_kin_obs[i, 0:drone_pos_idx_end] if self.NUM_DRONES > 1 and current_kin_obs.ndim == 2 else current_kin_obs[0:drone_pos_idx_end]
                my_target = self.current_targets[i]
                relative_target = my_target - my_position
                target_obs_features = np.array([my_target[0], my_target[1], my_target[2], relative_target[0], relative_target[1], relative_target[2]], dtype=np.float32)
                target_info_list.append(target_obs_features)
            all_targets_info = np.array(target_info_list)
            if self.NUM_DRONES == 1:
                augmented_kin = np.concatenate([current_kin_obs, all_targets_info[0]])
            else:
                augmented_kin = np.hstack([current_kin_obs, all_targets_info])
            return {"kin": augmented_kin.astype(np.float32), "depth": base_obs["depth"]}
        elif self.OBS_TYPE == ObservationType.KIN:
            current_kin_obs = base_obs
            target_info_list = []
            for i in range(self.NUM_DRONES):
                drone_pos_idx_end = 3
                my_position = current_kin_obs[i, 0:drone_pos_idx_end] if self.NUM_DRONES > 1 and current_kin_obs.ndim == 2 else current_kin_obs[0:drone_pos_idx_end]
                my_target = self.current_targets[i]
                relative_target = my_target - my_position
                target_obs_features = np.array([my_target[0], my_target[1], my_target[2], relative_target[0], relative_target[1], relative_target[2]], dtype=np.float32)
                target_info_list.append(target_obs_features)
            all_targets_info = np.array(target_info_list)
            if self.NUM_DRONES == 1:
                return np.concatenate([current_kin_obs, all_targets_info[0]]).astype(np.float32)
            else:
                return np.hstack([current_kin_obs, all_targets_info]).astype(np.float32)
        else: 
            return base_obs

    def reset(self, seed=None, options=None):
        super_obs, super_info = super().reset(seed=seed, options=options) 
        self.total_steps = 0
        self.first_step = True
        
        # Clear previous obstacles info
        self.obstacles_info = []
        
        # self.INIT_XYZS is set by BaseAviary's _housekeeping, which is called
        # during its __init__ and also at the start of its reset().
        # To generate targets for *this* episode before super().reset() might change INIT_XYZS
        # (if it re-randomizes them based on new seed), we generate targets based on
        # the current INIT_XYZS or what _get_initial_positions would give.
        if self.INIT_XYZS is not None:
             self.start_positions = self.INIT_XYZS.copy() # Use current drone starting positions
        else: # Fallback if called before BaseAviary init fully sets INIT_XYZS
            self.start_positions = self._get_initial_positions()

        self.current_targets = self._generate_random_targets()
        self.targets_reached_flags = np.zeros(self.NUM_DRONES, dtype=bool)
        
        if self.add_obstacles:
            self._addObstacles()

        
        # After super().reset(), self.pos is updated to the new reset positions
        current_positions_after_reset = self.pos

        if self.current_targets is None: # Should be set by now
             self.current_targets = self._generate_random_targets()

        self.previous_distances = np.linalg.norm(current_positions_after_reset - self.current_targets, axis=1)
        
        # Recompute info with fully reset state
        info = self._computeInfo() 
        obs = super_obs#self._computeObs()
        
        return obs, info

    def step(self, action):
        obs, reward_val, terminated_val, truncated_tuple, info = super().step(action)
        # In Gymnasium, truncated is a bool, not a tuple. Info is the dict.
        # The tuple return for truncated might be from an older gym version or custom wrapper.
        # Standard Gymnasium step returns: obs, reward, terminated, truncated, info
        # where truncated is bool.
        truncated_val = truncated_tuple[0] if isinstance(truncated_tuple, tuple) else truncated_tuple

        self.total_steps += 1 
        episode_done = terminated_val or truncated_val

        if episode_done:
            self.total_episodes += 1
            current_episode_success = info.get('episode_success', False) 
            self.episode_results.append(1.0 if current_episode_success else 0.0)
            self._update_adaptive_difficulty()
            
        self.first_step = False
        return obs, reward_val, terminated_val, truncated_val, info

    def _computeReward(self):
        reward = 0.0
        current_positions = self.pos 
        distances_to_targets = np.linalg.norm(current_positions - self.current_targets, axis=1)
        
        if not self.first_step and self.previous_distances is not None:
            distance_improvements = self.previous_distances - distances_to_targets
            for i in range(self.NUM_DRONES):
                if not self.targets_reached_flags[i]: 
                    reward += self.lambda_distance * distance_improvements[i]
        
        angle_reward_sum = 0.0
        for i in range(self.NUM_DRONES):
            if not self.targets_reached_flags[i]: 
                drone_pos = self.pos[i, :]
                drone_yaw = self.rpy[i, 2] 
                target_direction_xy = self.current_targets[i, :2] - drone_pos[:2]
                target_direction_norm = np.linalg.norm(target_direction_xy)
                if target_direction_norm > 0.01:
                    target_direction_xy /= target_direction_norm
                    drone_forward_xy = np.array([np.cos(drone_yaw), np.sin(drone_yaw)])
                    alignment = np.dot(drone_forward_xy, target_direction_xy)
                    normalized_alignment = (alignment + 1) / 2 
                    angle_reward_sum += self.lambda_angle * normalized_alignment
        reward += angle_reward_sum

        targets_newly_reached_this_step = (distances_to_targets < self.target_tolerance) & ~self.targets_reached_flags
        for i in range(self.NUM_DRONES):
            if targets_newly_reached_this_step[i]:
                reward += self.individual_target_reward
                self.targets_reached_flags[i] = True 
                if self.gui: print(f"[TARGET REACHED] Drone {i} distance: {distances_to_targets[i]:.3f}")
        
        current_pos_for_penalty_check = self.pos 
        for i in range(self.NUM_DRONES):
            if current_pos_for_penalty_check[i, 2] < 0.05: 
                reward -= self.crash_penalty 
                break 
        
        if self.NUM_DRONES > 1:
            collided_this_step = False
            for i in range(self.NUM_DRONES):
                for j in range(i + 1, self.NUM_DRONES):
                    if np.linalg.norm(current_pos_for_penalty_check[i] - current_pos_for_penalty_check[j]) < self.collision_distance:
                        reward -= self.crash_penalty 
                        collided_this_step = True
                        break
                if collided_this_step: break
        
        # Check for collisions with obstacles
        # if self.add_obstacles and len(self.obstacles_info) > 0:
        #     for i in range(self.NUM_DRONES):
        #         for obstacle in self.obstacles_info:
        #             obstacle_pos = np.array(obstacle['position'])
        #             obstacle_size = obstacle['size']
        #             # Collision distance depends on obstacle size
        #             collision_threshold = self.collision_distance + obstacle_size
        #             distance_to_obstacle = np.linalg.norm(current_pos_for_penalty_check[i] - obstacle_pos)
        #             if distance_to_obstacle < collision_threshold:
        #                 reward -= self.crash_penalty
        #                 if self.gui: print(f"[OBSTACLE COLLISION] Drone {i} hit {obstacle['type']}")
        #                 break

        # Timeout penalty calculation using self.max_episode_steps
        # self.step_counter is PyBullet steps, self.PYB_STEPS_PER_CTRL is step ratio
        num_control_steps_so_far = self.step_counter // self.PYB_STEPS_PER_CTRL if self.PYB_STEPS_PER_CTRL > 0 else self.step_counter
        # Check if this *current* step will be the one that meets or exceeds max_episode_steps
        if num_control_steps_so_far >= self.max_episode_steps -1 : # -1 because truncation happens *after* this step if it's the last one
             if not np.all(self.targets_reached_flags): 
                reward -= self.bounds_penalty / 2

        reward -= 0.05
        self.previous_distances = distances_to_targets.copy()
        return reward

    def _computeTerminated(self):
        terminated = False
        current_positions = self.pos
        if np.all(self.targets_reached_flags):
            terminated = True
            if self.gui: print(f"[ALL TARGETS REACHED] Terminating episode.")
            return terminated
            
        # Check for crashes with ground
        for i in range(self.NUM_DRONES):
            if current_positions[i, 2] < 0.05: 
                terminated = True
                if self.gui: print(f"[CRASH] Drone {i} crashed (z={current_positions[i, 2]:.3f})")
                break
                
        # Check for collisions between drones
        if self.NUM_DRONES > 1 and not terminated:
            for i in range(self.NUM_DRONES):
                for j in range(i + 1, self.NUM_DRONES):
                    if np.linalg.norm(current_positions[i] - current_positions[j]) < self.collision_distance:
                        terminated = True
                        if self.gui: print(f"[COLLISION] Drones {i} and {j} collided")
                        break
                if terminated: break
                
        # Check for collisions with obstacles
        # if self.add_obstacles and len(self.obstacles_info) > 0 and not terminated:
        #     for i in range(self.NUM_DRONES):
        #         for obstacle in self.obstacles_info:
        #             obstacle_pos = np.array(obstacle['position'])
        #             obstacle_size = obstacle['size']
        #             # Collision distance depends on obstacle size
        #             collision_threshold = self.collision_distance + obstacle_size
        #             distance_to_obstacle = np.linalg.norm(current_positions[i] - obstacle_pos)
        #             if distance_to_obstacle < collision_threshold:
        #                 terminated = True
        #                 if self.gui: print(f"[OBSTACLE COLLISION] Drone {i} hit {obstacle['type']} at distance {distance_to_obstacle:.3f}")
        #                 break
        #         if terminated: break
                
        return terminated

    def _computeTruncated(self):
        num_control_steps_taken = self.step_counter // self.PYB_STEPS_PER_CTRL if self.PYB_STEPS_PER_CTRL > 0 else self.step_counter
        
        if num_control_steps_taken >= self.max_episode_steps:
            if self.gui: print(f"[TIMEOUT] Episode truncated after {num_control_steps_taken} control steps (max: {self.max_episode_steps})")
            # Gymnasium's new signature for truncated might be just a bool, or (bool, dict) for info.
            # For SB3, usually just returning True is sufficient.
            return True 
        return False

    def _computeInfo(self):
        current_positions = self.pos 
        if self.current_targets is None:
            distances_to_targets = np.full(self.NUM_DRONES, np.inf)
        else:
            distances_to_targets = np.linalg.norm(current_positions - self.current_targets, axis=1)
        
        if self.targets_reached_flags is None: 
            self.targets_reached_flags = np.zeros(self.NUM_DRONES, dtype=bool)

        episode_success = np.all(self.targets_reached_flags)
        
        # Basic info dictionary
        info = {
            'current_targets': self.current_targets.copy() if self.current_targets is not None else np.array([]),
            'target_radius': self.current_target_radius,
            'total_episodes': self.total_episodes,
            'success_rate_last_100': np.mean(self.episode_results) if len(self.episode_results) > 0 else 0.0,
            'distance_to_targets': distances_to_targets.copy(),
            'min_distance_to_target': np.min(distances_to_targets) if distances_to_targets.size > 0 else np.inf,
            'episode_success': episode_success, 
            'targets_reached_flags': self.targets_reached_flags.copy(),
            'num_targets_reached': np.sum(self.targets_reached_flags),
        }
        
        # Add obstacle information if enabled
        if self.add_obstacles:
            # Check distances to obstacles
            min_obstacle_distances = []
            for i in range(self.NUM_DRONES):
                drone_pos = current_positions[i]
                drone_obstacle_distances = []
                for obs in self.obstacles_info:
                    obstacle_pos = np.array(obs['position'])
                    distance = np.linalg.norm(drone_pos - obstacle_pos)
                    drone_obstacle_distances.append(distance)
                
                min_dist = min(drone_obstacle_distances) if drone_obstacle_distances else np.inf
                min_obstacle_distances.append(min_dist)
            
            info.update({
                'obstacles': self.obstacles_info,
                'num_obstacles': len(self.obstacles_info),
                'min_obstacle_distances': np.array(min_obstacle_distances),
                'min_obstacle_distance': np.min(min_obstacle_distances) if min_obstacle_distances else np.inf
            })
            
        return info

# Test the environment
if __name__ == "__main__": # pragma: no cover
    print("Testing MultiTargetAviary with KIN_DEPTH observations and obstacles...")
    try:
        from gym_pybullet_drones.utils.enums import ObservationType
        if not hasattr(ObservationType, 'KIN_DEPTH'): raise ImportError("KIN_DEPTH not defined in imported ObservationType")
    except ImportError:
        from enum import Enum
        class ObservationType(Enum): KIN = "kin"; RGB = "rgb"; KIN_DEPTH = "kin_depth"

    env = MultiTargetAviary(
        num_drones=1, obs=ObservationType.KIN_DEPTH, act=ActionType.RPM,
        gui=True, record=False, episode_length_sec=3.0, ctrl_freq=30,
        target_radius_start=0.2, target_radius_max=1.0, evaluation_window=5,
        add_obstacles=True, obs_prob=0.5, obstacle_size=0.1)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    assert isinstance(env.observation_space, spaces.Dict)
    assert "kin" in env.observation_space.spaces and "depth" in env.observation_space.spaces
    print(f"KIN obs space shape: {env.observation_space.spaces['kin'].shape}")
    print(f"Depth obs space shape: {env.observation_space.spaces['depth'].shape}")

    for episode in range(2):
        obs, info = env.reset()
        print(f"\n=== Episode {episode + 1}, Target: {info['current_targets'][0]}, Radius: {info['target_radius']:.2f} ===")
        print(f"Number of obstacles: {info.get('num_obstacles', 0)}")
        episode_reward = 0
        max_steps_for_ep = env.max_episode_steps # Use the calculated attribute
        for step_num in range(max_steps_for_ep + 5): # Iterate up to max_steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action) # Gymnasium returns bool for truncated
            episode_reward += reward
            done = terminated or truncated
            if step_num % 30 == 0 or done:
                print(f"  Step {step_num:3d} | Dist to target: {info.get('min_distance_to_target', -1):.3f}")
                if 'min_obstacle_distance' in info:
                    print(f"  Min dist to obstacle: {info.get('min_obstacle_distance', -1):.3f}")
                print(f"  Reward: {reward:6.2f} | Term: {terminated} | Trunc: {truncated} | Success: {info.get('episode_success', False)}")
            if done:
                print(f"Episode ended. Total reward: {episode_reward:.2f}. Success: {info.get('episode_success')}")
                break
    env.close()
    print("\nMultiTargetAviary with obstacles test completed!")