import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
# Assuming enums are in this path or defined elsewhere accessible
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment."""
        
        self.ACTION_BUFFER_SIZE = 2 #2 #min(int(ctrl_freq // 2), 6) if ctrl_freq >=2 else 1 # ensure > 0
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        
        self.OBS_TYPE = obs
        self.ACT_TYPE = act

        # Determine if vision attributes are needed
        vision_attributes = True if obs == ObservationType.RGB or obs == ObservationType.KIN_DEPTH else False

        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True if obs==ObservationType.RGB else False, 
                         user_debug_gui=False, 
                         vision_attributes=vision_attributes 
                         )
        
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

        # Initialize action buffer for KIN obs type (must be after super init for NUM_DRONES)
        if self.OBS_TYPE == ObservationType.KIN or self.OBS_TYPE == ObservationType.KIN_DEPTH:
             # Determine action size for buffer initialization
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                action_size_for_buffer = 4
            elif self.ACT_TYPE == ActionType.PID:
                action_size_for_buffer = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                action_size_for_buffer = 1
            else: # Default or error
                action_size_for_buffer = 4 
                print(f"[WARNING] in BaseRLAviary.__init__(), unrecognized action type {self.ACT_TYPE} for buffer init, defaulting to size 4.")

            for _ in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, action_size_for_buffer)))

    ################################################################################
    def _addObstacles(self):
        """Add obstacles to the environment."""
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf", [1, 0, .1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.CLIENT)
            p.loadURDF("cube_small.urdf", [0, 1, .1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.CLIENT)
            p.loadURDF("duck_vhacd.urdf", [-1, 0, .1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.CLIENT)
            p.loadURDF("teddy_vhacd.urdf", [0, -1, .1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.CLIENT)
        else:
            pass # No obstacles for KIN or KIN_DEPTH unless explicitly added by subclass

    ################################################################################
    def _actionSpace(self):
        """Returns the action space of the environment."""
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        
        # For NUM_DRONES=1, SB3 expects Box(low=-1, high=1, shape=(size,), dtype=np.float32)
        # For NUM_DRONES>1, it's often Box(low=-1, high=1, shape=(NUM_DRONES, size), dtype=np.float32)
        # The provided training script uses extractors that expect (NUM_DRONES, size)
        # However, if NUM_DRONES=1, using shape (size,) is more standard for SB3 single agent.
        if self.NUM_DRONES == 1:
            act_lower_bound = np.array([-1.0] * size, dtype=np.float32)
            act_upper_bound = np.array([+1.0] * size, dtype=np.float32)
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        else: # NUM_DRONES > 1
            act_lower_bound = np.array([[-1.0] * size for _ in range(self.NUM_DRONES)], dtype=np.float32)
            act_upper_bound = np.array([[+1.0] * size for _ in range(self.NUM_DRONES)], dtype=np.float32)
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)


    ################################################################################
    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs."""
        # print("Action received by _preprocessAction:", action)
        # print("Action shape:", action.shape if isinstance(action, np.ndarray) else "N/A")
        # print("NUM_DRONES:", self.NUM_DRONES)
        # print("ACT_TYPE:", self.ACT_TYPE)

        # If NUM_DRONES is 1, action might be (size,) instead of (1, size)
        # We need to reshape it to (NUM_DRONES, size) for the loop and buffer
        if self.NUM_DRONES == 1 and isinstance(action, np.ndarray) and action.ndim == 1:
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL] and len(action) == 4:
                action = action.reshape(1, 4)
            elif self.ACT_TYPE == ActionType.PID and len(action) == 3:
                action = action.reshape(1, 3)
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID] and len(action) == 1:
                 action = action.reshape(1, 1)
            # else:
            #     print(f"[WARNING] Action shape {action.shape} for NUM_DRONES=1 and ACT_TYPE={self.ACT_TYPE} not automatically reshaped.")


        self.action_buffer.append(action) # Action should be (NUM_DRONES, action_size)
        rpm = np.zeros((self.NUM_DRONES,4))
        
        for k in range(self.NUM_DRONES): # action.shape[0] should be NUM_DRONES
            target = action[k, :] 
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                          cur_pos=state[0:3], cur_quat=state[3:7],
                                                          cur_vel=state[10:13], cur_ang_vel=state[13:16],
                                                          target_pos=next_pos)
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                         cur_pos=state[0:3], cur_quat=state[3:7],
                                                         cur_vel=state[10:13], cur_ang_vel=state[13:16],
                                                         target_pos=state[0:3], 
                                                         target_rpy=np.array([0,0,state[9]]),
                                                         target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector)
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3], cur_quat=state[3:7],
                                                        cur_vel=state[10:13], cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]]))
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################
    def _observationSpace(self):
        """Returns the observation space of the environment."""
        if self.OBS_TYPE == ObservationType.RGB:
            if self.NUM_DRONES == 1:
                 return spaces.Box(low=0, high=255, shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
            else:
                 return spaces.Box(low=0, high=255, shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        
        elif self.OBS_TYPE == ObservationType.KIN:
            # Kinematic observation space for one drone (12 features)
            # X, Y, Z, R, P, Y, VX, VY, VZ, WX, WY, WZ
            kin_obs_size_per_drone = 12 
            lo = -np.inf
            hi = np.inf
            
            # Base KIN obs bounds for one drone
            drone_kin_low = np.array([lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo]) 
            drone_kin_high = np.array([hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi])

            # Determine action size for action buffer part of obs
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]: action_obs_size = 4
            elif self.ACT_TYPE == ActionType.PID: action_obs_size = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]: action_obs_size = 1
            else: action_obs_size = 4 # Default

            total_kin_obs_size_per_drone = kin_obs_size_per_drone + (self.ACTION_BUFFER_SIZE * action_obs_size)
            
            obs_lower_b = np.full(total_kin_obs_size_per_drone, lo)
            obs_upper_b = np.full(total_kin_obs_size_per_drone, hi)
            obs_lower_b[:kin_obs_size_per_drone] = drone_kin_low # Set specific bounds for base kin
            obs_upper_b[:kin_obs_size_per_drone] = drone_kin_high
            obs_lower_b[kin_obs_size_per_drone:] = -1 # Action buffer part is normalized actions
            obs_upper_b[kin_obs_size_per_drone:] = +1
            
            if self.NUM_DRONES == 1:
                return spaces.Box(low=obs_lower_b, high=obs_upper_b, dtype=np.float32)
            else: # NUM_DRONES > 1
                # Stack for multiple drones
                stacked_lower = np.tile(obs_lower_b, (self.NUM_DRONES, 1))
                stacked_upper = np.tile(obs_upper_b, (self.NUM_DRONES, 1))
                return spaces.Box(low=stacked_lower, high=stacked_upper, dtype=np.float32)

        elif self.OBS_TYPE == ObservationType.KIN_DEPTH:
            # KIN part (same as ObservationType.KIN for a single drone)
            kin_obs_size_per_drone = 12 
            lo = -np.inf
            hi = np.inf
            drone_kin_low = np.array([lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo]) 
            drone_kin_high = np.array([hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi])

            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]: action_obs_size = 4
            elif self.ACT_TYPE == ActionType.PID: action_obs_size = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]: action_obs_size = 1
            else: action_obs_size = 4 

            total_kin_features_per_drone = kin_obs_size_per_drone + (self.ACTION_BUFFER_SIZE * action_obs_size)
            
            kin_low_bounds = np.full(total_kin_features_per_drone, lo)
            kin_high_bounds = np.full(total_kin_features_per_drone, hi)
            kin_low_bounds[:kin_obs_size_per_drone] = drone_kin_low
            kin_high_bounds[:kin_obs_size_per_drone] = drone_kin_high
            kin_low_bounds[kin_obs_size_per_drone:] = -1 # Action buffer part
            kin_high_bounds[kin_obs_size_per_drone:] = +1

            kin_space = spaces.Box(low=kin_low_bounds, high=kin_high_bounds, dtype=np.float32)
            
            # Depth part (H, W, C) - C=1 for depth
            # Using float for depth as in VisionAviary example and BaseAviary storage
            depth_img_shape = (self.IMG_RES[1], self.IMG_RES[0], 1)
            depth_space = spaces.Box(low=0.01, high=1000., shape=depth_img_shape, dtype=np.float32)

            if self.NUM_DRONES == 1:
                return spaces.Dict({
                    "kin": kin_space,
                    "depth": depth_space
                })
            else: # NUM_DRONES > 1 : Each key in dict will have NUM_DRONES as first dimension
                return spaces.Dict({
                    "kin": spaces.Box(low=np.tile(kin_low_bounds, (self.NUM_DRONES,1)), 
                                      high=np.tile(kin_high_bounds, (self.NUM_DRONES,1)), 
                                      dtype=np.float32),
                    "depth": spaces.Box(low=0.01, high=1000., 
                                        shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 1), 
                                        dtype=np.float32)
                })
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
            exit()
    
    ################################################################################
    def _computeObs(self):
        """Returns the current observation of the environment."""
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i, segmentation=False)
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB, img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))
            if self.NUM_DRONES == 1:
                return self.rgb[0].astype('float32') # Or uint8 if specified by space
            else:
                return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')

        elif self.OBS_TYPE == ObservationType.KIN:
            kin_obs_size_per_drone = 12
            obs_all_drones = []
            for i in range(self.NUM_DRONES):
                state_vec = self._getDroneStateVector(i) # (20,)
                # Extract 12 basic KIN features: X,Y,Z, R,P,Y, Vx,Vy,Vz, Wx,Wy,Wz
                kin_12_features = np.hstack([state_vec[0:3], state_vec[7:10], state_vec[10:13], state_vec[13:16]]).reshape(kin_obs_size_per_drone,)
                
                # Add action buffer
                action_buffer_features = []
                for buf_idx in range(self.ACTION_BUFFER_SIZE):
                    action_buffer_features.extend(self.action_buffer[buf_idx][i, :])
                
                obs_drone_i = np.concatenate([kin_12_features, np.array(action_buffer_features)]).astype('float32')
                obs_all_drones.append(obs_drone_i)
            
            if self.NUM_DRONES == 1:
                return obs_all_drones[0]
            else:
                return np.array(obs_all_drones).astype('float32')

        elif self.OBS_TYPE == ObservationType.KIN_DEPTH:
            kin_obs_list = []
            depth_obs_list = []
            kin_obs_size_per_drone = 12

            for i in range(self.NUM_DRONES):
                # KIN part
                state_vec = self._getDroneStateVector(i)
                kin_12_features = np.hstack([state_vec[0:3], state_vec[7:10], state_vec[10:13], state_vec[13:16]]).reshape(kin_obs_size_per_drone,)
                
                action_buffer_features = []
                for buf_idx in range(self.ACTION_BUFFER_SIZE):
                    action_buffer_features.extend(self.action_buffer[buf_idx][i, :])
                
                kin_obs_drone_i = np.concatenate([kin_12_features, np.array(action_buffer_features)]).astype('float32')
                kin_obs_list.append(kin_obs_drone_i)

                # Depth part
                if self.step_counter % self.IMG_CAPTURE_FREQ == 0 or self.dep[i] is None or np.sum(self.dep[i]) == self.dep[i].size : # check if dep is all ones (initial)
                    # self.dep is (NUM_DRONES, H, W)
                    _, drone_depth_img, _ = self._getDroneImages(i, segmentation=False) # returns rgb, dep, seg for drone i
                    self.dep[i] = drone_depth_img # Update stored depth
                
                depth_obs_list.append(self.dep[i].reshape(self.IMG_RES[1], self.IMG_RES[0], 1).astype('float32'))

            final_kin_obs = np.array(kin_obs_list)
            final_depth_obs = np.array(depth_obs_list)

            if self.NUM_DRONES == 1:
                return {
                    "kin": final_kin_obs[0],
                    "depth": final_depth_obs[0]
                }
            else:
                 return {
                    "kin": final_kin_obs,
                    "depth": final_depth_obs
                }
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
            exit()