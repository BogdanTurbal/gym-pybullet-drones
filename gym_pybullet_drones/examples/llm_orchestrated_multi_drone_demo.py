"""
llm_orchestrated_multi_drone_demo.py:
Demonstrates N drones, each controlled by a single pre-trained RL policy.
An LLM (Gemini API) orchestrates the drones by setting their targets
based on human commands. Command input is captured from terminal after
clicking a GUI button to enter command mode.
Observations are KINEMATIC. The demo will continue for the set duration,
resetting the environment if episodes terminate internally, unless 'quit' is issued.
"""
import os
import time
import argparse
import numpy as np
import gymnasium as gym 
from datetime import datetime
import json 
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque  # Added for history tracking


# Global variables for history tracking
HISTORY_MAX_ENTRIES = 10  # Maximum number of historical states to store
drone_state_history = deque(maxlen=HISTORY_MAX_ENTRIES)  # Will store tuples of (command, states, targets, timestamp)


def request_llm_targets_async(current_human_command: str, drone_states: list, num_drones: int, current_targets: np.ndarray):
    """
    Queues an asynchronous request to get targets from the Gemini API.
    Does not block the main thread.
    """
    global llm_request_queue
    
    # Create request data
    request_data = {
        "command": current_human_command,
        "drone_states": drone_states,
        "num_drones": num_drones,
        "current_targets": current_targets.copy()
    }
    
    # Add request to the queue
    llm_request_queue.put(request_data)
    print(f"[LLM_ASYNC] Queued request for command: '{current_human_command[:50]}...'")


def command_input_thread():
    """Thread function to get command input from terminal"""
    global command_queue, waiting_for_command, stop_simulation
    
    while not stop_simulation:
        if waiting_for_command:
            try:
                command = input("\n[TERMINAL_INPUT] Enter command: ")
                if command.lower() == "quit":
                    stop_simulation = True
                command_queue.put(command)
                waiting_for_command = False
            except EOFError:
                # Handle Ctrl+D or other EOF condition
                waiting_for_command = False
                command_queue.put("cancel")
            except KeyboardInterrupt:
                # Handle Ctrl+C
                waiting_for_command = False
                command_queue.put("cancel")
                stop_simulation = True
        else:
            time.sleep(0.1)  # Prevent CPU thrashing when not active#!/usr/bin/env python3

# --- Gemini API Setup ---
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") 
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or GEMINI_API_KEY is None:
        print("[WARNING] Gemini API key not set. LLM functionality will be disabled.")
        print("           Please set the GEMINI_API_KEY environment variable.")
        gemini_available = False
        gemini_model = None
    else:
        ##ohh
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-lite',#'gemini-1.5-flash', 
            generation_config={"response_mime_type": "application/json"} 
        ) 
        gemini_available = True
        print("[INFO] Gemini API configured successfully.")
except ImportError:
    print("[WARNING] Google Generative AI SDK not installed. LLM functionality will be disabled.")
    print("           Please install with: pip install google-generativeai")
    gemini_available = False
    gemini_model = None
except Exception as e:
    print(f"[WARNING] Error initializing Gemini API: {e}. LLM functionality will be disabled.")
    gemini_available = False
    gemini_model = None
# --- End Gemini API Setup ---

from stable_baselines3 import PPO, TD3, SAC, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy

from multi_agent_extractors_td3 import create_multiagent_model 
from gym_pybullet_drones.envs.MultiTargetAviary import MultiTargetAviary 
from gym_pybullet_drones.utils.Logger import Logger 
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics
import pybullet as p 

# Default settings for the demo
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'llm_demo_results_kin'
DEFAULT_OBS = ObservationType.KIN
DEFAULT_ACT = ActionType.RPM      
DEFAULT_NUM_DRONES = 4            
DEFAULT_DURATION_SEC = 1000.0       
DEFAULT_CTRL_FREQ = 20         
LLM_UPDATE_INTERVAL_SEC = 20.0     
DEFAULT_ADD_OBSTACLES = False      
DEFAULT_OBS_PROB = 0.5            
DEFAULT_OBSTACLE_SIZE = 0.2       
DEFAULT_ASYNC_LLM = True          # Default to asynchronous LLM processing

# Global variables for command input and LLM processing
human_command = "Drones, hover at 1 meter height at your current XY locations."
command_queue = queue.Queue()
llm_request_queue = queue.Queue()
llm_response_queue = queue.Queue()
stop_simulation = False 
waiting_for_command = False
llm_processing = False
last_llm_request_time = 0  # Timestamp of the last LLM request
next_llm_update_time = 0   # Time when next LLM update should happen

# PyBullet Key Constants
KEY_ENTER = p.B3G_RETURN      # Usually 65293
KEY_BACKSPACE = p.B3G_BACKSPACE # Usually 65288
KEY_ESCAPE = 65307            # Standard X11 keycode for Escape

def llm_processing_thread():
    """Thread function to process LLM requests asynchronously"""
    global llm_request_queue, llm_response_queue, stop_simulation, llm_processing
    global last_llm_request_time, next_llm_update_time
    
    while not stop_simulation:
        try:
            if not llm_request_queue.empty():
                # Get the request from the queue
                request_data = llm_request_queue.get_nowait()
                llm_processing = True
                last_llm_request_time = time.time()
                
                # Extract request parameters
                current_human_command = request_data["command"]
                drone_states = request_data["drone_states"]
                num_drones = request_data["num_drones"]
                current_targets = request_data["current_targets"]
                
                # Process the request
                new_targets = get_llm_targets_from_gemini(
                    current_human_command, drone_states, num_drones, current_targets
                )
                
                # Record response time and calculate next update time
                response_time = time.time() - last_llm_request_time
                print(f"[LLM_TIMING] Response received in {response_time:.2f} seconds")
                
                # Put the response in the response queue
                llm_response_queue.put(new_targets)
                llm_processing = False
                
                # Indicate that processing is complete for this request
                llm_request_queue.task_done()
            else:
                # Sleep briefly to prevent CPU thrashing
                time.sleep(0.05)
        except Exception as e:
            print(f"[LLM_THREAD_ERROR] Error processing LLM request: {e}")
            import traceback
            traceback.print_exc()
            llm_processing = False
            time.sleep(0.1)  # Brief pause after error


def format_history_for_prompt():
    """Format the drone state history for inclusion in the LLM prompt"""
    if not drone_state_history:
        return "No previous history available.\n"
    
    history_text = "### Drone Command History:\n"
    
    for idx, (timestamp, cmd, states, targets) in enumerate(drone_state_history):
        history_text += f"\n## Event {idx+1} (Time: {timestamp:.2f}s):\n"
        history_text += f"Command: \"{cmd}\"\n"
        history_text += "Drone States:\n"
        
        for i, state in enumerate(states):
            pos_str = ", ".join([f"{val:.2f}" for val in state['position']])
            vel_str = ", ".join([f"{val:.2f}" for val in state['velocity']])
            tgt_str = ", ".join([f"{val:.2f}" for val in targets[i]])
            history_text += f"- Drone {state['id']}: Position=[{pos_str}], Velocity=[{vel_str}], Target=[{tgt_str}]\n"
        
    return history_text


def get_llm_targets_from_gemini(current_human_command: str, drone_states: list, num_drones: int, current_targets: np.ndarray):
    """
    Queries the Gemini API to get new targets for each drone.
    Now includes history of previous states and commands.
    """
    if not gemini_available or gemini_model is None:
        print("[LLM_FALLBACK] Gemini API not available. Drones will attempt to hover or use previous targets.")
        new_targets_fallback = []
        for i in range(num_drones):
            if drone_states[i] and 'position' in drone_states[i]:
                new_targets_fallback.append([
                    drone_states[i]['position'][0], 
                    drone_states[i]['position'][1], 
                    max(0.5, drone_states[i]['position'][2] + 0.1) 
                ])
            elif current_targets is not None and i < len(current_targets):
                new_targets_fallback.append(current_targets[i])
            else: 
                new_targets_fallback.append([i*0.5, 0, 1.0]) 
        return np.array(new_targets_fallback, dtype=np.float32)

    # Record current state in history before making the LLM call
    # This happens inside the get_llm_targets function to ensure we record the state at the exact moment of the LLM call
    current_time = time.time()
    if 'sim_start_time' in globals():
        elapsed_time = current_time - sim_start_time
    else:
        elapsed_time = 0.0
    
    # Add current state to history
    drone_state_history.append((
        elapsed_time,
        current_human_command,
        drone_states.copy(),
        current_targets.copy()
    ))
    
    # Get formatted history text
    history_text = format_history_for_prompt()

    prompt = f"""You are an AI orchestrator for a swarm of {num_drones} drones.
Human command: "{current_human_command}"

Current drone states (position is [x,y,z], velocity is [vx,vy,vz], current target is [tx,ty,tz]):
"""
    for i, state in enumerate(drone_states):
        pos_str = ", ".join([f"{val:.2f}" for val in state['position']])
        vel_str = ", ".join([f"{val:.2f}" for val in state['velocity']])
        ct_str = ", ".join([f"{val:.2f}" for val in current_targets[i]])
        prompt += f"Drone {state['id']}: Position=[{pos_str}], Velocity=[{vel_str}], Current Target=[{ct_str}]\n"

    # Add history to the prompt
    prompt += f"""
{history_text}

Based on the human command, current drone states, and command history, provide new 3D target coordinates (x, y, z) for each of the {num_drones} drones.
The simulation environment is roughly a 10m x 10m x 5m box centered at (0,0,1.0).
Ensure targets are within a reasonable flight envelope (e.g., x, y between -4 to 4; z between 0.2 to 4).
Drones should avoid collisions with each other. If a formation is requested, try to maintain a safe inter-drone distance (e.g., > 0.5m apart).
If the command is to move, provide intermediate waypoints if the final destination is far.
Your response MUST be a JSON object with a single key "targets".
The value of "targets" MUST be a list of lists, where each inner list contains exactly three float numbers [x, y, z] representing the target for one drone.
The order of targets in the list MUST correspond to the drone IDs (Drone 0, Drone 1, ..., Drone N-1).

Example for 2 drones:
{{
  "targets": [
    [1.0, 1.0, 1.5],
    [-1.0, 1.0, 1.5]
  ]
}}

Now, generate the targets:
"""
    # Uncomment to debug the prompt with history
    # print(f"\n[LLM_PROMPT_WITH_HISTORY]\n{prompt}") 

    try:
        response = gemini_model.generate_content(prompt)
        
        parsed_response = json.loads(response.text)
        new_targets_list = parsed_response.get("targets")

        if not isinstance(new_targets_list, list) or len(new_targets_list) != num_drones:
            print(f"[LLM_ERROR] Response 'targets' is not a list of length {num_drones}. Response: {response.text}")
            return current_targets 

        new_targets_np = np.zeros((num_drones, 3), dtype=np.float32)
        valid_targets = True
        for i in range(num_drones):
            if isinstance(new_targets_list[i], list) and len(new_targets_list[i]) == 3:
                try:
                    x = np.clip(float(new_targets_list[i][0]), -4.5, 4.5)
                    y = np.clip(float(new_targets_list[i][1]), -4.5, 4.5)
                    z = np.clip(float(new_targets_list[i][2]), 0.15, 4.0) 
                    new_targets_np[i] = [x, y, z]
                except ValueError:
                    print(f"[LLM_ERROR] Could not convert target for drone {i} to floats: {new_targets_list[i]}")
                    valid_targets = False; break
            else:
                print(f"[LLM_ERROR] Target for drone {i} is not a list of 3 numbers: {new_targets_list[i]}")
                valid_targets = False; break
        
        if valid_targets:
            print(f"[LLM_SUCCESS] New targets received and clipped: {new_targets_np.tolist()}")
            return new_targets_np
        else:
            print("[LLM_ERROR] Invalid target format from LLM. Using previous targets.")
            return current_targets 

    except json.JSONDecodeError as e:
        print(f"[LLM_ERROR] Failed to parse JSON response: {e}. Raw response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'No response text available'}")
        return current_targets 
    except Exception as e:
        print(f"[LLM_ERROR] Error during Gemini API call or processing: {e}")
        import traceback
        traceback.print_exc()
        return current_targets 


def detect_algorithm_from_model(model_path, env_for_load):
    """Detect which algorithm was used for training from model file, requires an env instance."""
    algorithms_to_try = {
        'ppo': PPO, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG
    }
    for algo_name, algo_class in algorithms_to_try.items():
        try:
            custom_objects = {
                 "observation_space": env_for_load.observation_space,
                 "action_space": env_for_load.action_space,
            }
            model = algo_class.load(model_path, env=None, custom_objects=custom_objects, print_system_info=False)
            print(f"[INFO] Detected algorithm: {algo_name.upper()}")
            return algo_name, model
        except Exception:
            continue
    print(f"[ERROR] Could not load model with any of the supported algorithms: {list(algorithms_to_try.keys())}")
    raise RuntimeError(f"Failed to load model: {model_path}")

def run_llm_orchestrated_demonstration(
        model_path: str, num_drones: int, obs_type: ObservationType, act_type: ActionType,
        output_folder: str, gui: bool, record_video: bool, ctrl_freq: int,
        duration_sec: float, llm_update_interval_sec: float, plot: bool,
        add_obstacles: bool, obs_prob: float, obstacle_size: float,
        async_llm: bool = False  # New parameter to control LLM processing mode
    ):
    global human_command, stop_simulation, waiting_for_command, command_queue, llm_response_queue
    global sim_start_time, drone_state_history, llm_processing, last_llm_request_time, next_llm_update_time

    # Clear history at the start of a new demonstration
    drone_state_history.clear()
    
    print(f"[SETUP] Model Path: {model_path}")
    print(f"[SETUP] LLM Mode: {'Asynchronous' if async_llm else 'Synchronous'}")
    # ... (rest of print statements) ...

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    demo_folder_name = f"llm_demo_{obs_type.value}_{num_drones}drones_{timestamp}"
    current_output_folder = os.path.join(output_folder, demo_folder_name)
    os.makedirs(current_output_folder, exist_ok=True)

    env_params = {
        'drone_model': DroneModel.CF2X, 'num_drones': num_drones,
        'initial_xyzs': None, 'initial_rpys': None, 'physics': Physics.PYB,
        'pyb_freq': 240, 'ctrl_freq': ctrl_freq, 'gui': gui, 'record': record_video,
        'obs': obs_type, 'act': act_type, 'episode_length_sec': duration_sec, 
        'target_radius_start': 0.5, 'target_radius_max': 3.0,   
        'target_radius_increment': 0.1, 'target_tolerance': 0.15, 
        'add_obstacles': add_obstacles, 'obs_prob': obs_prob,
        'obstacle_size': obstacle_size, "never_end": True
    }

    env = MultiTargetAviary(**env_params)
    print(f"[INFO] MultiTargetAviary initialized for {num_drones} drones.")
    print(f"[INFO] Env Observation Space: {env.observation_space}")
    print(f"[INFO] Env Action Space: {env.action_space}")

    llm_gui_button = -1
    last_llm_gui_button_press_count = 0 
    llm_current_cmd_display_text_id = -1
    llm_status_text_id = -1  # For displaying LLM status (async mode only)

    if gui:
        llm_gui_button = p.addUserDebugParameter("Enter New Command", 1, 0, 0, physicsClientId=env.CLIENT)
        llm_current_cmd_display_text_id = p.addUserDebugText(
            f"Current Command: {human_command}", textPosition=[0, -1.5, 2.0], textColorRGB=[0.8, 0.8, 0.2], 
            textSize=1.2, physicsClientId=env.CLIENT
        )
        
        if async_llm:
            llm_status_text_id = p.addUserDebugText(
                "LLM Status: Ready", textPosition=[0, -1.8, 2.0], textColorRGB=[0.2, 0.8, 0.2], 
                textSize=1.0, physicsClientId=env.CLIENT
            )

    # Start command input thread
    input_thread = threading.Thread(target=command_input_thread, daemon=True)
    input_thread.start()
    
    # Start LLM processing thread (always needed for async mode, optional for sync mode)
    if async_llm:
        llm_thread = threading.Thread(target=llm_processing_thread, daemon=True)
        llm_thread.start()
        print("[INFO] Started asynchronous LLM processing thread")
    else:
        # In synchronous mode, we might still want the thread for handling user command inputs
        llm_thread = threading.Thread(target=llm_processing_thread, daemon=True)
        llm_thread.start()
        print("[INFO] Started LLM processing thread for command handling")

    try:
        print("[INFO] Creating temporary single-drone environment for model loading...")
        single_drone_env_params = env_params.copy()
        single_drone_env_params['num_drones'] = 1 
        single_drone_env_params['gui'] = False 
        single_drone_env_params['record'] = False
        single_drone_env_params['obs'] = obs_type 
        temp_env_for_load = MultiTargetAviary(**single_drone_env_params)
        
        algo_name, model = detect_algorithm_from_model(model_path, temp_env_for_load)
        print(f"[INFO] RL Model {model_path} loaded successfully using algorithm: {algo_name.upper()}.")
        temp_env_for_load.close() 
    except Exception as e:
        print(f"[ERROR] Failed to load RL model: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return

    obs, info = env.reset(seed=int(time.time())) 
    
    print("[INFO] Making initial LLM call for targets...")
    drone_states_for_llm = [{'id': i, 'position': env.pos[i].tolist(), 'velocity': env.vel[i].tolist()} for i in range(num_drones)]
    
    current_cmd_for_llm = human_command 
    
    # Record simulation start time for history timestamps
    sim_start_time = time.time()
    
    if current_cmd_for_llm.lower() != "quit":
        if async_llm:
            # In async mode, queue the initial request and continue (will use default targets until response arrives)
            print("[INFO] Queuing initial LLM request asynchronously")
            request_llm_targets_async(current_cmd_for_llm, drone_states_for_llm, num_drones, env.current_targets.copy())
            if gui and llm_status_text_id != -1:
                p.addUserDebugText("LLM Status: Processing Initial Command", textPosition=[0, -1.8, 2.0], 
                                  textColorRGB=[0.8, 0.2, 0.2], textSize=1.0, 
                                  physicsClientId=env.CLIENT, replaceItemUniqueId=llm_status_text_id)
        else:
            # In sync mode, make the initial call synchronously to ensure we have targets before starting
            print("[INFO] Making initial LLM request synchronously")
            env.current_targets = get_llm_targets_from_gemini(
                current_cmd_for_llm, drone_states_for_llm, num_drones, env.current_targets.copy()
            )
        
        obs = env._computeObs() 
        info = env._computeInfo() 
    
    if not async_llm:
        print(f"[INFO] Initial targets from LLM: {env.current_targets.tolist()}")

    if gui and llm_current_cmd_display_text_id != -1: 
         p.addUserDebugText(f"Current Command: {human_command}", textPosition=[0, -1.5, 2.0], textColorRGB=[0.8, 0.8, 0.2],
                            textSize=1.2, physicsClientId=env.CLIENT, replaceItemUniqueId=llm_current_cmd_display_text_id)

    logger = Logger(logging_freq_hz=ctrl_freq, num_drones=num_drones, output_folder=current_output_folder)
    
    last_llm_update_time = time.time()
    next_llm_update_time = last_llm_update_time + llm_update_interval_sec
    total_steps_taken = 0
    max_sim_steps = int(duration_sec * ctrl_freq)

    # Also create a history log file
    history_log_path = os.path.join(current_output_folder, "drone_history.json")
    
    try:
        for step_counter in range(max_sim_steps):
            total_steps_taken = step_counter
            if stop_simulation: 
                print("[INFO] Stop signal received in main loop.")
                break

            # Check GUI button for command input
            if gui and llm_gui_button != -1:
                button_val = p.readUserDebugParameter(llm_gui_button, physicsClientId=env.CLIENT)
                if button_val > last_llm_gui_button_press_count: 
                    last_llm_gui_button_press_count = button_val
                    waiting_for_command = True
                    print("\n[GUI_BUTTON_PRESSED] Ready for command input in terminal.")

            # Check if there's a new command from the input thread
            try:
                if not command_queue.empty():
                    new_command = command_queue.get_nowait()
                    if new_command != "cancel":
                        human_command = new_command
                        if llm_current_cmd_display_text_id != -1:
                            p.addUserDebugText(f"Current Command: {human_command}", textPosition=[0, -1.5, 2.0], 
                                              textColorRGB=[0.8, 0.8, 0.2], textSize=1.2, 
                                              physicsClientId=env.CLIENT, replaceItemUniqueId=llm_current_cmd_display_text_id)
                        print(f"[NEW_COMMAND] Received: '{human_command}'")
                        
                        # Process the command based on mode
                        drone_states_for_llm = [{'id': d_idx, 'position': env.pos[d_idx].tolist(), 'velocity': env.vel[d_idx].tolist()} 
                                               for d_idx in range(num_drones)]
                        
                        if async_llm:
                            # In async mode, queue request and update GUI status
                            request_llm_targets_async(human_command, drone_states_for_llm, num_drones, env.current_targets)
                            if gui and llm_status_text_id != -1:
                                p.addUserDebugText("LLM Status: Processing Command", textPosition=[0, -1.8, 2.0], 
                                                  textColorRGB=[0.8, 0.2, 0.2], textSize=1.0, 
                                                  physicsClientId=env.CLIENT, replaceItemUniqueId=llm_status_text_id)
                        else:
                            # In sync mode, make the call synchronously
                            print("[INFO] Processing command synchronously")
                            env.current_targets = get_llm_targets_from_gemini(
                                human_command, drone_states_for_llm, num_drones, env.current_targets.copy()
                            )
                            print(f"[INFO] New targets from LLM: {env.current_targets.tolist()}")
                            # Re-compute observations after target update
                            obs = env._computeObs()
                            info = env._computeInfo()
                    else:
                        print("[COMMAND_CANCELLED] Command input cancelled.")
            except queue.Empty:
                pass
                
            # For async mode: check if there's a response from the LLM thread
            if async_llm:
                try:
                    if not llm_response_queue.empty():
                        new_targets = llm_response_queue.get_nowait()
                        env.current_targets = new_targets
                        print(f"[LLM_ASYNC_RESPONSE] Received new targets: {new_targets.tolist()}")
                        # Update next LLM request time to be current time + interval
                        next_llm_update_time = time.time() + llm_update_interval_sec
                        # Re-compute observations after target update
                        obs = env._computeObs()
                        info = env._computeInfo()
                        
                        # Update GUI status in async mode
                        if gui and llm_status_text_id != -1:
                            p.addUserDebugText("LLM Status: Ready", textPosition=[0, -1.8, 2.0], 
                                              textColorRGB=[0.2, 0.8, 0.2], textSize=1.0, 
                                              physicsClientId=env.CLIENT, replaceItemUniqueId=llm_status_text_id)
                except queue.Empty:
                    pass

            current_time = time.time()
            
            # LLM update logic based on async_llm mode
            if async_llm:
                # In async mode, schedule next LLM request when interval has passed AND not already processing
                if current_time >= next_llm_update_time and not waiting_for_command and not stop_simulation and not llm_processing:
                    print(f"\n--- Async LLM Update Cycle (Step {step_counter}) ---")
                    next_llm_update_time = current_time + llm_update_interval_sec  # Schedule next update
                    last_llm_update_time = current_time
                    
                    drone_states_for_llm = [{'id': d_idx, 'position': env.pos[d_idx].tolist(), 'velocity': env.vel[d_idx].tolist()} 
                                           for d_idx in range(num_drones)]
                    current_cmd_for_llm = human_command 
                    
                    if current_cmd_for_llm.lower() == "quit":
                        print("[INFO] LLM processing 'quit' command. Shutting down simulation.")
                        stop_simulation = True
                        break 
                    
                    # Queue the request asynchronously
                    print(f"[LLM] Sending command: '{current_cmd_for_llm}' with drone states.")
                    request_llm_targets_async(current_cmd_for_llm, drone_states_for_llm, num_drones, env.current_targets)
                    
                    # Update GUI status
                    if gui and llm_status_text_id != -1:
                        p.addUserDebugText("LLM Status: Processing", textPosition=[0, -1.8, 2.0], 
                                          textColorRGB=[0.8, 0.2, 0.2], textSize=1.0, 
                                          physicsClientId=env.CLIENT, replaceItemUniqueId=llm_status_text_id)
                    
                    print(f"--- End Async LLM Update Cycle ---")
            else:
                # Original synchronous LLM update logic
                if (current_time - last_llm_update_time) >= llm_update_interval_sec and not waiting_for_command and not stop_simulation:
                    print(f"\n--- Sync LLM Update Cycle (Step {step_counter}) ---")
                    last_llm_update_time = current_time
                    drone_states_for_llm = [{'id': d_idx, 'position': env.pos[d_idx].tolist(), 'velocity': env.vel[d_idx].tolist()} 
                                           for d_idx in range(num_drones)]
                    current_cmd_for_llm = human_command 
                    
                    if current_cmd_for_llm.lower() == "quit":
                        print("[INFO] LLM processing 'quit' command. Shutting down simulation.")
                        stop_simulation = True
                        break 
                    
                    # Synchronous call - blocks until response is received
                    print(f"[LLM] Processing command: '{current_cmd_for_llm}'")
                    env.current_targets = get_llm_targets_from_gemini(
                        current_cmd_for_llm, drone_states_for_llm, num_drones, env.current_targets
                    )
                    
                    # Re-compute observations after target update
                    obs = env._computeObs()
                    info = env._computeInfo()
                    print(f"--- End Sync LLM Update Cycle ---")

            if len(env.action_space.shape) == 1: 
                action_dim_per_drone = env.action_space.shape[0]
            elif len(env.action_space.shape) == 2: 
                action_dim_per_drone = env.action_space.shape[1]
            else:
                raise ValueError(f"Unexpected action space shape: {env.action_space.shape}")
            actions_for_all_drones = np.zeros((num_drones, action_dim_per_drone))
            
            current_observations_all_drones = env._computeObs() 

            for drone_idx in range(num_drones):
                if obs_type == ObservationType.KIN:
                    obs_for_drone = current_observations_all_drones[drone_idx] if num_drones > 1 else current_observations_all_drones
                elif obs_type == ObservationType.KIN_DEPTH:
                    obs_for_drone = {
                        "kin": current_observations_all_drones["kin"][drone_idx] if num_drones > 1 else current_observations_all_drones["kin"],
                        "depth": current_observations_all_drones["depth"][drone_idx] if num_drones > 1 else current_observations_all_drones["depth"]
                    } if isinstance(current_observations_all_drones, dict) else current_observations_all_drones 
                else: 
                    print(f"[ERROR] Unsupported observation type {obs_type.value} for RL prediction loop.")
                    obs_for_drone = current_observations_all_drones[drone_idx] if num_drones > 1 else current_observations_all_drones

                action_drone, _ = model.predict(obs_for_drone, deterministic=True)
                actions_for_all_drones[drone_idx, :] = action_drone.reshape(1, -1) if action_drone.ndim == 1 else action_drone

            obs, reward, terminated, truncated, info = env.step(actions_for_all_drones)
            terminated = False
            truncated = False
            
            if gui:
                sync(step_counter, sim_start_time, env.CTRL_TIMESTEP) 

            for d_idx in range(num_drones):
                raw_state_vec_drone = env._getDroneStateVector(d_idx)
                logger.log(drone=d_idx, timestamp=step_counter/ctrl_freq, state=raw_state_vec_drone, control=np.zeros(12))

            # Only print status if not waiting for command input
            if step_counter % ctrl_freq == 0 and not waiting_for_command: 
                print(f"\nSim Step: {step_counter:05d}/{max_sim_steps} | Command: '{human_command[:60]}{'...' if len(human_command)>60 else ''}'")
                for d_idx in range(num_drones):
                    dist_to_target = np.linalg.norm(env.pos[d_idx] - env.current_targets[d_idx])
                    print(f"  Drone {d_idx}: Pos: [{env.pos[d_idx,0]:.2f},{env.pos[d_idx,1]:.2f},{env.pos[d_idx,2]:.2f}] | "
                          f"Target: [{env.current_targets[d_idx,0]:.2f},{env.current_targets[d_idx,1]:.2f},{env.current_targets[d_idx,2]:.2f}] | "
                          f"Dist: {dist_to_target:.2f}m")
                if add_obstacles and isinstance(info, dict) and info.get('num_obstacles',0) > 0: 
                     min_obs_dist_val = info.get('min_obstacle_distance', float('inf'))
                     min_obs_dist_str = f"{min_obs_dist_val:.2f}m" if min_obs_dist_val != float('inf') else "N/A"
                     print(f"  Obstacles: {info.get('num_obstacles',0)}. Min Obstacle Dist: {min_obs_dist_str}")
                
                # For async mode, print the LLM status
                if async_llm:
                    llm_status = "Processing" if llm_processing else "Ready"
                    print(f"  LLM Status: {llm_status}")
                    print(f"  Next LLM Update in: {max(0, next_llm_update_time - current_time):.1f}s")

            # --- Continuous Operation Logic ---
            if terminated or truncated: # Environment signals its own episode end
                print(f"[INFO] Environment episode ended at step {step_counter}. Terminated: {terminated}, Truncated: {truncated}")
                if stop_simulation: # User quit the whole demo
                    print("[INFO] Stop simulation signal active. Ending demo.")
                    break 
                
                # Environment's episode ended, but demo continues. Reset env and get new LLM targets.
                print("[INFO] Resetting environment to continue demo...")
                obs, info = env.reset(seed=int(time.time())+step_counter) 
                
                drone_states_for_llm = [{'id': i_drone, 'position': env.pos[i_drone].tolist(), 'velocity': env.vel[i_drone].tolist()} for i_drone in range(num_drones)]
                current_cmd_for_llm = human_command 
                if current_cmd_for_llm.lower() != "quit": 
                    if async_llm:
                        # For async mode, queue request for new targets after reset
                        request_llm_targets_async(current_cmd_for_llm, drone_states_for_llm, num_drones, env.current_targets.copy())
                        if gui and llm_status_text_id != -1:
                            p.addUserDebugText("LLM Status: Processing Reset Command", textPosition=[0, -1.8, 2.0], 
                                              textColorRGB=[0.8, 0.2, 0.2], textSize=1.0, 
                                              physicsClientId=env.CLIENT, replaceItemUniqueId=llm_status_text_id)
                    else:
                        # For sync mode, get targets synchronously after reset
                        env.current_targets = get_llm_targets_from_gemini(current_cmd_for_llm, drone_states_for_llm, num_drones, env.current_targets.copy())
                        print(f"[INFO] Targets after reset from LLM: {env.current_targets.tolist()}")
                    
                    obs = env._computeObs() 
                    info = env._computeInfo() 
                    
                    if gui and llm_current_cmd_display_text_id != -1: 
                        p.addUserDebugText(f"Current Command: {human_command}", textPosition=[0, -1.5, 2.0], textColorRGB=[0.8, 0.8, 0.2],
                                           textSize=1.2, physicsClientId=env.CLIENT, replaceItemUniqueId=llm_current_cmd_display_text_id)
                else: # If 'quit' was somehow set just before this reset
                    stop_simulation = True 
                    break

    except KeyboardInterrupt:
        print("[INFO] Demonstration interrupted by user (Ctrl+C).")
        stop_simulation = True 
    except Exception as e:
        print(f"[ERROR] An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()
        stop_simulation = True 
    finally:
        print("[INFO] Cleaning up...")
        stop_simulation = True 
        waiting_for_command = False  # Stop any pending command input
        
        # Save the drone state history to a file
        try:
            # Convert history to a serializable format
            history_data = []
            for timestamp, cmd, states, targets in drone_state_history:
                history_entry = {
                    "timestamp": timestamp,
                    "command": cmd,
                    "drone_states": states,
                    "targets": targets.tolist() if hasattr(targets, 'tolist') else targets
                }
                history_data.append(history_entry)
                
            with open(history_log_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"[INFO] Drone state history saved to {history_log_path}")
        except Exception as history_e:
            print(f"[WARNING] Error saving drone history: {history_e}")
            
        env.close()
        if logger is not None:
            try:
                logger.save() 
                print(f"[INFO] Trajectory data saved in {current_output_folder}")
                if plot:
                    print("[INFO] Generating trajectory plots...")
                    logger.plot(pwm=False) 
                    print(f"[INFO] Plots saved in {current_output_folder}")
            except Exception as log_e:
                print(f"[WARNING] Error during logger save/plot: {log_e}")
        print(f"[INFO] LLM-Orchestrated Multi-Drone KIN Demo finished. Output: {current_output_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Orchestrated Multi-Drone KIN Demonstration with Terminal Input.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained single-drone RL model (.zip file)')
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones to orchestrate')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Show PyBullet GUI')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Record video of the demo')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder for demo outputs')
    parser.add_argument('--plot', default=True, type=str2bool, help='Generate trajectory plots')
    parser.add_argument('--ctrl_freq', default=DEFAULT_CTRL_FREQ, type=int, help="Control frequency")
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=float, help="Max duration of the demo in seconds")
    parser.add_argument('--llm_update_interval', default=LLM_UPDATE_INTERVAL_SEC, type=float, help="LLM target update interval in seconds")
    parser.add_argument('--history_max_entries', default=HISTORY_MAX_ENTRIES, type=int, help="Maximum number of historical states to store")
    
    parser.add_argument('--add_obstacles', default=DEFAULT_ADD_OBSTACLES, type=str2bool, help='Enable obstacle generation in environment')
    parser.add_argument('--obs_prob', default=DEFAULT_OBS_PROB, type=float, help='Obstacle generation probability/density')
    parser.add_argument('--obstacle_size', default=DEFAULT_OBSTACLE_SIZE, type=float, help='Default size of generated obstacles')
    parser.add_argument('--async_llm', default=DEFAULT_ASYNC_LLM, type=str2bool, help='Use asynchronous LLM processing')

    args = parser.parse_args()
    
    # Update global history size from args
    HISTORY_MAX_ENTRIES = args.history_max_entries

    OBS_TYPE_FOR_DEMO = ObservationType.KIN
    ACT_TYPE_FOR_DEMO = ActionType.RPM 

    print("\n" + "="*60)
    print("LLM Orchestrated Multi-Drone KIN Demo")
    print(f"LLM Mode: {'ASYNCHRONOUS' if args.async_llm else 'SYNCHRONOUS'}")
    print("WITH DRONE STATE HISTORY")
    print("="*60)
    for arg_name, arg_val in vars(args).items():
        print(f"{arg_name:<25}: {arg_val}")
    print(f"{'OBS_TYPE_FOR_DEMO':<25}: {OBS_TYPE_FOR_DEMO.value}")
    print(f"{'ACT_TYPE_FOR_DEMO':<25}: {ACT_TYPE_FOR_DEMO.value}")
    print("="*60 + "\n")

    if not gemini_available:
        print("[FATAL_ERROR] Gemini API is not available. This demo requires LLM functionality.")
        print("              Please check your API key and 'pip install google-generativeai'.")
        exit(1)

    run_llm_orchestrated_demonstration(
        model_path=args.model_path,
        num_drones=args.num_drones,
        obs_type=OBS_TYPE_FOR_DEMO, 
        act_type=ACT_TYPE_FOR_DEMO, 
        output_folder=args.output_folder,
        gui=args.gui,
        record_video=args.record_video,
        ctrl_freq=args.ctrl_freq,
        duration_sec=args.duration_sec,
        llm_update_interval_sec=args.llm_update_interval,
        plot=args.plot,
        add_obstacles=args.add_obstacles,
        obs_prob=args.obs_prob,
        obstacle_size=args.obstacle_size,
        async_llm=args.async_llm  # Pass the async_llm parameter
    )