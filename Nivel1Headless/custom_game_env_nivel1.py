import numpy as np
import subprocess
import time
import datetime
import gymnasium as gym
from gymnasium import spaces
#from pynput.keyboard import Controller
import requests

class CustomGameEnv1(gym.Env):
    def __init__(self, exe_path,api_port=8000, max_steps=750):
        super(CustomGameEnv1, self).__init__()
        self.exe_path = exe_path

        self.api_port = api_port

        # Define action space and observation space
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Controller and environment variables
        #self.keyboard = Controller()
        self.current_step = 0
        self.max_steps = max_steps
        self.game_process = None
        self.last_distance_to_goal = None  # Track the distance to the goal
        self.last_position = None
        self.same_position_count = 0
        self.start_time = None
        self.max_episode_time = 60  # en segundos, puedes ajustarlo
        self.episode_reward_details = {  # Initialize reward tracking
            "approach_goal": 0,
            "move_away": 0,
            "goal_reached": 0,
            "no_move_penalty": 0,
            "max_steps_penalty": 0
        }
    
    def _send_command(self, action: int):
    #\"\"\"Envía el comando de movimiento a Godot via HTTP POST.\"\"\"
        cmd = {
            "left":  bool(action == 0),
            "right": bool(action == 1),
            #"jump":  bool(action == 2)  # si en este env tienes salto
        }
        try:
            requests.post(
                f"http://127.0.0.1:{self.api_port}/api/command",
                json=cmd,
                timeout=0.1
        )
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(0.3)


    def get_game_data(self):
        try:
            response = requests.get("http://127.0.0.1:8000/api/game_data")
            response.raise_for_status()
            game_data_list = response.json()

            if not game_data_list:
                print("Warning: No game data received.")
                return None

            # Retrieve the last game data entry
            game_data = game_data_list[-1]
            print(f"Latest game data: {game_data}")

            return {
                "position": [int(game_data["position"]["x"]), int(game_data["position"]["y"])],
                "end_position": [int(game_data["end_position"]["x"]), int(game_data["end_position"]["y"])],
                "final": bool(game_data.get("final", False))  # New boolean value for door collision
            }
        except requests.exceptions.RequestException as e:
            print("Error fetching game data:", e)
            return None
        except ValueError as e:
            print("Error converting game data:", e)
            return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_process:
            self.game_process.terminate()
        self.launch_game()
        time.sleep(1)

        self.current_step = 0
        self.same_position_count = 0
        self.last_distance_to_goal = None
        self.last_position = None
        self.episode_reward_details = {key: 0 for key in self.episode_reward_details}

        self.start_time = time.time()

        # Initial state (example structure)
        game_data = self.get_game_data()
        if game_data:
            position = game_data["position"]
            end_position = game_data["end_position"]
            self.final = game_data["final"]
            return np.array(position + end_position, dtype=np.float32), {}
        else:
            return np.zeros(4, dtype=np.float32), {}

    def launch_game(self):
        # Launch the game process
        self.game_process = subprocess.Popen(self.exe_path)
        time.sleep(2)  # Wait for the game to load

    def step(self, action):
        
        #t0 = time.time()

        self._send_command(action)
        reward = 0  # Initialize reward

        # Perform the action
        #if action == 0:
            #self.move_left()
        #elif action == 1:
            #self.move_right()

        game_data = self.get_game_data()
        
        #t1 = time.time()
        #real_dt = t1 - t0
        
        terminated = False
        truncated = False

        #sim_ts_str = game_data.get("timestamp")  # ej. "2025-05-06T00:12:34.567890"
        #ts_str = str(sim_ts_str)

        #if self.last_position is None or position != self.last_position:
            #self.last_moved_time = time.time()
        #self.last_position = position

        # DEBUG: imprime el valor y su tipo para saber qué recibes
        #print(f"[Perf][DEBUG] raw timestamp: {sim_ts_str!r} (type: {type(sim_ts_str)})")

        #try:
            #sim_ts = datetime.datetime.fromisoformat(ts_str)
        #except Exception:
            #print(f"[Perf] No pude parsear '{ts_str}', usando ahora()")
            #sim_ts = datetime.datetime.now()

        #if hasattr(self, "last_sim_ts"):
            #sim_dt = (sim_ts - self.last_sim_ts).total_seconds()
            #ratio = sim_dt / real_dt if real_dt > 0 else float("inf")
            #print(f"[Perf] real_dt={real_dt:.3f}s | sim_dt={sim_dt:.3f}s | ratio={ratio:.2f}×")
        #else:
            #print(f"[Perf] primer paso, real_dt={real_dt:.3f}s")

        #self.last_sim_ts = sim_ts

        if game_data:
            position = tuple(game_data["position"])
            end_position = tuple(game_data["end_position"])
            self.final = bool(game_data.get("final",False))
            # Calculate distance to goal
            distance_to_goal = np.sqrt((position[0] - end_position[0])**2 + (position[1] - end_position[1])**2)

            # Reward or penalty based on movement direction
        if self.last_distance_to_goal is not None:
            if distance_to_goal < self.last_distance_to_goal:
                # Reward for moving closer to the goal with a stable linear increase
                reward_gain = max(0.1, 3* np.log1p(self.last_distance_to_goal - distance_to_goal))
                reward += reward_gain
                self.episode_reward_details["approach_goal"] += reward_gain
            else:
                # Small penalty for moving away from the goal
                reward_penalty = -2
                reward += reward_penalty
                self.episode_reward_details["move_away"] += reward_penalty
        else:
            # Initial reward based on the distance, capped to avoid large values
            initial_reward = min(5, 0.5 / (distance_to_goal + 1e-6))
            reward += initial_reward
            self.episode_reward_details["approach_goal"] += initial_reward

            # Check if the agent has collided with the door
        if self.final == True:
            terminated = True
            goal_reward = 15  # Significant reward for reaching the goal
            reward += goal_reward
            self.episode_reward_details["goal_reached"] += goal_reward
            print("Episode terminated: Collided with the door (end position).")
            print(f"Terminated status at end of step(): {terminated}")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": True}       

        if self.last_position == position:
            self.same_position_count += 1
            if self.same_position_count >= 3:  # Verifica si ha permanecido en la misma posición por 3 o más pasos
                no_move_penalty = -15  # Penalización por quedarse en la misma posición (cambiado a negativo)
                reward += no_move_penalty
                print("Penalty: Agent did not move for 5 or more steps.")
                self.episode_reward_details["no_move_penalty"] += no_move_penalty
        else:
            self.same_position_count = 0 
        
        #elapsed = time.time() - self.last_moved_time
        #if elapsed > 2.0:
            #penalty = -15
            #reward += penalty
            #self.episode_reward_details["no_move_penalty"] += penalty
            # y reseteas el temporizador para no repetirlo inmediatamente
            #self.last_moved_time = time.time()    
                   
        self.last_position = position
        self.last_distance_to_goal = distance_to_goal


        # Increment the step count and check if max steps are reached
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
            max_steps_penalty = -30
            reward += max_steps_penalty
            self.episode_reward_details["max_steps_penalty"] += max_steps_penalty
            print("Episode terminated: Max steps reached.")
            print(f"Terminated status at end of step(): {terminated}")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False}
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_episode_time:
            terminated = True
            timeout_penalty = -20
            reward += timeout_penalty
            print("Episode terminated: Max real-time duration reached.")

        # State as position and end position
        if game_data:
            next_state = np.array(game_data["position"] + game_data["end_position"], dtype=np.float32)
            self.final = game_data["final"]
        else:
            next_state = np.zeros(5, dtype=np.float32)
            self.final = False

        # Print reward details at the end of the episode
        if terminated:
            print(f"Reward breakdown for the episode: {self.episode_reward_details}")
            # Reset the reward details for the next episode
            self.episode_reward_details = {key: 0 for key in self.episode_reward_details}

        return next_state, reward, terminated, truncated, {"final": self.final, "is_success": terminated}

    def close(self):
        if self.game_process:
            self.game_process.terminate()
                # Primero limpia el entorno de Gym
        #try:
            #super().close()
        #except Exception:
            #pass

        # Luego cierra el proceso de Godot si sigue vivo
        #if hasattr(self, "game_process") and self.game_process:
            #self.game_process.terminate()  # SIGTERM
            #try:
                #self.game_process.wait(timeout=5)  # espera hasta 5 s
            #except subprocess.TimeoutExpired:
                #self.game_process.kill()       # SIGKILL si no responde
            #finally:
                #self.game_process = None

    # Movement functions
    #def move_left(self):
        #self.keyboard.press('a')
        #time.sleep(0.5)
        #self.keyboard.release('a')

    #def move_right(self):
        #self.keyboard.press('d')
        #time.sleep(0.5)
        #self.keyboard.release('d')


