import numpy as np
import subprocess
import time
import gymnasium as gym
from gymnasium import spaces
import requests

class CustomGameEnv1(gym.Env):
    def __init__(self, exe_path,api_port=8000):
        super(CustomGameEnv1, self).__init__()
        self.exe_path = exe_path
        self.api_port = api_port
        # Define action space and observation space
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.current_step = 0
        self.game_process = None
        self.last_distance_to_goal = None  # Track the distance to the goal
        self.last_position = None
        self.same_position_count = 0
        self.has_started = False 
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
            #"attack: bool(action == 3)" # si en este env tienes ataque
        }
        try:
            requests.post(
                f"http://127.0.0.1:{self.api_port}/api/command",
                json=cmd,
                timeout=0.1
        )
        except requests.exceptions.RequestException:
            pass
        
        #time.sleep(0.3)

    def get_level_event(self):
        """
        Devuelve el último evento de nivel con campos 'timer' y 'reset'.
        """
        try:
            resp = requests.post(f"http://127.0.0.1:{self.api_port}/api/level_event")
            resp.raise_for_status()
            events = resp.json()

            if not events:  # Si la lista está vacía
                print("Warning: No level events received.")
                return None  # Valor predeterminado

            last = events[-1]
            return {
                "timer": float(last("timer", 0.0)),  # Cambiado a 0.0 por defecto
                "reset": bool(last.post("reset", False))  # Cambiado a False por defecto
            }
        except requests.exceptions.RequestException as e:
            print("Error fetching level event:", e)
            return None  # Valor predeterminado en caso de error

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
        
    def launch_game(self):
        # Launch the game process
        self.game_process = subprocess.Popen(self.exe_path)
        time.sleep(2)  # Wait for the game to load

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self.has_started:
            self.launch_game()
            self.has_started = True

        self.current_step = 0
        self.same_position_count = 0
        self.last_distance_to_goal = None
        self.last_position = None
        self.episode_reward_details = {key: 0 for key in self.episode_reward_details}


        # Initial state (example structure)
        game_data = self.get_game_data()
        if game_data:
            position = game_data["position"]
            end_position = game_data["end_position"]
            self.final = game_data["final"]
            return np.array(position + end_position, dtype=np.float32), {}
        else:
            return np.zeros(4, dtype=np.float32), {}

    def step(self, action):

        self._send_command(action)
        reward = 0  # Initialize reward

        game_data = self.get_game_data()
        le = self.get_level_event()
    
        terminated = False
        truncated = False
        # Increment the step count and check if max steps are reached
        self.current_step += 1

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

        if le:
            timer = float(le["timer"])
            reset = bool(le["reset", False])

        if reset == True:
            terminated = True
            reset_penalty = -5  # Penalización opcional por reinicio, ajusta según sea necesario
            reward += reset_penalty
            self.episode_reward_details["reset_penalty"] = reset_penalty
            print("Episode terminated: Reset event triggered by the API.")
            print(f"Episode terminated after {self.current_step} steps.")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False}     

        if  timer <= 0:
            terminated = True
            penalty = -10
            reward += penalty
            self.episode_reward_details["penalty_timeout"] += penalty
            print("Episode terminated: Max real-time duration reached.")
            print(f"Episode terminated after {self.current_step} steps.")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False} 
        
        # Check if the agent has collided with the door
        if self.final == True:
            terminated = True
            goal_reward = 15  # Significant reward for reaching the goal
            reward += goal_reward
            self.episode_reward_details["goal_reached"] += goal_reward
            print("Episode terminated: Collided with the door (end position).")
            print(f"Episode terminated after {self.current_step} steps.")
            print(f"Terminated status at end of step(): {terminated}")
            return np.array(position + end_position, dtype=np.float32), reward,terminated, truncated, {"final": self.final, "is_success": True}

        if self.last_position == position:
            self.same_position_count += 1
            if self.same_position_count >= 3:  # Verifica si ha permanecido en la misma posición por 3 o más pasos
                no_move_penalty = -15  # Penalización por quedarse en la misma posición (cambiado a negativo)
                reward += no_move_penalty
                print("Penalty: Agent did not move for 5 or more steps.")
                self.episode_reward_details["no_move_penalty"] += no_move_penalty
        else:
            self.same_position_count = 0 
                   
        self.last_position = position
        self.last_distance_to_goal = distance_to_goal

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
            print("Closing the game process...")
            self.game_process.terminate()
            self.game_process = None


