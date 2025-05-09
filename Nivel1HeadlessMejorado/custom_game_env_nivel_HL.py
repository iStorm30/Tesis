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
        self._reset_consumed = False
        self.episode_reward_details = {  # Initialize reward tracking
            "approach_goal": 0,
            "move_away": 0,
            "goal_reached": 0,
            "no_move_penalty": 0,
            "max_steps_penalty": 0,
            "penalty_timeout":  0,
            "reset_penalty":   0
        }
        # ← Aquí creas tu sesión persistente
        self._http = requests.Session()
        self._http.headers.update({"Connection": "keep-alive"})
    
    def _send_command(self, action: int):
    #\"\"\"Envía el comando de movimiento a Godot via HTTP POST.\"\"\"
        cmd = {
            "left":  bool(action == 0),
            "right": bool(action == 1),
            #"jump":  bool(action == 2)  # si en este env tienes salto
            #"attack: bool(action == 3)" # si en este env tienes ataque
        }
        try:
            self._http.post(
                f"http://127.0.0.1:{self.api_port}/api/command",
                json=cmd,
                timeout=0.1
        )
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(0.3)

    def get_level_event(self):
        """
        Devuelve el último evento de nivel con campos 'timer' y 'reset'.
        """
        try:
            resp = self._http.get(f"http://127.0.0.1:{self.api_port}/api/level_event/latest")
            resp.raise_for_status()
            event = resp.json()

            # event ya es un dict con 'timer' y 'reset'
            timer = float(event.get("timer", 0.0))
            reset = bool(event.get("reset", False))
            if reset:
            # Consumimos el evento en el servidor para que no vuelva a aparecer
                try:
                    self._http.delete(f"http://127.0.0.1:{self.api_port}/api/level_event")
                except requests.exceptions.RequestException:
                    pass
            return timer, reset

        except requests.exceptions.RequestException as e:
            print("Error fetching level event:", e)
            return None
        except ValueError as e:
            print("Error parsing level event data:", e)
            return None
        except:
            return 0.0, False

    def get_game_data(self):
        try:
            response = self._http.get("http://127.0.0.1:8000/api/game_data")
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
        self.final = False
        self._reset_consumed = False

        if hasattr(self, "level_timer"):
            self.level_timer.stop()
            self.level_timer.start()

        self.episode_reward_details = {key: 0 for key in [
            "approach_goal", "move_away", "goal_reached",
            "no_move_penalty", "max_steps_penalty",
            "penalty_timeout", "reset_penalty"
        ]}

        # Initial state (example structure)
        gd = self.get_game_data()
        if gd:
            obs = np.array(gd["position"] + gd["end_position"], dtype=np.float32)
            return obs, {}
        else:
            # Asegúrate de coincidir dimensiones if agregas timer
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        

    def step(self, action):

        self._send_command(action)
        self.current_step += 1
        reward = 0  # Initialize reward

        game_data = self.get_game_data()
        if game_data is None:
        # antes de que Godot haya enviado datos, devolvemos zeros
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {}
        
        timer, reset = self.get_level_event()
        #evt = self.get_level_event()
        #if #evt is None:
            #timer, reset = 0.0, False
        #else:
            #timer = float(evt.get("timer", 0.0))
            #reset = bool(evt.get("reset", False))

        if reset and not self._reset_consumed:
            # Primer reset de este episodio: lo procesamos
            self._reset_consumed = True
        elif reset:
            # Ya lo consumimos antes, lo ignoramos
            reset = False
    
        terminated = False
        truncated = False
        # Increment the step count and check if max steps are reached
        
        position = tuple(game_data["position"])
        end_position = tuple(game_data["end_position"])
        self.final = bool(game_data.get("final",False))
        distance_to_goal = np.sqrt((position[0] - end_position[0])**2 + (position[1] - end_position[1])**2)


        if self.last_distance_to_goal is not None:
            if distance_to_goal < self.last_distance_to_goal:
                # Reward for moving closer to the goal with a stable linear increase
                reward_gain = max(0.1, 3* np.log1p(self.last_distance_to_goal - distance_to_goal))
                reward += reward_gain
                self.episode_reward_details["approach_goal"] += reward_gain
            else:
                # Small penalty for moving away from the goal
                reward_penalty = -5
                reward += reward_penalty
                self.episode_reward_details["move_away"] += reward_penalty
        else:
            # Initial reward based on the distance, capped to avoid large values
            initial_reward = min(5, 0.5 / (distance_to_goal + 1e-6))
            reward += initial_reward
            self.episode_reward_details["approach_goal"] += initial_reward 

        if reset and self.final == False:
            terminated = True
            reset_penalty = -5  # Penalización opcional por reinicio, ajusta según sea necesario
            reward += reset_penalty
            self.episode_reward_details["reset_penalty"] = reset_penalty
            print("Episode terminated: Reset event triggered by the API.")
            print(f"Episode terminated after {self.current_step} steps.")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False}     
        
        # Check if the agent has collided with the door
        if self.final == True:
            terminated = True
            goal_reward = 100  # Significant reward for reaching the goal
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

        # Print reward details at the end of the episode
        if terminated:
            print(f"Reward breakdown for the episode: {self.episode_reward_details}")
            # Reset the reward details for the next episode
            self.episode_reward_details = {key: 0 for key in self.episode_reward_details}

        obs  = np.array(position + end_position, dtype=np.float32)
        info = {
            "final": self.final,
            "timer": timer,
            "reset": reset
        }
        return obs, reward, terminated, truncated, info

        #return next_state, reward, terminated, truncated, {"final": self.final, "is_success": terminated}

    def close(self):
        if self.game_process:
            print("Closing the game process...")
            self.game_process.terminate()
            self.game_process = None


