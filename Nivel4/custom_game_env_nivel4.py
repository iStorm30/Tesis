import numpy as np
import subprocess
import time
import gymnasium as gym
from gymnasium import spaces
from pynput.keyboard import Controller
import requests

class CustomGameEnv4(gym.Env):
    def __init__(self, exe_path, max_steps=1000):
        super(CustomGameEnv4, self).__init__()
        self.exe_path = exe_path

        # Define action space and observation space
        self.action_space = spaces.Discrete(3)  # 3 possible actions: left, right, up
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Controller and environment variables
        self.keyboard = Controller()
        self.current_step = 0
        self.max_steps = max_steps
        self.game_process = None
        self.last_distance_to_goal = None  # Track the distance to the goal
        self.last_position = None
        self.same_position_count = 0
        self.episode_reward_details = {  # Initialize reward tracking
            "approach_goal": 0,
            "move_away": 0,
            "goal_reached": 0,
            "no_move_penalty": 0,
            "max_steps_penalty": 0
        }

        # New variables for stability, jump limits, and initial y position
        self.highest_position_reached = None
        self.stability_steps = 0
        self.jump_count = 0
        self.initial_y_position = None  # Initialize as None

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
        self.highest_position_reached = None
        self.stability_steps = 0
        self.jump_count = 0
        self.episode_reward_details = {key: 0 for key in self.episode_reward_details}

        # Initial state (example structure)
        game_data = self.get_game_data()
        if game_data:
            position = game_data["position"]
            end_position = game_data["end_position"]
            self.final = game_data["final"]
            
            # Set initial_y_position only if it hasn't been set before
            if self.initial_y_position is None:
                self.initial_y_position = position[1]
                print(f"Initial Y position set to: {self.initial_y_position}")  # Optional debug print

            return np.array(position + end_position, dtype=np.float32), {}
        else:
            return np.zeros(4, dtype=np.float32), {}

    def launch_game(self):
        # Launch the game process
        self.game_process = subprocess.Popen(self.exe_path)
        time.sleep(2)  # Wait for the game to load

    def step(self, action):
        reward = 0  # Initialize reward

        # Perform the action
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.move_up()

        game_data = self.get_game_data()
        terminated = False
        truncated = False

        if game_data:
            position = tuple(game_data["position"])
            end_position = tuple(game_data["end_position"])
            self.final = bool(game_data.get("final", False))
            # Calculate distance to goal
            distance_to_goal_y = abs(position[1] - end_position[1])

        if self.last_distance_to_goal is not None:
            if distance_to_goal_y < self.last_distance_to_goal:
                # Reward for moving closer to the goal with a stable linear increase
                reward_gain = max(0.1, np.log1p(10 / (distance_to_goal_y + 1e-6)))
                reward += reward_gain
                self.episode_reward_details["approach_goal"] += reward_gain
            else:
                # Small penalty for moving away from the goal
                reward_penalty = -2
                reward += reward_penalty
                self.episode_reward_details["move_away"] += reward_penalty
        else:
            # Initial reward based on the distance, capped to avoid large values
            initial_reward = min(5, np.log1p(0.5 / (distance_to_goal_y + 1e-6)))
            reward += initial_reward
            self.episode_reward_details["approach_goal"] += initial_reward

        # Debug print to check positions
        print(f"Initial Y position: {self.initial_y_position}, Current Y position: {position[1]}")

        # Example strategy: reward for staying at a stable height above the initial y position
        if self.highest_position_reached is None or position[1] < self.highest_position_reached:
            self.highest_position_reached = position[1]
            reward += 10  # Reward for reaching a new height
            print("Recompensa: El agente alcanzó una nueva altura.")
        else:
            # Penalize if the agent stays below its highest point
            reward -= 5
            print("Penalización: El agente no ha subido más alto.")

        # Penalization for arbitrary jumps only when not progressing upwards
        if self.last_position is not None and position[1] >= self.last_position[1] and position[1] < self.initial_y_position:
            reward -= 5
            print("Penalización: El agente saltó sin progresar.")

        # Other logic for rewards and penalties can follow...
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
            if self.same_position_count >= 2:  # Verifica si ha permanecido en la misma posición por 5 o más pasos
                no_move_penalty = -10  # Penalización por quedarse en la misma posición (cambiado a negativo)
                reward += no_move_penalty
                print("Penalty: Agent did not move for 5 or more steps.")
                self.episode_reward_details["no_move_penalty"] += abs(no_move_penalty)                 
        else:
            self.same_position_count = 0 

        self.last_position = position
        self.last_distance_to_goal = distance_to_goal_y

        # Check for maximum steps
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
            max_steps_penalty = 15
            reward += max_steps_penalty
            self.episode_reward_details["max_steps_penalty"] += max_steps_penalty
            print("Episode terminated: Max steps reached.")
            print(f"Terminated status at end of step(): {terminated}")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False}

        # Return state, reward, terminated flag, and additional info
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

    # Movement functions
    def move_left(self):
        self.keyboard.press('a')
        time.sleep(0.5)
        self.keyboard.release('a')

    def move_right(self):
        self.keyboard.press('d')
        time.sleep(0.5)
        self.keyboard.release('d')

    def move_up(self):
        self.keyboard.press('w')
        time.sleep(0.3)
        self.keyboard.release('w')

