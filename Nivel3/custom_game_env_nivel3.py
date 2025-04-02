import numpy as np
import subprocess
import time
import gymnasium as gym
from gymnasium import spaces
from pynput.keyboard import Controller, Key
import requests

class CustomGameEnv3(gym.Env):
    def __init__(self, exe_path, max_steps=100):
        super(CustomGameEnv3, self).__init__()
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
        self.last_lives = 3
        #self.current_souls = 0
        self.episode_reward_details = {  # Initialize reward tracking
            "approach_goal": 0,
            "move_away": 0,
            "goal_reached": 0,
            "no_move_penalty": 0,
            "max_steps_penalty": 0
        }

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
                "final": bool(game_data.get("final", False)),  # New boolean value for door collision
                "lives": int(game_data["lives"]),
                "souls": int(game_data["souls"])
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

        # Initial state (example structure)
        game_data = self.get_game_data()
        if game_data:
            position = game_data["position"]
            end_position = game_data["end_position"]
            self.final = game_data["final"]
            self.current_lives = game_data["lives"]
            self.current_souls = game_data["souls"]
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
            self.atack()

        game_data = self.get_game_data()
        terminated = False
        truncated = False

        if game_data:
            position = tuple(game_data["position"])
            end_position = tuple(game_data["end_position"])
            self.final = bool(game_data.get("final",False))
            self.current_lives = game_data["lives"]
            self.current_souls = game_data["souls"]
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
                reward_penalty = 2
                reward += reward_penalty
                self.episode_reward_details["move_away"] += reward_penalty
        else:
            # Initial reward based on the distance, capped to avoid large values
            initial_reward = min(5, 0.5 / (distance_to_goal + 1e-6))
            reward += initial_reward
            self.episode_reward_details["approach_goal"] += initial_reward

        if self.current_lives < self.last_lives:
            # Calculate penalty based on remaining lives
            if self.current_lives == 2:
                penalty = -5  # Mild penalty for losing first life
            elif self.current_lives == 1:
                penalty = -15  # Stronger penalty for losing second life
            elif self.current_lives == 0:
                penalty = -50  # Significant penalty for losing all lives
                terminated = True  # End the episode if all lives are lost
                print("Episode terminated: Agent lost all lives.")
                return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False, "lives": self.current_lives, "souls": self.current_souls}
            reward += penalty
            print(f"Penalty applied for losing a life: {penalty}")

        self.last_lives = self.current_lives   
           
            # Check if the agent has collided with the door
        if self.final == True:
            terminated = True
            # Recompensas escalonadas según el número de vidas restantes
            if self.current_lives == 3:
                goal_reward = 30  # Alta recompensa por terminar con 3 vidas
            elif self.current_lives == 2:
                goal_reward = 20  # Recompensa moderada por terminar con 2 vidas
            elif self.current_lives == 1:
                goal_reward = 10  # Baja recompensa por terminar con 1 vida
            else:
                goal_reward = 0  # En teoría, esto no debería ocurrir si el agente no tiene vidas

            reward += goal_reward
            self.episode_reward_details["goal_reached"] += goal_reward
            print(f"Episode terminated: Collided with the door (end position). Lives remaining: {self.current_lives}. Reward given: {goal_reward}")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": True, "lives": self.current_lives, "souls": self.current_souls}

        if self.last_position == position:
            self.same_position_count += 1
            if self.same_position_count >= 5:  # Verifica si ha permanecido en la misma posición por 5 o más pasos
                no_move_penalty = -15  # Penalización por quedarse en la misma posición (cambiado a negativo)
                reward += no_move_penalty
                print("Penalty: Agent did not move for 5 or more steps.")
                self.episode_reward_details["no_move_penalty"] += no_move_penalty
        else:
            self.same_position_count = 0 
                    
        self.last_position = position
        self.last_distance_to_goal = distance_to_goal

        # Increment the step count and check if max steps are reached
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
            max_steps_penalty = 20
            reward += max_steps_penalty
            self.episode_reward_details["max_steps_penalty"] += max_steps_penalty
            print("Episode terminated: Max steps reached.")
            print(f"Terminated status at end of step(): {terminated}")
            return np.array(position + end_position, dtype=np.float32), reward, terminated, truncated, {"final": self.final, "is_success": False, "lives": self.current_lives,"souls": self.current_souls}

        if self.current_souls > 0:
            reward += self.current_souls * 0.2
            self.current_souls = 0
        
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

        return next_state, reward, terminated, truncated, {"final": self.final, "is_success": terminated, "lives": self.current_lives,"souls": self.current_souls}

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
        time.sleep(0.5)
        self.keyboard.release('w')

    def atack(self):
        self.keyboard.press(Key.space)
        time.sleep(0.3)
        self.keyboard.release(Key.space)