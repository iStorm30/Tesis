import numpy as np
import pyautogui
import subprocess
import time
import cv2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from pynput.keyboard import Controller, Key
import requests

class CustomGameEnv(gym.Env):
    def __init__(self, exe_path, max_steps=100):  # Reducir max_steps para entrenamiento inicial
        super(CustomGameEnv, self).__init__()
        self.exe_path = exe_path
        self.CAPTURE_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        
        # Definir espacios de acción y observación
        self.action_space = spaces.Discrete(4)  # 4 acciones posibles: izquierda, derecha, arriba, disparo
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 1), dtype=np.uint8)
        
        # Controladores y variables de entorno
        self.keyboard = Controller()
        self.state = None
        self.total_reward = 0
        self.last_position = None
        self.current_step = 0
        self.max_steps = max_steps
        self.game_process = None

        # Variables para recompensas y penalizaciones
        self.visited_areas = set()  # Conjunto para rastrear áreas ya exploradas
        self.penalty_count = 0       # Contador para verificar evasión de áreas o acciones nuevas
        self.last_distance_to_goal = None  # Variable para calcular la recompensa por moverse hacia el objetivo
        self.last_shoot_step = -15 

    def get_game_data(self):
        try:
            response = requests.get("http://127.0.0.1:8000/api/game_data")
            response.raise_for_status()
            game_data_list = response.json()
        
            if not game_data_list:  # Verifica si la lista está vacía
                print("Advertencia: No se recibieron datos del juego.")
                return None

        # Obtener la última entrada de datos
            game_data = game_data_list[-1]
        
        # Convertir los valores de 'lives', 'souls' y posiciones a enteros, usando listas para los vectores
            return {
                "lives": int(game_data["lives"]),
                "souls": int(game_data["souls"]),
                "position": [int(game_data["position"]["x"]), int(game_data["position"]["y"])],
                "end_position": [int(game_data["end_position"]["x"]), int(game_data["end_position"]["y"])]
            }
        except requests.exceptions.RequestException as e:
            print("Error al obtener datos del juego:", e)
            return None
        except ValueError as e:
            print("Error al convertir datos a enteros:", e)
            return None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        if self.game_process:
            self.game_process.terminate()
        self.launch_game()
        time.sleep(1)
        self.state = self.capture_state()
        self.total_reward = 0
        self.current_step = 0
        self.last_position = None
        self.visited_areas.clear()  # Reiniciar áreas visitadas al comienzo de un nuevo episodio
        self.previous_action = None
        self.penalty_count = 0
        self.last_distance_to_goal = None  # Reiniciar la distancia al objetivo
        self.last_shoot_step = -15 
        return self.process_state(self.state), {}
    
    def launch_game(self):
        # Inicia el proceso del juego
        self.game_process = subprocess.Popen(self.exe_path)
        time.sleep(2)  # Espera a que el juego cargue completamente

    def capture_state(self):
        screenshot = pyautogui.screenshot(region=(self.CAPTURE_REGION["left"], self.CAPTURE_REGION["top"],
                                                  self.CAPTURE_REGION["width"], self.CAPTURE_REGION["height"]))
        image = np.array(screenshot)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def step(self, action):
        reward = 0  # Inicializar recompensa para este paso

        # Realizar la acción
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_right()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.shoot()
            if self.current_step - self.last_shoot_step > 15:
                self.shoot()
                self.last_shoot_step = self.current_step
            else:
                reward -= 1  # Penalizar disparos muy frecuentes

        game_data = self.get_game_data()
        terminated = False
        truncated = False

        if game_data:
            lives = game_data["lives"]
            souls = game_data["souls"]
            position = tuple(game_data["position"])
            end_position = tuple(game_data["end_position"])

            # Penalización por quedarse en el mismo lugar
            if self.last_position == position:
                reward -= 3  # Penalización incrementada
                print("Penalización aplicada: el agente se mantuvo en la misma posición")
            self.last_position = position

                        # Recompensa por moverse en la dirección correcta
            distance_to_goal = np.sqrt((position[0] - end_position[0])**2 + (position[1] - end_position[1])**2)
            if self.last_distance_to_goal is not None:
                if distance_to_goal < self.last_distance_to_goal:
                    reward += max(0.1, 5 / (distance_to_goal + 1))  # Recompensa decreciente conforme se acerca
                    print("Recompensa aplicada: el agente se movió en la dirección correcta")
                else:
                    reward -= 1  # Penalización leve si se aleja del objetivo
            self.last_distance_to_goal = distance_to_goal

            if distance_to_goal < 10:
                reward += 20
                print("Recompensa significativa: el agente esta cerca de completar el nivel.")

            # Recompensa por eliminar enemigos (souls)
            if souls > 0:
                reward += souls * 5
                souls = 0

            reward += (lives - 3) * 5

            if position == end_position:
                terminated = True
                reward += 100
                print("Episodio terminado: el jugador ha llegado a la posición final.")
            elif lives <= 0:
                terminated = True
                reward -= 10
                print("Episodio terminado: el jugador se quedó sin vidas.")

            # Recompensa Incremental y Penalización Ajustada por Exploración de Áreas
            if position not in self.visited_areas:
                reward += 3
                self.visited_areas.add(position)
                print(f"Área nueva explorada en {position}: recompensa asignada.")
            else:
                reward -= 0.5  # Penalización ajustada por regresar a áreas ya visitadas

        else:
            reward = -1

        # Incrementar el contador de pasos y verificar el límite
        self.current_step += 1
        print(f"Paso actual: {self.current_step}/{self.max_steps}")

        if self.current_step >= self.max_steps:
            terminated = True
            reward -= 1
            print("Episodio terminado: límite de pasos alcanzado.")

        next_state = self.process_state(self.capture_state())
        return next_state, reward, terminated, truncated, {}

    def process_state(self, image):
        resized_image = cv2.resize(image, (80, 80)).reshape(80, 80, 1)
        return resized_image

    def close(self):
        if self.game_process:
            self.game_process.terminate()

    # Funciones de movimiento y disparo
    def move_left(self):
        self.keyboard.press('a')
        time.sleep(0.1)
        self.keyboard.release('a')

    def move_right(self):
        self.keyboard.press('d')
        time.sleep(0.1)
        self.keyboard.release('d')

    def move_up(self):
        self.keyboard.press('w')
        time.sleep(0.1)
        self.keyboard.release('w')

    def shoot(self):
        self.keyboard.press(Key.space)
        time.sleep(0.3)
        self.keyboard.release(Key.space)