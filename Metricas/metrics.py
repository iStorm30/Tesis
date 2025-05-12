import numpy as np
import subprocess
import time
import gymnasium as gym
from gymnasium import spaces
import requests

class CustomGameEnv1(gym.Env):
    """
    Entorno limpio para recolección de métricas: sin reward shaping,
    solo expone observaciones y flags de error/éxito en info.
    """
    def __init__(self, exe_path, api_port=8000):
        super(CustomGameEnv1, self).__init__()
        self.exe_path = exe_path
        self.api_port = api_port
        # Definir espacio de acciones y observaciones
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.game_process = None
        self.success_count = 0
        # Sesión HTTP permanente
        self._http = requests.Session()
        self._http.headers.update({"Connection": "keep-alive"})

    def _send_command(self, action: int):
        cmd = {
            "left":  bool(action == 0),
            "right": bool(action == 1),
            #"jump":  bool(action == 2),
            #"attack": bool(action == 3)
        }
        try:
            self._http.post(
                f"http://127.0.0.1:{self.api_port}/api/command",
                json=cmd,
                timeout=0.1
            )
        except requests.exceptions.RequestException:
            pass
        # Pequeña espera para recibir datos de Godot
        time.sleep(0.3)

    def get_level_event(self):
        try:
            resp = self._http.get(f"http://127.0.0.1:{self.api_port}/api/level_event/latest")
            resp.raise_for_status()
            evt = resp.json()
            timer = float(evt.get("timer", 0.0))
            reset = bool(evt.get("reset", False))
            if reset:
                # Consumir evento para no duplicarlo
                try:
                    self._http.delete(f"http://127.0.0.1:{self.api_port}/api/level_event")
                except requests.exceptions.RequestException:
                    pass
            return timer, reset, False, None
        except requests.exceptions.RequestException as e:
            # aquí capturamos el código o mensaje de error
            code = getattr(e.response, "status_code", None) or str(e)
            # print(f"ERROR: Error enviando GET: {code}")   # ya no hace falta para contar
            return 0.0, False, True, code        # señalamos el erro

    def get_game_data(self):
        try:
            resp = self._http.get(f"http://127.0.0.1:{self.api_port}/api/game_data")
            resp.raise_for_status()
            data_list = resp.json()
            if not data_list:
                return None
            gd = data_list[-1]
            return {
                "position": [int(gd["position"]["x"]), int(gd["position"]["y"])],
                "end_position": [
                    int(gd["end_position"]["x"]),
                    int(gd["end_position"]["y"])
                ],
                "final": bool(gd.get("final", False))
            }
        except Exception:
            return None

    def launch_game(self):
        if self.game_process is None:
            self.game_process = subprocess.Popen(self.exe_path)
            time.sleep(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Arrancar juego solo una vez
        self.launch_game()
        # Obtener la primera observación
        print("[DEBUG] CustomGameEnv1.reset() llamado. success_count se reinicia.")
        self.success_count = 0
        gd = self.get_game_data()
        if gd:
            obs = np.array(gd["position"] + gd["end_position"], dtype=np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        # No hay info extra al reset
        return obs, {}

    def step(self, action):
        # Enviar acción y avanzar un paso
        self._send_command(action)
        # Observación y flags
        gd = self.get_game_data()
        api_error = gd is None
        if api_error:
            # Si falla la API devolvemos estado cero y flag de error
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {"api_error": True, "success": False}

        # Procesar evento de nivel
        timer, reset, api_error, error_code = self.get_level_event()
        info = {
            "api_error": api_error,
            "api_error_code": error_code,
            "success": False
        }
        pos = tuple(gd["position"])
        end_pos = tuple(gd["end_position"])
        final = gd.get("final", False)

        # Inicio de retorno
        obs = np.array(pos + end_pos, dtype=np.float32)
        info = {"api_error": False, "success": False}

        # Reinicio externo (evento de timeout)
        if reset:
            #terminated = True
            info["reset_event"] = True

        # Meta alcanzada
        if final:
            #terminated = True
            info["success"] = True
            self.success_count += 1
            # DEBUG: imprime cada vez que haya un “final”
            print(f"[DEBUG] success_count incrementado: ahora vale {self.success_count}")
            
        info["success_count"] = self.success_count

        # Paso normal
        return obs, 0.0, False, False, info

    def close(self):
        if self.game_process:
            self.game_process.terminate()
            self.game_process = None
