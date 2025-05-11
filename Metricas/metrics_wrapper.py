import gymnasium as gym
import time
import psutil
import numpy as np

class MetricsWrapper(gym.Wrapper):
    """
    Wrapper para capturar métricas de testing en cada episodio:
      - bug_count: bugs únicos (incluye api_error con código)
      - state_coverage: cobertura de estados visitados
      - action_diversity: diversidad de acciones tomadas
      - episode_time: duración del episodio en segundos
      - fps: pasos por segundo
    """
    def __init__(self, env, stuck_threshold=5):
        super().__init__(env)
        self.stuck_threshold = stuck_threshold
        self.process = psutil.Process()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Inicializar contadores
        self.episode_start = time.time()
        self.step_count = 0
        self.prev_pos = self._get_pos(obs)
        self.stuck_counter = 0
        self.bugs = set()
        self.states = set()
        self.actions = set()
        self.success_count   = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # -- Detección de bugs --
        if info.get("api_error", False):
            code = info.get("api_error_code", "unknown")
            self.bugs.add(f"api_error_{code}")

        if info.get("success", False):
            self.success_count += 1

        # -- Cobertura de estados --
        flat = np.asarray(obs).flatten()
        self.states.add(tuple(flat.tolist()))

        # -- Diversidad de acciones --
        # Manejo de scalars, 0-d arrays y arrays multidimensionales
        if isinstance(action, np.ndarray):
            if action.shape == ():
                action_id = action.item()
            else:
                action_id = tuple(action.flatten().tolist())
        elif np.isscalar(action):
            action_id = int(action)
        else:
            try:
                action_id = tuple(action)
            except TypeError:
                action_id = action
        self.actions.add(action_id)

        # -- Stuck & out_of_bounds --
        pos = self._get_pos(obs)
        if not self._in_bounds(pos):
            self.bugs.add("out_of_bounds")
        if pos == self.prev_pos:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                self.bugs.add("stuck")
        else:
            self.stuck_counter = 0
        self.prev_pos = pos

        # Al terminar episodio, inyectar métricas
        if terminated or truncated:
            ep_time = time.time() - self.episode_start
            fps = self.step_count / ep_time if ep_time > 0 else 0.0
            info.update({
                "bug_count": len(self.bugs),
                "state_coverage": len(self.states),
                "action_diversity": len(self.actions),
                "success_count":    self.success_count,
                "success_rate":     self.success_count / self.step_count if self.step_count > 0 else 0.0,
                "episode_time": ep_time,
                "fps": fps
            })
        return obs, reward, terminated, truncated, info

    def _get_pos(self, obs):
        # Adaptar según formato de tu obs
        if isinstance(obs, dict) and "position" in obs:
            return tuple(obs["position"])
        arr = np.asarray(obs).flatten()
        return (arr[0], arr[1]) if arr.size >= 2 else (0, 0)

    def _in_bounds(self, pos):
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        x, y = pos
        return bool(low[0] <= x <= high[0] and low[1] <= y <= high[1])

