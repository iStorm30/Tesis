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
        # Inicializar métricas de episodio
        self.episode_start = time.time()
        self.step_count = 0
        self.prev_pos = self._get_pos(obs)
        self.stuck_counter = 0
        self.bugs = set()
        self.states = set()
        # Asegurar clave de conteo de éxitos
        info.setdefault('success_count', 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Cobertura de estados
        pos = self._get_pos(obs)
        self.states.add(pos)
        # Detección de stuck
        if pos == self.prev_pos:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                self.bugs.add("stuck")
        else:
            self.stuck_counter = 0
        self.prev_pos = pos

        # Al terminar episodio, inyectar métricas y flags
        if terminated or truncated:
            ep_time = time.time() - self.episode_start
            fps = self.step_count / ep_time if ep_time > 0 else 0.0
            state_coverage = len(self.states)
            action_diversity = len(set(info.get('actions', [])))
            # Métricas base
            info.update({
                'bug_count': len(self.bugs),
                'state_coverage': state_coverage,
                'action_diversity': action_diversity,
                'episode_time': ep_time,
                'fps': fps,
                'cpu_percent': self.process.cpu_percent(interval=None),
                'mem_mb': self.process.memory_info().rss / (1024**2),
            })
            # Flags para detección de bugs externos
            flags = {
                'restart_triggered': info.get('success_count', 0) > 0,
                'variables_reset': info.get('variables_reset', False),
                'respawn_position_invalid': info.get('respawn_position_invalid', False),
                'entered_portal': info.get('entered_portal', False),
                'portal_collision': info.get('portal_collision', True),
                'stuck': 'stuck' in self.bugs,
                'ghost_collision': info.get('ghost_collision', False),
                'through_walls': info.get('through_walls', False),
                'enemy_movement': info.get('enemy_movement', True),
                'enemy_reposition': info.get('enemy_reposition', False),
                'enemy_persistent': info.get('enemy_persistent', False),
                'slow_response': info.get('slow_response', False),
                'timeout_error': info.get('timeout_error', False),
                'freezing': info.get('freezing', False),
                'dead_end': info.get('dead_end', False),
                'incomplete_level': info.get('incomplete_level', False),
                'infinite_loop': info.get('infinite_loop', False),
                'api_data_correct': info.get('api_data_correct', True),
            }
            info.update(flags)
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

