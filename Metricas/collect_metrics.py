import csv
import time
import psutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from custom_game_env_metrics import CustomGameEnv1
from metrics_wrapper import MetricsWrapper


def main():
    # Ruta al ejecutable de Godot y puerto de la API
    exe_path = "ruta/a/tu_game.exe"
    api_port = 8000

    # 1) Construir entorno limpio y monitoreado
    base_env = CustomGameEnv1(exe_path=exe_path, api_port=api_port)
    monitored_env = Monitor(base_env, filename=None)
    env = MetricsWrapper(monitored_env)

    # 2) Cargar modelo entrenado (ajusta la ruta)
    model = PPO.load("ruta/a/tu_modelo.zip", env=env, device="cuda")

    # 3) Preparar CSV de salida
    csv_path = "metrics_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "length", "success",
            "bug_count", "state_coverage", "action_diversity",
            "episode_time", "fps", "cpu_percent", "mem_mb", "timestamp"
        ])

    process = psutil.Process()
    num_episodes = 100  # Ajusta según lo que necesites

    # 4) Loop de evaluación
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False

        # Ejecutar episodio hasta 'done'
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

        # Extraer métricas del episodio
        length           = info.get("episode", {}).get("l", 0)
        success          = int(info.get("success", False))
        bug_count        = info.get("bug_count", 0)
        state_coverage   = info.get("state_coverage", 0)
        action_diversity = info.get("action_diversity", 0)
        episode_time     = info.get("episode_time", 0.0)
        fps              = info.get("fps", 0.0)
        cpu_percent      = process.cpu_percent(interval=None)
        mem_mb           = process.memory_info().rss / (1024**2)
        timestamp        = time.time()

        # Volcar fila al CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                length,
                success,
                bug_count,
                state_coverage,
                action_diversity,
                episode_time,
                fps,
                cpu_percent,
                mem_mb,
                timestamp
            ])

        print(f"Episodio {ep}: éxito={success}, bugs={bug_count}, cobertura={state_coverage}")

    print(f"Evaluación completa. Métricas guardadas en {csv_path}")


if __name__ == "__main__":
    main()
