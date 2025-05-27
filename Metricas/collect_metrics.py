import csv
import time
import psutil
import numpy as np
import argparse
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from metrics import CustomGameEnv1
from metrics_wrapper import MetricsWrapper

# Catálogo de bugs con ID y descripción
def get_bug_catalog():
    return {
        'BUG-01': 'El personaje llega al final pero no se activa el reinicio del nivel.',
        'BUG-02': 'El reinicio se ejecuta pero no se restablece correctamente las variables.',
        'BUG-03': 'El personaje reaparece en una posición incorrecta o fuera del mapa.',
        'BUG-04': 'El área de colisión del portal no responde al contacto del personaje.',
        'BUG-05': 'El personaje se queda atascado entre plataformas o esquinas.',
        'BUG-06': 'El personaje muere sin contacto visible (colisión fantasma).',
        'BUG-07': 'El personaje atraviesa zonas donde debería haber colisión.',
        'BUG-08': 'Enemigos no se mueven o tienen animaciones congeladas.',
        'BUG-09': 'Enemigos desaparecen o aparecen mal posicionados tras reinicio.',
        'BUG-10': 'Enemigos siguen activos tras “morir” y siguen causando daño.',
        'BUG-11': 'Lenta respuesta del juego a los comandos del agente.',
        'BUG-12': 'Tiempo de espera muy corto (timeout) causando errores en consola.',
        'BUG-13': 'Congelamiento del personaje tras múltiples acciones rápidas.',
        'BUG-14': 'El personaje llega a un área sin salida.',
        'BUG-15': 'Nivel no se puede completar por error de diseño o plataformas mal conectadas.',
        'BUG-16': 'El agente puede reiniciar el nivel infinitamente sin restricciones (loop).',
        'BUG-17': 'Información incorrecta recibida desde la API (vidas, posición, estado).'
    }

# Reglas para detección de bugs según información del entorno y wrapper

def detect_bugs(info, step_counter, success):
    bugs_detected = []
    # Extraer flags de info
    restart_triggered = info.get('restart_triggered', False)
    variables_reset = info.get('variables_reset', False)
    respawn_invalid = info.get('respawn_position_invalid', False)
    entered_portal = info.get('entered_portal', False)
    portal_collision = info.get('portal_collision', True)
    stuck = info.get('stuck', False)
    ghost_collision = info.get('ghost_collision', False)
    through_walls = info.get('through_walls', False)
    enemy_movement = info.get('enemy_movement', True)
    enemy_reposition = info.get('enemy_reposition', False)
    enemy_persistent = info.get('enemy_persistent', False)
    slow_response = info.get('slow_response', False)
    timeout_error = info.get('timeout_error', False)
    freezing = info.get('freezing', False)
    dead_end = info.get('dead_end', False)
    incomplete_level = info.get('incomplete_level', False)
    infinite_loop = info.get('infinite_loop', False)
    api_ok = info.get('api_data_correct', True)

    # Evaluar cada bug
    if success > 0 and not restart_triggered:
        bugs_detected.append('BUG-01')
    if restart_triggered and not variables_reset:
        bugs_detected.append('BUG-02')
    if respawn_invalid:
        bugs_detected.append('BUG-03')
    if entered_portal and not portal_collision:
        bugs_detected.append('BUG-04')
    if stuck:
        bugs_detected.append('BUG-05')
    if ghost_collision:
        bugs_detected.append('BUG-06')
    if through_walls:
        bugs_detected.append('BUG-07')
    if not enemy_movement:
        bugs_detected.append('BUG-08')
    if enemy_reposition:
        bugs_detected.append('BUG-09')
    if enemy_persistent:
        bugs_detected.append('BUG-10')
    if slow_response:
        bugs_detected.append('BUG-11')
    if timeout_error:
        bugs_detected.append('BUG-12')
    if freezing:
        bugs_detected.append('BUG-13')
    if dead_end:
        bugs_detected.append('BUG-14')
    if incomplete_level:
        bugs_detected.append('BUG-15')
    if infinite_loop:
        bugs_detected.append('BUG-16')
    if not api_ok:
        bugs_detected.append('BUG-17')

    # Eliminar duplicados y mantener orden
    return list(dict.fromkeys(bugs_detected))


def main(num_episodes=100, seed=42):
    np.random.seed(seed)
    process = psutil.Process()
    bug_catalog = get_bug_catalog()

    # Acumuladores
    success_counts = []
    episode_times = []
    steps_per_completion = []
    bug_counts = []
    found_bug_ids = []
    fps_values = []
    cpu_values = []
    mem_values = []

    # Configuración del entorno
    exe_path = "C:\\Users\\ghost\\Documents\\TESIS-REPO\\Tesis\\Metricas\\Abby's Redemption-PruebaHL-Nivel1.console"
    api_port = 8000
    base_env = CustomGameEnv1(exe_path=exe_path, api_port=api_port)
    monitored_env = Monitor(base_env, filename=None)
    timed_env = TimeLimit(monitored_env, max_episode_steps=15000)
    env = MetricsWrapper(timed_env)

    model = PPO.load(
        "C:\\Users\\ghost\\Documents\\TESIS-REPO\\Tesis\\Metricas\\dqn_custom_game_model_PPO",
        env=env, device="cuda"
    )

    # Inicializar CSV detalle
    with open("metrics_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["Episodio", "Éxitos", "BugsIDs", "Pasos_por_fin", "Tiempo_s", "FPS", "CPU_%", "Mem_MB"]
        writer.writerow(headers)

    # Ejecutar episodios
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated = truncated = False
        step_counter = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            step_counter += 1

        # Métricas por episodio
        success = info.get('success_count', 0)
        ep_time = info.get('episode_time', 0.0)
        fps = info.get('fps', 0.0)
        cpu = process.cpu_percent(interval=None)
        mem = process.memory_info().rss / (1024**2)

        # Detectar bugs con reglas
        bug_ids = detect_bugs(info, step_counter, success)
        bug_counts.append(len(bug_ids))
        found_bug_ids.extend(bug_ids)

        # Pasos promedio por finalización
        steps_pc = (step_counter / success) if success > 0 else 0

        # Escribir detalle
        with open("metrics_report.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                success,
                ";".join(bug_ids) if bug_ids else "None",
                f"{steps_pc:.2f}",
                f"{ep_time:.2f}",
                f"{fps:.2f}",
                f"{cpu:.2f}",
                f"{mem:.2f}"
            ])

        print(f"Episodio {ep}: éxitos={success}, Bugs detected={bug_ids}")

    # Agregados
    avg_completions = np.mean(success_counts)
    avg_time = np.mean(episode_times)
    avg_steps = np.mean(steps_per_completion)
    success_rate = np.count_nonzero(success_counts) / num_episodes * 100
    total_bugs = sum(bug_counts)
    avg_fps = np.mean(fps_values)
    avg_cpu = np.mean(cpu_values)
    avg_mem = np.mean(mem_values)

    # Frecuencia de bugs
    bug_freq = Counter(found_bug_ids)

    # CSV resumen
    with open("metrics_report_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Métrica", "Valor"])
        writer.writerow(["Finalizaciones promedio por episodio", f"{avg_completions:.2f}"])
        writer.writerow(["Tiempo promedio por finalización (s)", f"{avg_time:.2f}"])
        writer.writerow(["Pasos promedio por finalización", f"{avg_steps:.2f}"])
        writer.writerow(["Tasa de éxito general (%)", f"{success_rate:.2f}"])
        writer.writerow(["Total de errores identificados", total_bugs])
        writer.writerow(["FPS promedio", f"{avg_fps:.2f}"])
        writer.writerow(["Porcentaje de CPU (%)", f"{avg_cpu:.2f}"])
        writer.writerow(["Memoria usada (MB)", f"{avg_mem:.2f}"])
        writer.writerow([])
        writer.writerow(["BugID", "Descripción", "Frecuencia"])
        for bug_id, freq in bug_freq.items():
            writer.writerow([bug_id, bug_catalog.get(bug_id), freq])

    # Consola resumen
    print("\n--- Métricas agregadas ---")
    print(f"Tasa de éxito general: {success_rate:.2f}%")
    print(f"Total de errores identificados: {total_bugs}")
    print("\n--- Frecuencia de bugs ---")
    for bug_id, freq in bug_freq.items():
        print(f"{bug_id} ({bug_catalog.get(bug_id)}): {freq}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de métricas reales con reglas de bug detection")
    parser.add_argument('--episodes', type=int, default=100, help='Número de episodios a evaluar')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')
    args = parser.parse_args()

    main(num_episodes=args.episodes, seed=args.seed)
