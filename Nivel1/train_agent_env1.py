import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from custom_game_env_nivel1 import CustomGameEnv1  # Importa el entorno personalizado

# Configuración del entorno
exe_path = "C:\\Users\\ghost\\Documents\\TESIS-REPO\\Tesis\\Nivel1\\Abby's Redemption1SIN"
env = CustomGameEnv1(exe_path)

# Verificar el entorno para asegurarse de que es compatible con Stable Baselines3
check_env(env)

# Configuración del agente DQN
model = PPO("MlpPolicy", env, verbose=1, buffer_size=100000, learning_rate=1e-4, batch_size=128,
            exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=1000, gamma=0.99, train_freq=4, device="cuda")

class RewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []  # Para almacenar recompensas acumuladas por episodio
        self.episode_steps = []  # Para almacenar el número de pasos por episodio
        self.completion_count = 0  # Para contar episodios completados
        self.action_counts = [0] * env.action_space.n  # Contar el uso de cada acción
        self.current_reward = 0
        self.current_steps = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        self.current_steps += 1
        # Contar la acción realizada
        action = self.locals["actions"][0]
        self.action_counts[action] += 1

        if self.locals["dones"][0]:  # Cuando el episodio termina
            self.episode_rewards.append(self.current_reward)
            self.episode_steps.append(self.current_steps)
            print(f"Episodio terminado. Recompensa acumulada: {self.current_reward}, Pasos: {self.current_steps}")
            # Contar episodios completados si aplica
            if self.locals.get("infos", [{}])[0].get("is_success", False):
                self.completion_count += 1
            self.current_reward = 0
            self.current_steps = 0

        return True

    def plot_metrics(self):
        # Gráfico de recompensas por episodio
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        episodes = list(range(1, len(self.episode_rewards) + 1))
        plt.plot(episodes, self.episode_rewards)
        plt.xlabel("Episodios")
        plt.ylabel("Recompensa Total")
        plt.title("Progreso del Agente en Recompensas")

        # Gráfico de número de pasos por episodio
        plt.subplot(1, 2, 2)
        plt.plot(episodes, self.episode_steps)
        plt.xlabel("Episodios")
        plt.ylabel("Número de Pasos")
        plt.title("Número de Pasos por Episodio")

        plt.tight_layout()
        plt.show()

        # Mostrar estadísticas de métricas adicionales
        print(f"\nEstadísticas de Entrenamiento:")
        print(f"Recompensa Promedio por Episodio: {np.mean(self.episode_rewards):.2f}")
        print(f"Número Total de Episodios Completados: {self.completion_count}")
        print(f"Frecuencia de Uso de Acciones: {self.action_counts}")

# Crear una instancia del callback
reward_callback = RewardCallback()

# Entrenar el agente con model.learn y el callback (prueba inicial con 5000 steps)
model.learn(total_timesteps=5000, callback=reward_callback)

# Guardar el modelo entrenado
model.save("dqn_custom_game_model_PPO")

# Graficar las métricas después del entrenamiento
reward_callback.plot_metrics()

# Cerrar el entorno
env.close()


