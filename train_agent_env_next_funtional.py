# train_agent_continue.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from custom_game_env import CustomGameEnv1  # Asegúrate de tener importado el entorno personalizado

# Configuración del entorno
exe_path = "C:\\Users\\ghost\\Documents\\TESIS\\Nivel1\\Abby's Redemption"
env = CustomGameEnv1(exe_path)

# Verificar el entorno para asegurarse de que es compatible con Stable Baselines3
check_env(env)

# Cargar el modelo guardado
model = DQN.load("dqn_custom_game_model", env=env)

# Callback para registrar las recompensas por episodio
class RewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []  # Para almacenar recompensas acumuladas por episodio
        self.current_reward = 0    # Recompensa actual del episodio

    def _on_step(self) -> bool:
        # Acumular recompensa del paso actual
        self.current_reward += self.locals["rewards"][0]

        # Verificar si el episodio ha terminado
        if self.locals["dones"][0]:  
            self.episode_rewards.append(self.current_reward)
            print(f"Episodio terminado. Recompensa acumulada: {self.current_reward}")
            self.current_reward = 0  # Reiniciar la recompensa acumulada para el siguiente episodio

        return True  # Continuar el entrenamiento

# Crear una instancia del callback
reward_callback = RewardCallback()

# Continuar el entrenamiento del agente con model.learn y el callback
model.learn(total_timesteps=10000, callback=reward_callback)

# Guardar el modelo nuevamente después de continuar el entrenamiento
model.save("dqn_custom_game_model_2")

# Visualización de las recompensas acumuladas durante el entrenamiento
def plot_rewards(episode_rewards):
    episodes = list(range(1, len(episode_rewards) + 1))
    plt.plot(episodes, episode_rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Total")
    plt.title("Progreso del Agente en Recompensas")
    plt.show()

# Graficar las recompensas al final del entrenamiento
plot_rewards(reward_callback.episode_rewards)

# Cerrar el entorno
env.close()