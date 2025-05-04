if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from custom_game_env_nivel2 import CustomGameEnv2  # Importa el entorno personalizado

    # Configuración del entorno
    exe_path = "C:\\Users\\ghost\\Documents\\TESIS-REPO\\Tesis\\Nivel2\\Abby's Redemption2SIN"
    ports = [8001, 8002, 8003, 8004]
    n_envs = len(ports)
    # env = CustomGameEnv2(exe_path)

    # Función generadora de entornos
    def make_env(port):
        def _init():
            env = CustomGameEnv2(exe_path=exe_path, api_port=port)
            return env
        return _init

    # Crear entornos paralelos
    env_fns = [make_env(port) for port in ports]
    vec_env = SubprocVecEnv(env_fns)

    # Opcional: Verificar un entorno individual
    check_env(env_fns[0]())

    # Verificar el entorno para asegurarse de que es compatible con Stable Baselines3
    #check_env(env)

    # Cargar el modelo preentrenado
    model = PPO.load("dqn_custom_game_model_PPO", env=vec_env, device="cuda")

    # Modificar parámetros del modelo si es necesario
    # Ejemplo de modificación de parámetros:
    #model.learning_rate = 5e-4  # Cambiar la tasa de aprendizaje
    #model.exploration_fraction = 0.2  # Cambiar la fracción de exploración
    #model.exploration_final_eps = 0.1  # Cambiar el valor final de epsilon

    class RewardCallback(BaseCallback):
        def __init__(self, verbose=1):
            super(RewardCallback, self).__init__(verbose)
            self.episode_rewards = []  # Para almacenar recompensas acumuladas por episodio
            self.episode_steps = []  # Para almacenar el número de pasos por episodio
            self.success_per_episode = []
            self.completion_count = 0  # Para contar episodios completados
            self.action_counts = [0] * vec_env.action_space.n  # Contar el uso de cada acción
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
                current_lives = self.locals.get("infos", [{}])[0].get("lives", 3)            
                print(f"Episodio terminado. Recompensa acumulada: {self.current_reward}, Pasos: {self.current_steps}, Vidas restantes: {current_lives}")
                # Contar episodios completados si aplica
                if self.locals.get("infos", [{}])[0].get("is_success", False):
                    self.completion_count += 1
                self.current_reward = 0
                self.current_steps = 0

            return True

        def plot_metrics(self):
            # Gráfico de recompensas por episodio
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            episodes = list(range(1, len(self.episode_rewards) + 1))
            plt.plot(episodes, self.episode_rewards)
            plt.xlabel("Episodios")
            plt.ylabel("Recompensa Total")
            plt.title("Progreso del Agente en Recompensas")

            # Gráfico de número de pasos por episodio
            plt.subplot(1, 3, 2)
            plt.plot(episodes, self.episode_steps)
            plt.xlabel("Episodios")
            plt.ylabel("Número de Pasos")
            plt.title("Número de Pasos por Episodio")

            # Éxitos acumulados
            plt.subplot(1, 3, 3)
            episodes_success = list(range(1, len(self.success_per_episode) + 1))
            plt.plot(episodes_success, np.cumsum(self.success_per_episode), label="Éxitos acumulados", color='green')
            plt.xlabel("Episodios")
            plt.ylabel("Total acumulado")
            plt.title("Episodios Exitosos")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig("training_metrics.png")  # Guarda la figura como imagen
            plt.close()  # Cierra la figura para liberar memoria

            # Mostrar estadísticas de métricas adicionales
            print(f"\nEstadísticas de Entrenamiento:")
            print(f"Recompensa Promedio por Episodio: {np.mean(self.episode_rewards):.2f}")
            print(f"Número Total de Episodios Completados: {self.completion_count}")
            print(f"Frecuencia de Uso de Acciones: {self.action_counts}")

    # Crear una instancia del callback
    reward_callback = RewardCallback()

    # Continuar entrenando el agente con el modelo cargado y el callback
    model.learn(total_timesteps=5000, callback=reward_callback)

    # Guardar el modelo actualizado
    model.save("dqn_custom_game_model_updated_PPO")

    # Graficar las métricas después del entrenamiento
    reward_callback.plot_metrics()

    # Cerrar el entorno
    #env.close()

    vec_env.close()


