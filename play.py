import gymnasium as gym
from stable_baselines3 import DQN
import os
import ale_py
import tensorflow as tf
import torch


class BreakoutAgent:
    """Handling the setup and execution for a Breakout game agent"""

    def __init__(self, model_file="models/policy.zip", display_mode="human"):
        """Setting up the agent, including environment and model loading"""
        gym.register_envs(ale_py)
        self.env = gym.make("ALE/Breakout-v5", render_mode=display_mode)
        self.model_file = model_file
        self.model = self.load_model()

    def load_model(self):
        """Loading the DQN agent"""
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file not found at: {self.model_file}")
        try:
            print(f"Attempting to load model from: {self.model_file}")
            model = DQN.load(self.model_file)
            print(f"Model loaded successfully from: {self.model_file}")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError("Check the model file")

    def execute(self, episodes=5):
        """Running the trained agent for 5 episodes"""
        for ep in range(episodes):
            observation, _ = self.env.reset()
            total_reward = 0
            is_done = False
            steps = 0

            while not is_done:
                # Selecting the action
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                is_done = terminated or truncated

            print(
                f"Episode {ep + 1}: Total Score: {total_reward} | Total Steps: {move_count}"
            )

        self.env.close()

    def convert_to_h5(self, h5_filepath="models/policy.h5"):
        """Converting the trained model (PyTorch) to an HDF5 (.h5) file for compatibility"""
        print("Initiating PyTorch to HDF5 model conversion...")

        # Accessing the PyTorch layers
        torch_model = self.model.policy.q_net

        # Creating a Keras sequential model and converting each layer
        keras_model = tf.keras.Sequential()
        for layer in torch_model.children():
            if isinstance(layer, torch.nn.Conv2d):
                keras_model.add(
                    tf.keras.layers.Conv2D(
                        filters=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        strides=layer.stride,
                        activation="relu",
                        padding="valid",
                        input_shape=(210, 160, 3),
                    )
                )
            elif isinstance(layer, torch.nn.Linear):
                keras_model.add(
                    tf.keras.layers.Dense(
                        units=layer.out_features,
                        activation="relu"
                        if layer.weight.shape[0] > layer.weight.shape[1]
                        else None,
                    )
                )
            elif isinstance(layer, torch.nn.Flatten):
                keras_model.add(tf.keras.layers.Flatten())

        # Saving the resulting Keras model
        keras_model.save(h5_filepath)
        print(f"HDF5 model saved successfully at: {h5_filepath}")


def run_agent():
    """Initialization of the Breakout agent"""
    trained_model = "models/policy.zip"
    breakout_agent = BreakoutAgent(model_file=trained_model)

    breakout_agent.execute(episodes=5)

    save_as_h5 = input("Save model in .h5 format? (yes/no): ").strip().lower()
    if save_as_h5 in ("yes", "y"):
        breakout_agent.convert_to_h5(h5_filepath="models/policy.h5")


if __name__ == "__main__":
    run_agent()
