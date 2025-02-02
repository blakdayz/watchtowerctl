import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Union


class RLModel:
    def __init__(
        self,
        state_space: dict,
        action_space: dict,
        learning_rate: float = 0.001,
        n_steps: int = 2048,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.95,
        gaussian_fuzz: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.25,
        max_grad_norm: Union[int, float] = 0.5,
        clip_range: Union[int, float] = 0.2,
    ):
        self.model = PPO(
            policy=Ml,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            clip_range=clip_range,
        )
        self.action_space = action_space
        self.observation_space = state_space
        self.env = DummyVecEnv(
            [
                lambda: (
                    np.random.randint(
                        0, high=self.observation_space["n_agents"], size=(1,)
                    )
                    for _ in range(n_steps)
                )
            ]
        )

    def learn_from_experience(self, experiences):
        observations, rewards, actions, next_obs, dones = experiences
        self.model.learn(
            transition_dict={
                "fwd": observations,
                "a": actions.squeeze(),
                "r": rewards,
                "nb": next_obs,
            }
        )

    def predict(self, observation):
        return self.model.predict(observation)

    def reset(self) -> None:
        # Reset the environment if needed
        pass

    @property
    def action_space_shape(self) -> tuple:
        return (self.action_space["n_actions"],)

    @property
    def observation_space_shape(self) -> tuple:
        return self.observation_space["n_agents"] * (
            self.observation_space["n_continuous_state_features"],
        )
