import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete
import chex
from typing import Tuple, Dict, Any
from functools import partial    


class NormalForm(MultiAgentEnv):
    def __init__(
        self,
        payoffs,
    ):
        self.num_agents = len(payoffs.shape)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.payoffs = payoffs

        self.action_spaces = {agent: Discrete(payoffs.shape[i]) for i, agent in enumerate(self.agents)}
        self.all_action_space = Discrete(payoffs.shape[0])

        fixed_obs_array = [-1,1]
        self.fixed_obs = {a: jnp.array([o]) for a,o in zip(self.agents, fixed_obs_array)}

        self.observation_spaces = {i: Discrete(0) for i in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Any]:
        obs = self.fixed_obs
        state = jnp.array([0])
        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: Any, actions: dict):
        # get the actions as array        

        actions = tuple(actions[a] for a in self.agents)
        noise = jax.random.normal(key) * 0.001
        reward = self.payoffs[actions] + noise
        obs = self.fixed_obs
        state = jnp.array([0])
        rewards = {a: reward for a in self.agents}
        dones = {a: True for a in self.agents + ["__all__"]}
        info = {}

        return obs, state, rewards, dones, info

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
    
    def action_space(self, agent_id="") -> Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""

        return self.action_spaces[agent_id]

    def observation_space(self, agent_id="") -> Discrete:
        """Observation space of the environment."""
        return Discrete(self.num_agents)


