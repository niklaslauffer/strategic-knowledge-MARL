""" 
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

import jax
import jax.numpy as jnp
from typing import Dict
import chex
from functools import partial
from flax import struct
from typing import Tuple, Optional


@struct.dataclass
class State:
    done: chex.Array
    step: int


class MultiAgentEnv(object):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(
        self,
        num_agents: int,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError
    
@struct.dataclass
class StateObs:
    state: State
    obs: Dict[str, chex.Array]

class OverridePlayer(MultiAgentEnv):
    def __init__(self, baseEnv, agent_override):
        # self.__class__ = type(baseEnv.__class__.__name__,
        #                       (self.__class__, baseEnv.__class__),
        #                       {})
        # self.__dict__ = baseEnv.__dict__
        self.baseEnv = baseEnv
        self.agents = [agent for agent in baseEnv.agents if agent not in agent_override]
        self.num_agents = baseEnv.num_agents - len(agent_override)
        self.overridden_agents = agent_override.keys()
        self.overridden_policies = agent_override


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        b_obs, b_state = self.baseEnv.reset(key)
        
        overridden_obs = {}
        for agent in self.overridden_agents:
            overridden_obs[agent] = b_obs.pop(agent)

        state_obs = StateObs(b_state, overridden_obs)
        return b_obs, state_obs

    def step_env(
        self, key: chex.PRNGKey, state_obs: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        for agent in self.overridden_agents:
            actions[agent] = self.overridden_policies[agent](state_obs.obs[agent])

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step(key, state_obs.state, actions)

        overridden_obs = {}
        for agent in self.overridden_agents:
            overridden_obs[agent] = obs_st.pop(agent)
            rewards.pop(agent)
            dones.pop(agent)
        
        state_obs = StateObs(states_st, overridden_obs)

        return obs_st, state_obs, rewards, dones, infos

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        base_obs = self.baseEnv.get_obs(state)
        for agent in self.overridden_agents:
            base_obs.pop(agent)
        return base_obs

    def observation_space(self, agent: str=''):
        """Observation space for a given agent."""
        return self.baseEnv.observation_space()

    def action_space(self, agent: str=''):
        """Action space for a given agent."""
        return self.baseEnv.action_space(agent)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        return self.baseEnv.agent_classes()

