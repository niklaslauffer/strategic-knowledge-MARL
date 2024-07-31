""" 
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

from functools import partial
from typing import Dict, List, Optional, Tuple
from jaxmarl.environments.spaces import Discrete, Box
# from gymnax.environments.spaces import Box, Discrete

import chex
import jax
import jax.numpy as jnp
from flax import struct


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
    

class DualingLOLAEnv(MultiAgentEnv):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(
        self,
        base_env: MultiAgentEnv,
        K : int,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        assert base_env.num_agents == 2, "Only 2 player env is supported for now"
        self.K = K
        self.num_env = K * 3
        self.num_agents = self.num_env * base_env.num_agents
        self.base_env = base_env

        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.agents = []
        opt_prefixes = ["pair", "hp", "rp"]
        for opt_prefix in opt_prefixes:
            for i in range(K):
                for agent in base_env.agents:
                    b_obs_s = base_env.observation_space(agent)
                    b_act_s = base_env.action_space(agent)
                    # modify obs space with space for agent id
                    obs_s = Box(b_obs_s.low, b_obs_s.high, (K + 2 + b_obs_s.shape[0],), b_obs_s.dtype)
                    act_s = b_act_s

                    self.observation_spaces[f'{agent}_{opt_prefix}{i}'] = obs_s
                    self.action_spaces[f'{agent}_{opt_prefix}{i}'] = act_s
                    self.agents.append(f'{agent}_{opt_prefix}{i}')

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        vkey = jax.random.split(key, self.num_env)
        vobs, vstates = jax.vmap(self.base_env.reset, in_axes=(0))(vkey)
        obs = self.dict_batch_obs(vobs)

        return obs, vstates

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: chex.PRNGKey, vstates: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        vacts = dict()
        for agent in self.base_env.agents:
            # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
            temp_pairs = jnp.array([actions[f'{agent}_pair{i}'] for i in range(self.K)])
            temp_hp = jnp.array([actions[f'{agent}_hp{i}'] for i in range(self.K)])
            temp_rp = jnp.array([actions[f'{agent}_rp{i}'] for i in range(self.K)])
            vacts[agent] = jnp.concatenate((temp_pairs, temp_hp, temp_rp))

        vkey = jax.random.split(key, self.num_env)
        vobs, vnew_states, vrewards, vdones, vinfos = jax.vmap(self.base_env.step, in_axes=(0,0,0))(vkey, vstates, vacts)

        obs = self.dict_batch_obs(vobs)
        rewards = self.dict_batch_rewards(vrewards)
        dones = self.dict_batch_dones(vdones)
        infos = self.dict_batch_logging_info(vinfos)

        return obs, vnew_states, rewards, dones, infos

    def get_obs(self, vstate: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        vobs = jax.vmap(self.base_env.get_obs, in_axes=(0))(vstate)
        return self.dict_batch_obs(vobs)

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    def agent_id(self, agent, i, _type):
        # ago and partner share the same id (i.e., they share the same policy)
        if _type == 'pair':
            idx = i
        elif _type == 'hp':
            if agent == self.base_env.agents[0]:
                idx = i
            elif agent == self.base_env.agents[1]:
                idx = self.K
            else:
                raise ValueError("Invalid agent")
        elif _type == 'rp':
            if agent == self.base_env.agents[0]:
                idx = self.K+1
            elif agent == self.base_env.agents[1]:
                idx = i
            else:
                raise ValueError("Invalid agent")
            
        return jax.nn.one_hot(idx, self.K+2)
                
    def assign_reward(self, reward, agent, _type):
        # ago and partner share the same id (i.e., they share the same policy)
        if _type == 'pair':
            return reward * .5
        elif _type == 'hp':
            if agent == self.base_env.agents[0]:
                return -reward * .5
            elif agent == self.base_env.agents[1]:
                return reward * (1.0/self.K)
            else:
                raise ValueError("Invalid agent")
        elif _type == 'rp':
            if agent == self.base_env.agents[0]:
                return reward * (1.0/self.K)
            elif agent == self.base_env.agents[1]:
                return -reward * .5
            else:
                raise ValueError("Invalid agent")
            
        raise ValueError("Invalid type", _type)
               
    def dict_batch_obs(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for agent in self.base_env.agents:
            for i in range(self.K):
                _dict[f'{agent}_pair{i}'] = jnp.concatenate((self.agent_id(agent, i, 'pair'), _vector[agent][i]))
                _dict[f'{agent}_hp{i}'] = jnp.concatenate((self.agent_id(agent, i, 'hp'), _vector[agent][self.K+i]))
                _dict[f'{agent}_rp{i}'] = jnp.concatenate((self.agent_id(agent, i, 'rp'), _vector[agent][2*self.K+i]))
        return _dict
    
    def dict_batch_rewards(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for agent in self.base_env.agents:
            for i in range(self.K):
                _dict[f'{agent}_pair{i}'] = self.assign_reward(_vector[agent][i], agent, 'pair')
                _dict[f'{agent}_hp{i}'] = self.assign_reward(_vector[agent][self.K+i], agent, 'hp')
                _dict[f'{agent}_rp{i}'] = self.assign_reward(_vector[agent][2*self.K+i], agent, 'rp')
        return _dict
    
    def dict_batch_dones(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for agent in self.base_env.agents:
            for i in range(self.K):
                _dict[f'{agent}_pair{i}'] = _vector[agent][i]
                _dict[f'{agent}_hp{i}'] = _vector[agent][self.K+i]
                _dict[f'{agent}_rp{i}'] = _vector[agent][2*self.K+i]
        return _dict
        
    def dict_batch_logging_info(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for k,v in _vector.items():
            _dict[k] = v.reshape(-1)
        # metrics = ["returned_episode_returns", "returned_episode_lengths", "returned_episode"]
        # for metric in metrics:
        #     for i in range(self.K):
        #         _dict[f'pair{i}_{metric}'] = _vector[metric][i]
        #         _dict[f'hp{i}_{metric}'] = _vector[metric][self.K+i]
        #         _dict[f'rp{i}_{metric}'] = _vector[metric][2*self.K+i]
        return _dict


class DualingEnv(MultiAgentEnv):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(
        self,
        base_env: MultiAgentEnv,
        K : int,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        assert base_env.num_agents == 2, "Only 2 player env is supported for now"
        self.K = K
        self.num_env = K * 3
        self.num_agents = self.num_env * base_env.num_agents
        self.base_env = base_env

        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.agents = []
        opt_prefixes = ["pair", "hp", "rp"]
        for opt_prefix in opt_prefixes:
            for i in range(K):
                for agent in base_env.agents:
                    b_obs_s = base_env.observation_space(agent)
                    b_act_s = base_env.action_space(agent)
                    # modify obs space with space for agent id
                    obs_s = Box(b_obs_s.low, b_obs_s.high, (K + 2 + b_obs_s.shape[0],), b_obs_s.dtype)
                    act_s = b_act_s

                    self.observation_spaces[f'{agent}_{opt_prefix}{i}'] = obs_s
                    self.action_spaces[f'{agent}_{opt_prefix}{i}'] = act_s
                    self.agents.append(f'{agent}_{opt_prefix}{i}')

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        vkey = jax.random.split(key, self.num_env)
        vobs, vstates = jax.vmap(self.base_env.reset, in_axes=(0))(vkey)
        obs = self.dict_batch_obs(vobs)

        return obs, vstates

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: chex.PRNGKey, vstates: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        vacts = dict()
        for agent in self.base_env.agents:
            # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
            temp_pairs = jnp.array([actions[f'{agent}_pair{i}'] for i in range(self.K)])
            temp_hp = jnp.array([actions[f'{agent}_hp{i}'] for i in range(self.K)])
            temp_rp = jnp.array([actions[f'{agent}_rp{i}'] for i in range(self.K)])
            vacts[agent] = jnp.concatenate((temp_pairs, temp_hp, temp_rp))

        vkey = jax.random.split(key, self.num_env)
        vobs, vnew_states, vrewards, vdones, vinfos = jax.vmap(self.base_env.step, in_axes=(0,0,0))(vkey, vstates, vacts)

        obs = self.dict_batch_obs(vobs)
        rewards = self.dict_batch_rewards(vrewards)
        dones = self.dict_batch_dones(vdones)
        infos = self.dict_batch_logging_info(vinfos)
        # infos = vinfos

        return obs, vnew_states, rewards, dones, infos

    def get_obs(self, vstate: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        vobs = jax.vmap(self.base_env.get_obs, in_axes=(0))(vstate)
        return self.dict_batch_obs(vobs)

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    def agent_id(self, agent, i, _type):
        # ago and partner share the same id (i.e., they share the same policy)
        if _type == 'pair':
            idx = i
        elif _type == 'hp':
            if agent == self.base_env.agents[0]:
                idx = i
            elif agent == self.base_env.agents[1]:
                idx = self.K
            else:
                raise ValueError("Invalid agent")
        elif _type == 'rp':
            if agent == self.base_env.agents[0]:
                idx = self.K+1
            elif agent == self.base_env.agents[1]:
                idx = i
            else:
                raise ValueError("Invalid agent")
            
        return jax.nn.one_hot(idx, self.K+2)
                
    def assign_reward(self, reward, agent, _type):
        # ago and partner share the same id (i.e., they share the same policy)
        if _type == 'pair':
            return reward * .5
        elif _type == 'hp':
            if agent == self.base_env.agents[0]:
                return -reward * .5
            elif agent == self.base_env.agents[1]:
                return reward * (1.0/self.K)
            else:
                raise ValueError("Invalid agent")
        elif _type == 'rp':
            if agent == self.base_env.agents[0]:
                return reward * (1.0/self.K)
            elif agent == self.base_env.agents[1]:
                return -reward * .5
            else:
                raise ValueError("Invalid agent")
            
        raise ValueError("Invalid type", _type)
               
    def dict_batch_obs(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for agent in self.base_env.agents:
            for i in range(self.K):
                _dict[f'{agent}_pair{i}'] = jnp.concatenate((self.agent_id(agent, i, 'pair'), _vector[agent][i]))
                _dict[f'{agent}_hp{i}'] = jnp.concatenate((self.agent_id(agent, i, 'hp'), _vector[agent][self.K+i]))
                _dict[f'{agent}_rp{i}'] = jnp.concatenate((self.agent_id(agent, i, 'rp'), _vector[agent][2*self.K+i]))
        return _dict
    
    def dict_batch_rewards(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for agent in self.base_env.agents:
            for i in range(self.K):
                _dict[f'{agent}_pair{i}'] = self.assign_reward(_vector[agent][i], agent, 'pair')
                _dict[f'{agent}_hp{i}'] = self.assign_reward(_vector[agent][self.K+i], agent, 'hp')
                _dict[f'{agent}_rp{i}'] = self.assign_reward(_vector[agent][2*self.K+i], agent, 'rp')
        return _dict
    
    def dict_batch_dones(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for agent in self.base_env.agents:
            for i in range(self.K):
                _dict[f'{agent}_pair{i}'] = _vector[agent][i]
                _dict[f'{agent}_hp{i}'] = _vector[agent][self.K+i]
                _dict[f'{agent}_rp{i}'] = _vector[agent][2*self.K+i]
        return _dict
        
    def dict_batch_logging_info(self, _vector):
        # env 1 - K are for pairs, K+1 - 2K are for human prior, 2K+1 - 3K are for robot prior
        _dict = dict()
        for k,v in _vector.items():
            _dict[k] = v.reshape(-1)
        # metrics = ["returned_episode_returns", "returned_episode_lengths", "returned_episode"]
        # for metric in metrics:
        #     for i in range(self.K):
        #         _dict[f'pair{i}_{metric}'] = _vector[metric][i]
        #         _dict[f'hp{i}_{metric}'] = _vector[metric][self.K+i]
        #         _dict[f'rp{i}_{metric}'] = _vector[metric][2*self.K+i]
        return _dict



@struct.dataclass
class StateObs:
    state: State
    obs: Dict[str, chex.Array]

class OverridePlayer(MultiAgentEnv):
    def __init__(self, baseEnv, overridden_agents, co_network):
        # self.__class__ = type(baseEnv.__class__.__name__,
        #                       (self.__class__, baseEnv.__class__),
        #                       {})
        # self.__dict__ = baseEnv.__dict__
        self.baseEnv = baseEnv
        self.agents = [agent for agent in baseEnv.agents if agent not in overridden_agents]
        self.num_agents = baseEnv.num_agents - len(overridden_agents)
        self.overridden_agents = overridden_agents

        self.co_network = co_network

    @partial(jax.jit, static_argnums=(0,))
    def overridden_policy(self, key, params, obs):
        pi, _ = self.co_network.apply(params, obs.flatten())
        action = pi.sample(seed=key)
        return action


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        b_obs, b_state = self.baseEnv.reset(key)
        
        overridden_obs = {}
        for agent in self.overridden_agents:
            overridden_obs[agent] = b_obs.pop(agent)

        state_obs = StateObs(b_state, overridden_obs)
        return b_obs, state_obs
    
    @partial(jax.jit, static_argnums=(0,))
    def step_copolicy(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        co_params,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env_copolicy(key, state, actions, co_params)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    
    @partial(jax.jit, static_argnums=(0,))
    def step_env_copolicy(
        self, key: chex.PRNGKey, state_obs: State, actions: Dict[str, chex.Array], co_params
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        key, key_co = jax.random.split(key, 2)
        
        for agent in self.overridden_agents:
            actions[agent] = self.overridden_policy(key_co, co_params, state_obs.obs[agent])

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step(key, state_obs.state, actions)

        overridden_obs = {}
        for agent in self.overridden_agents:
            overridden_obs[agent] = obs_st.pop(agent)
            rewards.pop(agent)
            dones.pop(agent)
        
        state_obs = StateObs(states_st, overridden_obs)

        return obs_st, state_obs, rewards, dones, infos
    
    @partial(jax.jit, static_argnums=(0,))
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
        return self.baseEnv.observation_space(agent)

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
    
class OverrideTorchPlayer(MultiAgentEnv):
    def __init__(self, baseEnv, overridden_agents, co_network):
        # self.__class__ = type(baseEnv.__class__.__name__,
        #                       (self.__class__, baseEnv.__class__),
        #                       {})
        # self.__dict__ = baseEnv.__dict__
        self.baseEnv = baseEnv
        self.agents = [agent for agent in baseEnv.agents if agent not in overridden_agents]
        self.num_agents = baseEnv.num_agents - len(overridden_agents)
        self.overridden_agents = overridden_agents

        self.co_network = co_network

    @partial(jax.jit, static_argnums=(0,))
    def overridden_policy(self, key, idx, obs):
        logits = self.co_network(idx, obs.flatten())
        action = jax.random.categorical(key, logits)
        return action

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        b_obs, b_state = self.baseEnv.reset(key)
        
        overridden_obs = {}
        for agent in self.overridden_agents:
            overridden_obs[agent] = b_obs.pop(agent)

        state_obs = StateObs(b_state, overridden_obs)
        return b_obs, state_obs
    
    @partial(jax.jit, static_argnums=(0,))
    def step_copolicy(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        co_params,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env_copolicy(key, state, actions, co_params)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    
    @partial(jax.jit, static_argnums=(0,))
    def step_env_copolicy(
        self, key: chex.PRNGKey, state_obs: State, actions: Dict[str, chex.Array], co_params
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""

        key, key_co = jax.random.split(key, 2)
        
        for agent in self.overridden_agents:
            actions[agent] = self.overridden_policy(key_co, co_params, state_obs.obs[agent])

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step(key, state_obs.state, actions)

        overridden_obs = {}
        for agent in self.overridden_agents:
            overridden_obs[agent] = obs_st.pop(agent)
            rewards.pop(agent)
            dones.pop(agent)
        
        state_obs = StateObs(states_st, overridden_obs)

        return obs_st, state_obs, rewards, dones, infos
    
    @partial(jax.jit, static_argnums=(0,))
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
        return self.baseEnv.observation_space(agent)

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

class MultiAgentSlidingWindowEnv(MultiAgentEnv):
    
    def __init__(self, num_agents: int, window_size: int = 1) -> None:
        super().__init__(num_agents)
        self.window_size = window_size
    
    def window_reset(self, obs):
        return [obs] * self.window_size
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""
        obs, states, rewards, dones, infos = super().step(key, state, actions) 

    
@struct.dataclass
class StateWindowObs:
    state: State
    obs_window: List[chex.Array]
    most_recent_timestep: int

class DelayedObsWrapper(MultiAgentEnv):
    
    def __init__(self, baseEnv, delay):
        self.baseEnv = baseEnv
        self.num_agents = baseEnv.num_agents
        self.agents = baseEnv.agents
        self.window_size = delay

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], StateWindowObs]:
        obs, state = self.baseEnv.reset(key)
        
        dummy_obs, timestep = self.update_obs_dict(obs, -1, dummy=True)
        obs_window = [dummy_obs]
        for _ in range(self.window_size - 1):
            new_dummy_obs, timestep = self.update_obs_dict(obs, timestep, dummy=True)
            obs_window.append(new_dummy_obs)
        new_current_obs, timestep = self.update_obs_dict(obs, timestep)
        obs_window.append(new_current_obs)
        # obs_window = [dummy_obs] * (self.window_size - 1) + [obs]
        state = StateWindowObs(state, obs_window, timestep)
        return obs_window[0], state

    def update_obs_dict(self, obs_dict, prev_timestep, dummy=False):
        new_obs = {}
        for agent, observ in obs_dict.items():
            if dummy:
                observ = jnp.zeros_like(observ)
            new_dummy_observ, timestep = self.add_timestep_dimension(observ, prev_timestep)
            new_obs[agent] = new_dummy_observ
        return new_obs, timestep
    
    def add_timestep_dimension(self, observation_array, prev_timestep):
        # assert prev_timestep < 255
        timestep_layer = jnp.zeros(observation_array.shape[:-1])
        new_timestep_layer = timestep_layer.at[tuple(0 for _ in range(timestep_layer.ndim))].set(prev_timestep + 1)
        return jnp.concatenate([observation_array, jnp.expand_dims(new_timestep_layer, -1)], axis=-1), prev_timestep + 1
    
        
    @partial(jax.jit, static_argnums=(0,))
    def step_copolicy(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        co_params,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env_copolicy(key, state, actions, co_params)

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
        self, key: chex.PRNGKey, state_window_obs: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # import pdb; pdb.set_trace()

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step(key, state_window_obs.state, actions)
        obs_st, new_timestep = self.update_obs_dict(obs_st, state_window_obs.most_recent_timestep)
        new_obs_window = state_window_obs.obs_window[1:] + [obs_st]
        new_state_window_obs = StateWindowObs(states_st, new_obs_window, new_timestep)
        curr_obs = new_state_window_obs.obs_window[0]
        return curr_obs, new_state_window_obs, rewards, dones, infos
    
    def step_env_copolicy(
        self, key: chex.PRNGKey, state_window_obs: State, actions: Dict[str, chex.Array], coparams
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # import pdb; pdb.set_trace()

        obs_st, states_st, rewards, dones, infos = self.baseEnv.step_copolicy(key, state_window_obs.state, actions, coparams)
        obs_st, new_timestep = self.update_obs_dict(obs_st, state_window_obs.most_recent_timestep)
        new_obs_window = state_window_obs.obs_window[1:] + [obs_st]
        new_state_window_obs = StateWindowObs(states_st, new_obs_window, new_timestep)
        curr_obs = new_state_window_obs.obs_window[0]
        return curr_obs, new_state_window_obs, rewards, dones, infos

    def observation_space(self, agent: str=''):
        """Observation space for a given agent."""
        assert isinstance(self.baseEnv.observation_space(agent), Box)
        new_low = self.baseEnv.observation_space(agent).low
        new_high = self.baseEnv.observation_space(agent).high
        new_shape = list(self.baseEnv.observation_space(agent).shape)
        new_shape[-1] += 1
        return Box(new_low, new_high, tuple(new_shape))


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
    


class ConcatenatePlayerSpaces(MultiAgentEnv):
    def __init__(self, baseEnv):
        assert baseEnv.num_agents == 2, "Concatenation currently only supported for two agent games"
        self.baseEnv = baseEnv
        self.agents = baseEnv.agents
        self.num_agents = baseEnv.num_agents
        agent_0 = self.baseEnv.agents[0]
        conc_n = sum([x.n for x in baseEnv.action_spaces.values()])
        self.action_spaces = {a: Discrete(conc_n) for a in baseEnv.agents}
        base_obs = baseEnv.observation_spaces[agent_0]
        conc_shape = sum([x.shape[0] for x in baseEnv.observation_spaces.values()])
        self.observation_spaces = {a: Box(base_obs.low, base_obs.high, (conc_shape,), base_obs.dtype) for a in baseEnv.agents}

    @partial(jax.jit, static_argnums=(0,))
    def concatenate_obs(self, obs):
        agent_0 = self.baseEnv.agents[0]
        agent_1 = self.baseEnv.agents[1]
        return {agent_0: jnp.concatenate((obs[agent_0], jnp.zeros_like(obs[agent_1]))), agent_1: jnp.concatenate((jnp.zeros_like(obs[agent_0]), obs[agent_1]))}
        # return {agent_0: jnp.concatenate((obs[agent_0], obs[agent_1])), agent_1: jnp.concatenate((obs[agent_0], obs[agent_1]))}
    
    @partial(jax.jit, static_argnums=(0,))
    def truncate_actions(self, actions):
        agent_0 = self.baseEnv.agents[0]
        agent_1 = self.baseEnv.agents[1]
        cutoff = self.baseEnv.action_space(agent_0).n
        agent_0_act = jnp.where(actions[agent_0] >= cutoff, 0, actions[agent_0])
        agent_1_act = jnp.where(actions[agent_1] < cutoff, 0, actions[agent_1] - cutoff)
        # jax.debug.print("0: trunc {x} to {y}", x=actions[agent_0], y=agent_0_act)
        # jax.debug.print("1: trunc {x} to {y}", x=actions[agent_1], y=agent_1_act)

        return {agent_0: agent_0_act.squeeze(), agent_1: agent_1_act.squeeze()}
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        obs, state = self.baseEnv.reset(key)
        conc_obs = self.concatenate_obs(obs)
        return conc_obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        truncated_actions = self.truncate_actions(actions)
        obs, states, rewards, dones, infos = self.baseEnv.step(key, state, truncated_actions)
        conc_obs = self.concatenate_obs(obs)
        return conc_obs, states, rewards, dones, infos

    