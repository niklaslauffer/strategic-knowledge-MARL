""" 
Based on PureJaxRL Implementation of PPO
"""

import pickle
from functools import partial
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
from baselines.IPPO.mer_ff import ActorCritic, Transition, make_train
from jaxmarl.environments.multi_agent_env import OverridePlayer, OverridePlayer2
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.normal_form_visualizer import animate_triangle
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.wrappers.baselines import LogWrapperCoPolicy


def extract_normal_policy(config, params):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    co_network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    co_network.init(key_a, init_x)

    env = OverridePlayer(env, [env.agents[0]], co_network)

    action_space_size = env.action_space().n
    network = ActorCritic(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    dummy_obs = np.array([[0],[0]])

    pis, _ = jax.vmap(network.apply, in_axes=(0,0))(params, dummy_obs)

    return pis

def test_normal_1():

    num_trials = 10

    config = OmegaConf.load('tests/mer/test_config/mer_ff_normal_1.yaml')
    config = OmegaConf.to_container(config) 

    payoffs = jnp.array([
        [2,-1,-1],
        [-1,0,1],
        [-1,1,0]
    ])
    config["ENV_KWARGS"]["payoffs"] = payoffs
    config["COPARAMS_SOURCE"] = 'file'

    payoff_string = str(payoffs).replace('\n','')
    config['COPARAMS_FILE'] = f"{config['ENV_NAME']}_{payoff_string}_save_params50.pkl"

    rng = jax.random.PRNGKey(30)
    rngs = jax.random.split(rng, num_trials)
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)

    train_state = out["runner_state"][0]

    ### Evaluate
    pis = jax.vmap(extract_normal_policy, in_axes=(None, 0))(config, train_state.params)

    for pi in pis.probs:
        print(pi)
    
    # check convergence in each trial
    for prob in pis.probs:
        assert(prob[0][0] >= .99 and prob[1][1]+prob[1][2] >= .99
           or prob[1][0] >= .99 and prob[0][1]+prob[0][2] >= .99)

def probs2params(key, probs, config):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space().n
    network = ActorCritic(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    init_params = network.init(key_a, init_x)
    # print(init_params)
    params = jax.tree_map(lambda x :jnp.zeros(probs.shape[0:1] + x.shape), init_params)
    params['params']['Dense_2']['bias'] = jnp.log(probs)

    # dummy_obs = np.array([[1],[1],[1],[1]])
    # pis, _ = jax.vmap(network.apply, in_axes=(0,0))(params, dummy_obs)

    return params

def test_normal_2():

    rng = jax.random.PRNGKey(30)

    num_trials = 10

    config = OmegaConf.load('tests/mer/test_config/mer_ff_normal_2.yaml')
    config = OmegaConf.to_container(config) 

    payoffs = jnp.array([
        [2,-1,-1],
        [-1,0,1],
        [-1,1,0]
    ])
    probs = jnp.array([[.98,.01,.01],
                       [.98,.01,.01],
                       [.01,.98,.01],
                       [.01,.01,.98]])

    config["ENV_KWARGS"]["payoffs"] = payoffs
        
    rng, _rng = jax.random.split(rng, 2)
    coparams = probs2params(_rng, probs, config)

    config["COPARAMS_SOURCE"] = 'pytree'
    config["COPARAMS_BATCH"] = coparams

    rngs = jax.random.split(rng, num_trials)
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)

    train_state = out["runner_state"][0]

    ### Evaluate
    pis = jax.vmap(extract_normal_policy, in_axes=(None, 0))(config, train_state.params)
    # check convergence in each trial
    for prob in pis.probs:
        assert((prob[0][0] >= .95 and prob[1][1]+prob[1][2] >= .95)
           or (prob[1][0] >= .95 and prob[0][1]+prob[0][2] >= .95))


if __name__ == "__main__":
    test_normal_1()    