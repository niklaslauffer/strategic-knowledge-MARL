""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapperCoPolicy
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.multi_agent_env import OverridePlayer, OverridePlayer2
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.viz.normal_form_visualizer import animate_triangle
import hydra
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import pickle

from baselines.IPPO.mer_ff import ActorCritic, Transition, make_train


def get_rollout(train_state, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)

    filename = f'{config["ENV_NAME"]}_{config["LAYOUT_NAME"]}_save'
    co_params = pickle.load(open(f'{filename}_params.pkl', "rb"))

    co_network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_p, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    co_network.init(key_a, init_x)
    def policy_fn(obs):
        pi, _ = co_network.apply(co_params, obs.flatten())
        action = pi.sample(seed=key_p)
        return action

    override_map = {env.agents[0] : policy_fn}
    env = OverridePlayer(env, [env.agents[0]], co_network)

    action_space_size = env.action_space().n
    network = ActorCritic(action_space_size, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state.params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()
        obs = {k: v.flatten() for k, v in obs.items()}

        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {"agent_1": pi_1.sample(seed=key_a1)}
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step_copolicy(key_s, state, actions, co_params)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


@hydra.main(version_base=None, config_path="config", config_name="mer_ff_normal")
def main(config):
    np.set_printoptions(threshold=np.inf)
    # idea: do a warm start so that all policies gain basic competency level
    config = OmegaConf.to_container(config) 
    payoffs = jnp.array([
        [2,-1,-1],
        [-1,0,1],
        [-1,1,0]
    ])
    config["ENV_KWARGS"]["payoffs"] = payoffs
    rng = jax.random.PRNGKey(30)
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)

    filename = f'{config["ENV_NAME"]}_{payoffs}_mer'
    
    grad_mask = out["metrics"]["grad_mask"].reshape(out["metrics"]["returned_episode_returns"].shape)
    masked_returns = out["metrics"]["returned_episode_returns"].at[grad_mask==0.0].set(np.nan)
    mean_returns = jnp.nanmean(masked_returns, axis=-1).reshape(-1, config['NUM_PARTICLES'])

    lines = jnp.swapaxes(mean_returns, 0, 1)
    for l in lines:
        plt.scatter(jnp.arange(len(l)), l)
        # print(l)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

    pi = out["metrics"]["action_probs"]
    pi = pi.reshape(pi.shape[:2] + (config["NUM_PARTICLES"], config['NUM_COPOLICIES'],config["NUM_ENVS"], -1))
    action_seq = pi[:,0,:,0,0,:]

    point_colors = ['green', 'red']
    point_markers = ['o', 'o']

    animate_triangle(action_seq, point_colors, point_markers, save_gif=f'{filename}.gif')


if __name__ == "__main__":
    main()