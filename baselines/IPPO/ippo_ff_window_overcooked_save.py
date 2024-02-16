""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
from jaxmarl.environments.multi_agent_env import DelayedObsWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
from itertools import product
import seaborn as sns

import matplotlib.pyplot as plt
import pickle

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_rollout_pair(config):

    def rollout_pair(rng, params):

        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)

        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, key_r, key_a = jax.random.split(rng, 3)

        init_x = jnp.zeros(env.observation_space().shape)
        init_x = init_x.flatten()
        
        network.init(key_a, init_x)

         # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config['NUM_PAIRS'])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # state_seq = [state]
        # rew_seq = []
        # while not done:
        #     key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        #     # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        #     # breakpoint()
        #     obs = {k: v.flatten() for k, v in obs.items()}

        #     pi_0, _ = network.apply(network_params1, obs["agent_0"])
        #     pi_1, _ = network.apply(network_params2, obs["agent_1"])

        #     actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        #     # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        #     # env_act = {k: v.flatten() for k, v in env_act.items()}

        #     # STEP ENV
        #     obs, state, reward, done, info = env.step(key_s, state, actions)
        #     print(done)
        #     done = done["__all__"]

        #     state_seq.append(state)
        #     rew_seq.append(reward)

        runner_state = (env_state, obsv, rng)

        def _env_step(runner_state, unused):
            env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            obs0 = last_obs['agent_0'].reshape((config['NUM_PAIRS'], -1))
            # jax.tree_map(lambda x : print(x.shape), params)
            # print(params['agent_0']['params']['Dense_0']['kernel'].shape)
            # print(obs0.shape)
            pi0, _ = jax.vmap(network.apply, in_axes=(0,0))(params['agent_0'], obs0)
            action0 = pi0.sample(seed=_rng)
            # env_act1 = {k:v.flatten() for k,v in action1.items()}
            
            rng, _rng = jax.random.split(rng)

            obs1 = last_obs['agent_1'].reshape((config['NUM_PAIRS'], -1))
            pi1, _ = jax.vmap(network.apply, in_axes=(0,0))(params['agent_1'], obs1)
            action1 = pi1.sample(seed=_rng)

            env_act = {'agent_0':action0, 'agent_1':action1}
            
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config['NUM_PAIRS'])
            
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                rng_step, env_state, env_act
            )
            info = jax.tree_map(lambda x: x.reshape((config['NUM_PAIRS'], -1)), info)
            transition = (done, info, env_state)
            runner_state = (env_state, obsv, rng)

            return runner_state, transition

        runner_state, transitions = jax.lax.scan(
            _env_step, runner_state, None, config['MAX_ROLLOUT_STEPS']
        )

        return transitions

    return rollout_pair

def get_rollout(train_state, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state.params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    rew_seq = []
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()
        obs = {k: v.flatten() for k, v in obs.items()}

        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)
        rew_seq.append(reward)


    return state_seq, rew_seq

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config['OBS_DELAY']:
        env = DelayedObsWrapper(env, delay=config['OBS_DELAY'])


    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env)
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        
        init_x = init_x.flatten()
        
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info
                    
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train



@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_window_overcooked_save")
def main(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]

    num_samples = 5
    rng = jax.random.PRNGKey(30)
    rng_trains = jax.random.split(rng, num_samples)
    with jax.disable_jit(False):
        train_vjit = jax.jit(jax.vmap(make_train(config), out_axes=0))
        outs = train_vjit(rng_trains)

    filename = f'{config["ENV_NAME"]}_{layout_name}_obsDelay{config["OBS_DELAY"]}_save'

    train_states = outs["runner_state"][0]
    print(outs["metrics"]["returned_episode_returns"].shape)
    # for i in range(num_samples):

    pickle.dump(train_states.params, open(f'{filename}_params{num_samples}.pkl', "wb"))

    for i in range(num_samples):
        plt.plot(outs["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1))
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

    num_rollouts = 100

    # take cartesian product of params
    def make_cartesian_and_flatten(array, agent, num_rollouts):
        expanded = np.zeros((num_samples, num_samples, num_rollouts) + array.shape[1:])
        expanded[:,:,:] = np.expand_dims(array, axis=(agent,2))
        return expanded.reshape((num_samples**2*num_rollouts,) + array.shape[1:])
    
    params = {}
    params['agent_0'] = jax.tree_map(lambda x : make_cartesian_and_flatten(x, 0, num_rollouts), train_states.params)
    params['agent_1'] = jax.tree_map(lambda x : make_cartesian_and_flatten(x, 1, num_rollouts), train_states.params)

    print('rolling out...')
    config['MAX_ROLLOUT_STEPS'] = 400
    config['NUM_SAMPLES'] = num_samples
    config['NUM_PAIRS'] = config['NUM_SAMPLES']**2 * num_rollouts

    rng = jax.random.PRNGKey(30)
    with jax.disable_jit(False):
        rollout_vjit = jax.jit(make_rollout_pair(config))
        outs = rollout_vjit(rng, params)

    dones, infos, states = outs
    rollout_returns = infos['returned_episode_returns']
    crossplay_rollouts = np.mean(np.sum(rollout_returns, axis=0), axis=-1).reshape(num_samples,num_samples,num_rollouts)

    crossplays = np.mean(crossplay_rollouts, axis=2)
    crossvars = np.var(crossplay_rollouts, axis=2)



    # crossplays = np.zeros((num_samples,num_samples))
    # for i,j in product(jnp.arange(num_samples), repeat=2):
    #     train_state1 = jax.tree_map(lambda x: x[i], train_states)
    #     train_state2 = jax.tree_map(lambda x: x[i], train_states)

    #     _, rew_seq = get_rollout_pair(train_state1, train_state2, config)
    #     rew_seq = [r['agent_0'] for r in rew_seq]
    #     crossplays[i,j] = np.sum(rew_seq)
    #     print(np.sum(rew_seq))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(crossplays, ax=axes[0], annot=True, linewidth=0.5, fmt='.1f')
    sns.heatmap(crossvars, ax=axes[1], annot=True, linewidth=0.5, fmt='.1f')

    plt.savefig(f'{filename}_crossplays.png')


    # for robustness: randomize initial states
    # look through lit for robustness stuff. Anything better than cooperatin w/o human data? If not, do random rollouts idea and zero out gradients so
    # that you only train after randomness
    # crossplays should be equal w/o observation delays


if __name__ == "__main__":
    main()