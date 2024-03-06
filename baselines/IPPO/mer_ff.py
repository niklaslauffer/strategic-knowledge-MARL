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
from data import DATA_DIR
from jaxmarl.environments.multi_agent_env import OverridePlayer, OverridePlayer2
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.normal_form_visualizer import animate_triangle
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.wrappers.baselines import LogWrapperCoPolicy


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
    action_probs: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

# def vbatchify(x: dict, agent_list, _dim):
#     x = jnp.stack([x[a] for a in agent_list])
#     return x.reshape((*_dim, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def unbatchifyp(x: jnp.ndarray, agent_list, env_dim, num_actors):
    x = x.reshape((num_actors, *env_dim, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
   
def make_train(config):
    base_env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])


    # load_payoffs = jnp.array([
    #     [2,-1],
    #     [-1,1]
    # ])
    if config['COPARAMS_SOURCE'] == 'file':
        filepath = config['COPARAMS_FILE']
        load_params_batch = pickle.load(open(filepath, "rb"))
    elif config['COPARAMS_SOURCE'] == 'pytree':
        load_params_batch = config['COPARAMS_BATCH']
    num_samples = load_params_batch['params']['Dense_2']['bias'].shape[0]

    def sample_co_params(_rng,n):
        get_ith_params = lambda i : jax.tree_util.tree_map(lambda x : x[i], load_params_batch)
        copolicy_rand_ints = jax.random.randint(_rng, shape=(n,), minval=0, maxval=num_samples)
        co_params = jax.vmap(get_ith_params, in_axes=(0,))(copolicy_rand_ints)
        return co_params

    # def sample_overridden_env(rng, env):
    #     rng, _rng = jax.random.split(rng, 2)

    #     rand_int = jax.random.randint(_rng, shape=(1,), minval=0, maxval=num_samples).squeeze()
    #     sampled_params = get_ith_params(rand_int)
        
    #     network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    #     rng, key_p, key_a = jax.random.split(rng, 3)

    #     init_x = jnp.zeros(env.observation_space().shape)
    #     init_x = init_x.flatten()

    #     network.init(key_a, init_x)
    #     def policy_fn(obs):
    #         pi, _ = network.apply(sampled_params, obs.flatten())
    #         action = pi.sample(seed=key_p)
    #         return action

    #     override_map = {env.agents[0] : policy_fn}
    #     out_env = OverridePlayer(env, override_map)
    #     return out_env
         

    # network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    # key = jax.random.PRNGKey(0)
    # key, key_p, key_a = jax.random.split(key, 3)

    # init_x = jnp.zeros(env.observation_space().shape)
    # init_x = init_x.flatten()

    # network.init(key_a, init_x)
    # def policy_fn(obs):
    #     pi, _ = network.apply(loaded_params, obs.flatten())
    #     action = pi.sample(seed=key_p)
    #     return action

    # override_map = {env.agents[0] : policy_fn}
    # env = OverridePlayer(env, override_map)
    
    # TODO change the 1 below to allow multiple ego
    config["NUM_ACTORS"] = 1 * config["NUM_ENVS"] * config['NUM_COPOLICIES'] * config["NUM_PARTICLES"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
            
        co_network = ActorCritic(base_env.action_space(base_env.agents[0]).n, activation=config["ACTIVATION"])
        rng, key_a = jax.random.split(rng, 2)

        init_x = jnp.zeros(base_env.observation_space(base_env.agents[0]).shape)
        init_x = init_x.flatten()

        co_network.init(key_a, init_x)
        
        env = OverridePlayer(base_env, [base_env.agents[0]], co_network)
        env = LogWrapperCoPolicy(env)


        # INIT NETWORK
        # now agent 0 is the ego agent since we override the previous agent earlier
        network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        
        init_x = init_x.flatten()

        rng, _rng = jax.random.split(rng)
        particle_rng = jax.random.split(_rng, config["NUM_PARTICLES"])
        network_params_n = jax.vmap(network.init, in_axes=(0,None))(particle_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state_n = jax.vmap(lambda a,b,c : TrainState.create(
            apply_fn=a,
            params=b,
            tx=c,
        ), in_axes=(None,0,None))(network.apply,network_params_n,tx)
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ACTORS"])
        obsv_n, env_state_n = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):

            train_state, env_state, obsv, rng = runner_state

            rng, _rng = jax.random.split(rng, 2)
            co_params = sample_co_params(_rng, config["NUM_COPOLICIES"])
            
            # rng, _rng = jax.random.split(rng)
            # reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            # obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

            runner_state = (train_state, env_state, obsv, rng)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                ### TODO write unit test for this function
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents])
                obs_batch = obs_batch.reshape((config["NUM_PARTICLES"], config['NUM_COPOLICIES']*config["NUM_ENVS"], -1))
                
                # pi, value = network.apply(train_state.params, obs_batch)
                pi, value = jax.vmap(network.apply, in_axes=(0,0))(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                action = action.flatten()
                log_prob = log_prob.flatten()
                value = value.flatten()
                action_probs = pi.probs.reshape((config["NUM_PARTICLES"]*config['NUM_COPOLICIES']*config["NUM_ENVS"], -1))
                obs_batch = obs_batch.reshape((config["NUM_PARTICLES"]*config['NUM_COPOLICIES']*config["NUM_ENVS"], -1))
                
                env_act = unbatchifyp(action, env.agents, (config["NUM_PARTICLES"], config['NUM_COPOLICIES'], config["NUM_ENVS"]), env.num_agents)
                
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_PARTICLES"]*config['NUM_COPOLICIES']*config["NUM_ENVS"])
                
                # def step_copolicy_k(k, *args):
                #     return envs[k].step(*args)
                # copolicy_idxs = jnp.expand_dims(jnp.arange(0,config['NUM_COPOLICIES']), axis=(0,2))
                # copolicy_idxs = jnp.broadcast_to(
                #     jnp.arange(0,config['NUM_COPOLICIES'])[jnp.newaxis, :, jnp.newaxis], \
                #     shape=(config["NUM_PARTICLES"], config['NUM_COPOLICIES'], config["NUM_ENVS"])
                # )
                # obsv, env_state, reward, done, info = jax.vmap(step_copolicy_k, in_axes=(0,0,0,0))(
                #     copolicy_idxs, rng_step, env_state, env_act
                # )
                # @partial(jax.jit, static_argnums=0)
                # def step_copolicy_k(k, *args):
                #     return envs[k].step(*args)
                co_params_reshaped = jax.tree_map(lambda x : jnp.expand_dims(x, axis=(0,2)), co_params)
                co_params_reshaped = jax.tree_map(lambda x : jnp.broadcast_to(
                        x[jnp.newaxis, :, jnp.newaxis], \
                        shape=(config["NUM_PARTICLES"], config['NUM_COPOLICIES'], config["NUM_ENVS"], *x.shape[1:])
                    ),
                    co_params
                )
                # co_params_reshaped = jnp.empty(((config["NUM_PARTICLES"], config['NUM_COPOLICIES'], config["NUM_ENVS"])))
                # co_params_reshaped[:,0,:] = co_params
                co_params_reshaped = jax.tree_map(lambda x : jnp.reshape(x, (-1, *x.shape[3:])), co_params_reshaped)
                obsv, env_state, reward, done, info = jax.vmap(env.step_copolicy, in_axes=(0,0,0,0))(
                    rng_step, env_state, env_act, co_params_reshaped
                )
                # info['action_probs'] = action_probs
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    action_probs
                    
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )


            traj_returns = traj_batch.info["returned_episode_returns"].reshape((config["NUM_STEPS"], config["NUM_PARTICLES"], config['NUM_COPOLICIES'], config["NUM_ENVS"]))
            avg_returns = traj_returns.mean(axis=-1).sum(axis=0) # TODO add in discount to average reward
            # std_returns = traj_returns.sum(axis=0).std(axis=-1)
            # std_mean_returns = std_returns / jnp.sqrt(config["NUM_ENVS"])
            # jax.debug.print("mean\n{x}", x=avg_returns)
            # jax.debug.print("std {x}", x=std_mean_returns)
            grad_mask = jnp.zeros((config["NUM_STEPS"], config["NUM_PARTICLES"], config['NUM_COPOLICIES'], config["NUM_ENVS"]))

            if config['MATCHING'] == "single":
                for k in range(config["NUM_COPOLICIES"]):
                    max_idx = jnp.argmax(avg_returns[:,k])
                    grad_mask = grad_mask.at[:,max_idx,k,:].set(1)
                    avg_returns = avg_returns.at[max_idx,:].set(jnp.ones(config["NUM_COPOLICIES"]) * -jnp.inf)
            elif config['MATCHING'] == "multi_averaged":
                max_idxs = jnp.argmax(avg_returns, axis=0)
                col_idxs = jnp.arange(config["NUM_COPOLICIES"])
                grad_mask = grad_mask.at[:,max_idxs,col_idxs,:].set(1)

                grad_mask_sum = jnp.sum(grad_mask, axis=2, keepdims=True)
                # avoid divide by zeros
                grad_mask = jnp.where(grad_mask_sum==0, 0, jnp.divide(grad_mask, grad_mask_sum))
            elif config['MATCHING'] == "linear_rescale":
                max_returns_with_copolicy = jnp.max(avg_returns, axis=0, keepdims=True)
                # for n in range(config["NUM_PARTICLES"]):
                relative_return = avg_returns - max_returns_with_copolicy
                max_relative_return = jnp.max(relative_return, axis=1, keepdims=True)
                avg_returns = avg_returns - max_relative_return + jnp.finfo(avg_returns.dtype).eps
            
                max_idxs = jnp.argmax(avg_returns, axis=0)
                col_idxs = jnp.arange(config["NUM_COPOLICIES"])
                grad_mask = grad_mask.at[:,max_idxs,col_idxs,:].set(1)

                grad_mask_sum = jnp.sum(grad_mask, axis=2, keepdims=True)
                # avoid divide by zeros
                grad_mask = jnp.where(grad_mask_sum==0, 0, jnp.divide(grad_mask, grad_mask_sum))
            elif config['MATCHING'] == "averaged_otherwise_every":
                max_idxs = jnp.argmax(avg_returns, axis=0)
                col_idxs = jnp.arange(config["NUM_COPOLICIES"])
                grad_mask = grad_mask.at[:,max_idxs,col_idxs,:].set(1)

                every_max_mask = jnp.full((config["NUM_PARTICLES"], config["NUM_COPOLICIES"]), 1.0/config["NUM_COPOLICIES"])
                every_max_mask = jnp.expand_dims(every_max_mask, axis=(0,3))
                
                grad_mask_sum = jnp.sum(grad_mask, axis=2, keepdims=True)
                grad_mask = jnp.where(grad_mask_sum==0, every_max_mask, jnp.divide(grad_mask, grad_mask_sum))

            elif config['MATCHING'] == "normalized_rescale":
                avg_returns = avg_returns - jnp.min(avg_returns, axis=1, keepdims=True)
                avg_returns = avg_returns / jnp.max(avg_returns, axis=1, keepdims=True)
                # avg_returns = avg_returns / jnp.sum(avg_returns, axis=1, keepdims=True)
                avg_returns = jnp.nan_to_num(avg_returns, nan=1.0/config["NUM_PARTICLES"])
                max_idxs = jnp.argmax(avg_returns, axis=0)
                col_idxs = jnp.arange(config["NUM_COPOLICIES"])
                grad_mask = grad_mask.at[:,max_idxs,col_idxs,:].set(1)

                grad_mask_sum = jnp.sum(grad_mask, axis=2, keepdims=True)
                # avoid divide by zeros
                grad_mask = jnp.where(grad_mask_sum==0, 0, jnp.divide(grad_mask, grad_mask_sum))
            else:
                raise ValueError('Invalid MER matching type.') 

            # jax.debug.print("grad_mask\n{x}", x=grad_mask[0,:,:,0])
            grad_mask = grad_mask.reshape((config["NUM_STEPS"], -1))


            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents])
            last_obs_batch = last_obs_batch.reshape((config["NUM_PARTICLES"], config['NUM_COPOLICIES']*config["NUM_ENVS"], -1))                
            _, last_val = jax.vmap(network.apply, in_axes=(0,0))(train_state.params, last_obs_batch)
            last_val = last_val.flatten()

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
                    traj_batch, advantages, targets, grad_mask = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, grad_mask=1):
                        # RERUN NETWORK
                        # traj_batch_split = jax.tree_util.tree_map(lambda x : x.reshape((config["NUM_PARTICLES"], config['NUM_COPOLICIES']*config["NUM_ENVS"], *x.shape[1:])), traj_batch)
                
                        # pi, value = jax.vmap(network.apply, in_axes=(0,0))(params, traj_batch_split.obs)
                        # log_prob = pi.log_prob(traj_batch_split.action)

                        # log_prob = log_prob.flatten()
                        # value = value.flatten()

                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * (jnp.maximum(value_losses, value_losses_clipped)*grad_mask).mean()
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
                        loss_actor = (loss_actor*grad_mask).mean()
                        entropy = (pi.entropy()*grad_mask).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        total_loss = total_loss

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, grad_mask
                    )
                    # TODO apply gradient mask here
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"] // config["NUM_PARTICLES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"] // config["NUM_PARTICLES"]
                ), "batch size must be equal to number of steps * number of actors / number of particles"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets, grad_mask)
                batch = jax.tree_util.tree_map(
                    lambda x : x.reshape((config["NUM_STEPS"], config["NUM_PARTICLES"], config['NUM_COPOLICIES']*config["NUM_ENVS"]) + x.shape[2:]), batch
                )
                batch = jax.tree_util.tree_map(
                    lambda x : jnp.moveaxis(x, 1, 0), batch
                )
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_PARTICLES"],batch_size) + x.shape[3:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config['NUM_PARTICLES'], config["NUM_MINIBATCHES"], -1) + tuple(x.shape[2:])
                    ),
                    shuffled_batch,
                )
                # train_state, total_loss = jax.lax.scan(
                #     _update_minbatch, train_state, minibatches
                # )
                train_state, total_loss = jax.vmap(
                    lambda ts, minibatch : jax.lax.scan(
                        _update_minbatch, ts, minibatch
                    ),
                    in_axes=(0,0)
                )(train_state, minibatches)

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree_util.tree_map(
                    lambda x : x.reshape((config["NUM_STEPS"], config["NUM_PARTICLES"], config['NUM_COPOLICIES']*config["NUM_ENVS"]) + x.shape[2:]), traj_batch.info
                )
            metric['grad_mask'] = grad_mask
            # metric['avg_returns'] = avg_returns
            # metric['actions'] = traj_batch.action
            metric['action_probs'] = traj_batch.action_probs
            rng = update_state[-1]
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_n, env_state_n, obsv_n, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


