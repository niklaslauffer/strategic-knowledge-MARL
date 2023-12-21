import jax
from jaxmarl import make
rng = jax.random.PRNGKey(0)

key, key_reset, key_act, key_step = jax.random.split(rng, 4)
# Initialise environment.
env = make("overcooked")
# Reset the environment.
obs, state = env.reset(key_reset)
# Sample random actions.
key_act = jax.random.split(key_act, env.num_agents)
actions = {
    agent: env.action_space(agent).sample(key_act[i])
    for i, agent in enumerate(env.agents)
}

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)

a=4