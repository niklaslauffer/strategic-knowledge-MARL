import jax

from jaxmarl import make
from jaxmarl.environments.multi_agent_env import DelayedObsWrapper

rng = jax.random.PRNGKey(2)

key, key_reset, key_act, key_step = jax.random.split(rng, 4)
# Initialise environment.
env = make("overcooked")
# Reset the environment.
env = DelayedObsWrapper(env, delay=2)

import pdb; pdb.set_trace()
obs, state = env.reset(key_reset)
# Sample random actions.
key_act = jax.random.split(key_act, env.baseEnv.num_agents)
actions = {
    agent: env.action_space(agent).sample(key_act[i])
    for i, agent in enumerate(env.baseEnv.agents)
}

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
import pdb; pdb.set_trace()

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
import pdb; pdb.set_trace()

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
import pdb; pdb.set_trace()

a = 4