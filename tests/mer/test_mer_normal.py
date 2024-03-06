""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from data import DATA_DIR
from data.testing import TESTING_DATA_DIR
from tests.mer.utils import run_fixed_coparam_setup, run_test_core


"""
Testing for normal form games. The particles play columns and the copolicies play rows.
"""

def test_normal_3_by_3_learned_policies():

    rng = jax.random.PRNGKey(30)

    config = OmegaConf.load(TESTING_DATA_DIR + '/test_mer_ff_normal.yaml')
    config = OmegaConf.to_container(config) 

    payoffs = jnp.array([
        [2,-1,-1],
        [-1,0,1],
        [-1,1,0]
    ])
    config["ENV_KWARGS"]["payoffs"] = payoffs
    config["COPARAMS_SOURCE"] = 'file'

    payoff_string = str(payoffs).replace('\n','')
    config['COPARAMS_FILE'] = DATA_DIR + f"{config['ENV_NAME']}_{payoff_string}_save_params50.pkl"

    ### Evaluate
    pis, _ = run_test_core(rng, config)
    
    # check convergence in each trial
    for prob in pis.probs:
        assert(prob[0][0] >= .99 and prob[1][1]+prob[1][2] >= .99
           or prob[1][0] >= .99 and prob[0][1]+prob[0][2] >= .99)


def test_normal_3_by_3_balanced():

    rng = jax.random.PRNGKey(30)

    payoffs = jnp.array([
        [2,-1,-1],
        [-1,0,1],
        [-1,1,0]
    ])
    probs = jnp.array([[.98,.01,.01],
                       [.98,.01,.01],
                       [.01,.98,.01],
                       [.01,.01,.98]])

    pis, _ = run_fixed_coparam_setup(rng, payoffs, probs)

    # for pi in pis.probs:
    #     print(pi)
    
    # check convergence in each trial
    for prob in pis.probs:
        assert((prob[0][0] >= .95 and prob[1][1]+prob[1][2] >= .95)
           or (prob[1][0] >= .95 and prob[0][1]+prob[0][2] >= .95))
        

def test_normal_4_by_4_balanced_2particles():

    rng = jax.random.PRNGKey(30)

    payoffs = jnp.array([
        [ 2, 0, 0, 0],
        [ 0, 2, 0, 0],
        [ 0, 0, 1, 1],
        [-1,-1,-1,-1]
    ])
    probs = jnp.array([[.97,.01,.01,.01],
                       [.01,.97,.01,.01],
                       [.01,.01,.97,.01],
                       [.01,.01,.01,.97,]])
    
    pis, _ = run_fixed_coparam_setup(rng, payoffs, probs)        
    
    # check convergence in each trial
    for prob in pis.probs:
        print(prob)
        assert((prob[0][0] >= .95 and prob[1][1] >= .95)
           or (prob[1][0] >= .95 and prob[0][1] >= .95))
        
def test_dominance_problem():
    """ This test is a negative result. We need scaling to solve the dominance in this problem."""

    rng = jax.random.PRNGKey(27)

    payoffs = jnp.array([
        [ 2, 0, 0,-1,-1,-1,-1,-1,-1],
        [ 0, 2, 0,-1,-1,-1,-1,-1,-1],
        [ 0, 0, 1,-1,-1,-1,-1,-1,-1]
    ])
    probs = jnp.array([[.98,.01,.01],
                       [.01,.98,.01],
                       [.01,.01,.98]])
    
    override_config = {"MATCHING" : "multi_averaged", "NUM_PARTICLES" : 3, "NUM_STEPS": 100, "TOTAL_TIMESTEPS": 5e5, "LR": 2.5e-3}

    pis, _ = run_fixed_coparam_setup(rng, payoffs, probs, override_config)        
    
    # check convergence in each trial
    success = True
    for prob in pis.probs:
        success = success and (any(prob[:,0] >= .9)
           and any(prob[:,1] >= .9)
           and any(prob[:,2] >= .9))
        
    assert(not success)
        
def test_dominance_problem_scaling():

    rng = jax.random.PRNGKey(27)

    payoffs = jnp.array([
        [ 2, 0, 0,-1,-1,-1,-1,-1,-1],
        [ 0, 2, 0,-1,-1,-1,-1,-1,-1],
        [ 0, 0, 1,-1,-1,-1,-1,-1,-1]
    ])
    probs = jnp.array([[.98,.01,.01],
                       [.01,.98,.01],
                       [.01,.01,.98]])
    
    override_config = {"MATCHING" : "averaged_otherwise_every", "NUM_PARTICLES" : 3, "NUM_STEPS": 100, "TOTAL_TIMESTEPS": 5e5, "LR": 2.5e-3}

    pis, _ = run_fixed_coparam_setup(rng, payoffs, probs, override_config)        
    
    # check convergence in each trial
    for prob in pis.probs:
        assert(any(prob[:,0] >= .9)
           and any(prob[:,1] >= .9)
           and any(prob[:,2] >= .9))
        
def counter_example1():

    rng = jax.random.PRNGKey(30)

    payoffs = jnp.array([
        [1,0],
        [0,2]
    ])
    probs = jnp.array([[.60,.40], # should respond with 1
                       [.70,.30]]) # should respond with 0

    pis, _ = run_fixed_coparam_setup(rng, payoffs, probs, num_trials=1)

    # for pi in pis.probs:
    #     print(pi)
    
    # check convergence in each trial
    for prob in pis.probs:
        print(prob)
        # assert((prob[0][0] >= .95 and prob[1][1] >= .95)
        #    or (prob[1][0] >= .95 and prob[0][1] >= .95))


def test_normal_3_by_3_unbalanced():

    rng = jax.random.PRNGKey(30)

    payoffs = jnp.array([
        [2,0,0],
        [0,0,3],
        [0,1,0]
    ])
    probs = jnp.array([[.98,.01,.01],
                       [.98,.01,.01],
                       [.01,.98,.01],
                       [.01,.01,.98]])

    pis = run_fixed_coparam_setup(rng, payoffs, probs)

    # for pi in pis.probs:
    #     print(pi)
    
    # check convergence in each trial
    for prob in pis.probs:
        assert((prob[0][0] >= .95 and prob[1][1]+prob[1][2] >= .95)
           or (prob[1][0] >= .95 and prob[0][1]+prob[0][2] >= .95))



def test_normal_2_by_2_simple():

    rng = jax.random.PRNGKey(30)

    payoffs = jnp.array([
        [0,-1],
        [-1,1]
    ])
    probs = jnp.array([[.98,.02],
                       [.98,.02],
                       [.60,.40],
                       [.70,.30]])
    # All the particles go to the first action (the second one is not worth it because you always risk getting -1 and the first one is better than the second one in expectation)

    pis = run_fixed_coparam_setup(rng, payoffs, probs)

    for pi in pis.probs:
        print(pi)
    
    # check convergence in each trial
    for prob in pis.probs:
        assert((prob[0][0] >= .95 and prob[1][1]+prob[1][2] >= .95)
           or (prob[1][0] >= .95 and prob[0][1]+prob[0][2] >= .95))
        

def test_normal_2_by_2_simple_unbalanced():

    rng = jax.random.PRNGKey(30)

    payoffs = jnp.array([
        [0,-1],
        [-1,2]
    ])
    probs = jnp.array([[.98,.02],
                       [.98,.02],
                       [.60,.40],
                       [.70,.30]])
    # The two particles cover the two possible actions.
    
    pis = run_fixed_coparam_setup(rng, payoffs, probs)

    for pi in pis.probs:
        print(pi)
    
    # check convergence in each trial
    for prob in pis.probs:
        assert((prob[0][0] >= .95 and prob[1][1]+prob[1][2] >= .95)
           or (prob[1][0] >= .95 and prob[0][1]+prob[0][2] >= .95))



if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True)
    test_normal_4_by_4_balanced_2particles()