from .environments import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    MiniSMAC,
    HeuristicEnemyMiniSMAC,
    LearnedPolicyEnemyMiniSMAC,
    SwitchRiddle,
    Ant,
    Humanoid,
    Hopper,
    Walker2d,
    HalfCheetah,
    InTheGrid,
    HanabiGame,
)


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered SMAX environments.")

    # 1. MPE PettingZoo Environments
    if env_id == "MPE_simple_v3":
        env = SimpleMPE(**env_kwargs)
    elif env_id == "MPE_simple_tag_v3":
        env = SimpleTagMPE(**env_kwargs)
    elif env_id == "MPE_simple_world_comm_v3":
        env = SimpleWorldCommMPE(**env_kwargs)
    elif env_id == "MPE_simple_spread_v3":
        env = SimpleSpreadMPE(**env_kwargs)
    elif env_id == "MPE_simple_crypto_v3":
        env = SimpleCryptoMPE(**env_kwargs)
    elif env_id == "MPE_simple_speaker_listener_v4":
        env = SimpleSpeakerListenerMPE(**env_kwargs)
    elif env_id == "MPE_simple_push_v3":
        env = SimplePushMPE(**env_kwargs)
    elif env_id == "MPE_simple_adversary_v3":
        env = SimpleAdversaryMPE(**env_kwargs)
    elif env_id == "MPE_simple_reference_v3":
        env = SimpleReferenceMPE(**env_kwargs)

    # 2. Switch Riddle
    elif env_id == "switch_riddle":
        env = SwitchRiddle(**env_kwargs)

    # 3. MiniSMAC
    elif env_id == "MiniSMAC":
        env = MiniSMAC(**env_kwargs)
    elif env_id == "HeuristicEnemyMiniSMAC":
        env = HeuristicEnemyMiniSMAC(**env_kwargs)
    elif env_id == "LearnedPolicyEnemyMiniSMAC":
        env = LearnedPolicyEnemyMiniSMAC(**env_kwargs)

    # 4. Mujoco
    elif env_id == "ant_4x2":
        env = Ant(**env_kwargs)
    elif env_id == "halfcheetah_6x1":
        env = HalfCheetah(**env_kwargs)
    elif env_id == "hopper_3x1":
        env = Hopper(**env_kwargs)
    elif env_id == "humanoid_9|8":
        env = Humanoid(**env_kwargs)
    elif env_id == "walker2d_2x3":
        env = Walker2d(**env_kwargs)

    # 5. InTheGrid
    elif env_id == "mg_in_the_grid":
        env = InTheGrid(**env_kwargs)
    
    # 6. Hanabi
    elif env_id == "hanabi":
        env = HanabiGame(**env_kwargs)

    return env


registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "switch_riddle",
    "MiniSMAC",
    "HeuristicEnemyMiniSMAC",
    "LearnedPolicyEnemyMiniSMAC",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    "mg_in_the_grid",
    "hanabi",
]
