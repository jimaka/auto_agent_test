import gymnasium as gym
import numpy as np

import custom_envs  # noqa: F401


def test_ship_env_registration():
    env = gym.make("ShipPathTracking3DOF-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (14,)
    assert "curriculum_stage" in info

    for _ in range(5):
        action = np.zeros(2, dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert obs.shape == (14,)
        assert np.isfinite(reward)
        assert "reward_terms" in step_info
        if terminated or truncated:
            break

    env.close()
