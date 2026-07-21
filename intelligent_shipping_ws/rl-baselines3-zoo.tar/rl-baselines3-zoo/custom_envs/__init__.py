from gymnasium.envs.registration import register, registry


SHIP_ENV_ID = "ShipPathTracking3DOF-v0"


if SHIP_ENV_ID not in registry:
    register(
        id=SHIP_ENV_ID,
        entry_point="custom_envs.ship_3dof.env:ShipPathTracking3DOFEnv",
        max_episode_steps=3000,
    )
