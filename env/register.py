# %%
# Registering Envs
# ----------------
#
# In order for the custom environments to be detected by Gymnasium, they
# must be registered as follows. We will choose to put this code in
# ``gym-examples/gym_examples/__init__.py``.
#
# .. code:: python
#
from gymnasium.envs.registration import register

register(
    id='GSSP-v0',
    entry_point='env.gssp_env:GSSP',
    kwargs={
        'num_jobs': 5,
        'num_machines': 3,
        'operations_data': None,
    }
)
