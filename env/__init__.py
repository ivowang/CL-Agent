from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
}
