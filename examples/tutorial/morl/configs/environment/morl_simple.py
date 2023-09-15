# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for tutorial level: morl_simple."""

from ml_collections import config_dict
from meltingpot.python.utils.substrates import shapes #add python

AVATAR = { #prefab to create game object which contain components
    "name": "avatar",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "player",
                "stateConfigs": [
                    {"state": "player",
                     "layer": "upperPhysical",
                     "contact": "avatar",
                     "sprite": "Avatar",},
                ]
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Avatar"],
                "spriteShapes": [shapes.CUTE_AVATAR],
                "palettes": [{}],  # Will be overridden
                "noRotates": [True],
            }
        },
        {
            "component": "Avatar",
            "kwargs": {
                "aliveState": "player",
                "waitState": "playerWait",
                "spawnGroup": "spawnPoints",
                "view": {
                    "left": 3,
                    "right": 3,
                    "forward": 5,
                    "backward": 1,
                    "centered": False,
                }
            }
        },
    ]
}

SPAWN_POINT = {
    "name": "spawn_point",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "groups": ["spawnPoints"],
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

WALL = {
    "name": "wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall",
                "stateConfigs": [{
                    "state": "wall",
                    "layer": "upperPhysical",
                    "sprite": "Wall",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall",],
                "spriteShapes": [shapes.WALL ],
                "palettes": [shapes.WALL_PALETTE],
                "noRotates": [True],
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "gift"
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "zap"
            }
        },
    ]
}

APPLE = {
    "name": "apple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "apple",
                "stateConfigs": [{
                    "state": "apple",
                    "layer": "lowerPhysical",
                    "sprite": "Apple",
                }, {
                    "state": "appleWait",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Apple",],
                "spriteShapes": [shapes.LEGACY_APPLE],
                "palettes": [shapes.GREEN_COIN_PALETTE],
                "noRotates": [True],
            }
        },
        {
            "component": "Edible",
            "kwargs": {
                "liveState": "apple",
                "waitState": "appleWait",
                "rewardForEating": 1.0,
            }
        },
    ]
}

DIRT = {
    "name": "dirt",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "dirt",
                "stateConfigs": [{
                    "state": "dirt",
                    "layer": "lowerPhysical",
                    "sprite": "Dirt",
                }, {
                    "state": "dirtWait",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Dirt",],
                "spriteShapes": [shapes.DIRT_PATTERN],
                "palettes": [{
                    "x": (81, 70, 32, 255),
                    "X": (89, 77, 36, 255),
                }],
                "noRotates": [True],
            }
        },
        {
            "component": "DensityRegrow",
            "kwargs": {
                "liveState": "dirt",
                "liveStateOther": "apple",
                "waitState": "dirtWait",
                "baseRate": 0.1,
                "neighborhoodRadius": 5,
            }
        },
        {
            "component": "Edible",
            "kwargs": {
                "liveState": "dirt",
                "waitState": "dirtWait",
                "rewardForEating": 0.0,
            }
        },
    ]
}

ASCII_MAP = """
D _ A
D   A
D   A
D   A
D _ A
"""
#5x5 sprites
#8x8 pixels = 40x40
# ASCII_MAP = """
# *******
# *D _ A*
# *D   A*
# *D   A*
# *D   A*
# *D _ A*
# *******
# """
#7x7
def get_config():
  """Default configuration for the morl_simple level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  #config.num_players = 2 no need
  config.individual_observation_names = ["RGB"]
  config.global_observation_names = ["WORLD.RGB"]

  # Lua script configuration.
  config.lab2d_settings = {#no substrate
      "levelName": "morl_simple",
      "levelDirectory":
          "examples/tutorial/morl/levels",
      "maxEpisodeLengthFrames": 1000,
      "numPlayers": 2,
      "spriteSize": 8,
      "simulation": {
                    "map": ASCII_MAP,
            "prefabs": {
              "avatar": AVATAR,
              "spawn_point": SPAWN_POINT,
              "wall": WALL,
              "apple": APPLE,
              "dirt": DIRT,
          },
          "charPrefabMap": {"_": "spawn_point", "*": "wall", "A": "apple", "D": "dirt"},
          "playerPalettes": [],
      },
  }

  return config
