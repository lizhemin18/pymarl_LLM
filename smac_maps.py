from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "3m": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":3},
        "enemies":{"marines":3},
    },
    "8m": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":8},
        "enemies":{"marines":8},
    },
    "25m": {
        "n_agents": 25,
        "n_enemies": 25,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":25},
        "enemies":{"marines":25},
    },
    "5m_vs_6m": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":5},
        "enemies":{"marines":6},
    },
    "8m_vs_9m": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":8},
        "enemies":{"marines":9},
    },
    "10m_vs_11m": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":10},
        "enemies":{"marines":11},
    },
    "27m_vs_30m": {
        "n_agents": 27,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines"},
        "all_enemies_type":{"marines"},
        "agents":{"marines":27},
        "enemies":{"marines":30},
    },
    "MMM": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
        "all_agents_type":{"marauders", "marines", "medivacs"},
        "all_enemies_type":{"marauders", "marines", "medivacs"},
        "agents":{"marauders":2, "marines":7,"medivacs":1},
        "enemies":{"marauders":2, "marines":7,"medivacs":1},
    },
    "MMM2": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
        "all_agents_type":{"marauders", "marines", "medivacs"},
        "all_enemies_type":{"marauders", "marines", "medivacs"},
        "agents":{"marauders":2, "marines":7,"medivacs":1},
        "enemies":{"marauders":3, "marines":8,"medivacs":1},
    },
    "2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
        "all_agents_type":{"stalkers","zealots"},
        "all_enemies_type":{"zealots","stalkers"},
        "agents":{"stalkers":2,"zealots":3},
        "enemies":{"zealots":3,"stalkers":2},
    },
    "3s5z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
        "all_agents_type":{"stalkers","zealots"},
        "all_enemies_type":{"zealots","stalkers"},
        "agents":{"stalkers":3,"zealots":5},
        "enemies":{"zealots":5,"stalkers":3},
    },
    "3s5z_vs_3s6z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
        "all_agents_type":{"stalkers","zealots"},
        "all_enemies_type":{"zealots","stalkers"},
        "agents":{"stalkers":3,"zealots":5},
        "enemies":{"zealots":6,"stalkers":3},
    },
    "3s_vs_3z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "all_agents_type":{"stalkers", "zealots"},
        "all_enemies_type":{"zealots","stalkers"},
        "agents":{"stalkers":3},
        "enemies":{"zealots":3},
    },
    "3s_vs_4z": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "all_agents_type":{"stalkers", "zealots"},
        "all_enemies_type":{"zealots","stalkers"},
        "agents":{"stalkers":3},
        "enemies":{"zealots":4},
    },
    "3s_vs_5z": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "all_agents_type":{"stalkers", "zealots"},
        "all_enemies_type":{"zealots","stalkers"},
        "agents":{"stalkers":3},
        "enemies":{"zealots":5},
    },
    "1c3s5z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
        "all_agents_type":{"colossus", "stalkers", "zealots"},
        "all_enemies_type":{"colossus", "stalkers", "zealots"},
        "agents":{"colossus":1,"stalkers":3,"zealots":5},
        "enemies":{"colossus":1,"stalkers":3,"zealots":5},
    },
    "2m_vs_1z": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "marines",
        "all_agents_type":{"marines", "zeolots"},
        "all_enemies_type":{"zeolots"},
        "agents":{"marines":2},
        "enemies":{"zealots":1},
    },
    "corridor": {
        "n_agents": 6,
        "n_enemies": 24,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
        "all_agents_type":{"zealots", "zerglings"},
        "all_enemies_type":{"zerglings"},
        "agents":{"zealots":6},
        "enemies":{"zerglings":24},
    },
    "6h_vs_8z": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
        "all_agents_type":{"hydralisks", "zealots"},
        "all_enemies_type":{"zealots"},
        "agents":{"hydralisks":6},
        "enemies":{"zealots":8},
    },
    "2s_vs_1sc": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 300,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
        "all_agents_type":{"stalkers", "spine crawlers"},
        "all_enemies_type":{"spine crawlers"},
        "agents":{"stalkers":2},
        "enemies":{"spine crawlers":1},
    },
    "so_many_baneling": {
        "n_agents": 7,
        "n_enemies": 32,
        "limit": 100,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
        "all_agents_type":{"zealots", "banelings"},
        "all_enemies_type":{"banelings"},
        "agents":{"zealots":7},
        "enemies":{"banelings":32},
    },
    "bane_vs_bane": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 200,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "bane",
        "all_agents_type":{"banelings", "zerglings"},
        "all_enemies_type":{"banelings", "zerglings"},
        "agents":{"banelings":4, "zerglings":20},
        "enemies":{"banelings":4, "zerglings":20},
    },
    "2c_vs_64zg": {
        "n_agents": 2,
        "n_enemies": 64,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
        "all_agents_type":{"colossus", "zerglings"},
        "all_enemies_type":{"zerglings"},
        "agents":{"colossus":2},
        "enemies":{"zerglings":64},
    },
    "agents_attr":{
        "marines": {"damage":6, "maximal health":45, "maximal cooldown":15},
        "marauders": {"damage":10, "maximal health":125, "maximal cooldown":25},
        "medivacs": {"heal_amount":25, "maximal health":150, "maximal energy":200, "energy_cost":10},
        "stalkers": {"damage":13, "maximal health":80, "maximal cooldown":35, "maximal shield": 80},
        "zealots": {"damage":8, "maximal health":100, "maximal cooldown":22, "maximal shield": 50},
        "colossus": {"damage":10, "maximal health":250, "maximal cooldown":24, "maximal shield": 150},
        "hydralisks": {"damage":12, "maximal health":80, "maximal cooldown":10},
        "zerglings": {"damage":5, "maximal health":35, "maximal cooldown":11},
        "banelings": {"damage":16, "maximal health":30, "maximal cooldown":1},
    },
}


def get_smac_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap,), dict(filename=name))
