import ast

import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
#from zhipuai import ZhipuAI
import re
#import dashscope
from openai import OpenAI
import math
import random
# from components.action_selectors import EpsilonGreedyActionSelector
from envs.starcraft.smac_maps import get_map_params
from ast import literal_eval
from math import sqrt
from ollama import chat


class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()

        map_name = args.env_args["map_name"]
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.agents_attr = get_map_params("agents_attr")

        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]

        self.agents = map_params["agents"]
        self.enemies = map_params["enemies"]
        self.all_agents_type = map_params["all_agents_type"]

        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]

        self.unit_type_bits = map_params["unit_type_bits"]

        # self.action_selector = EpsilonGreedyActionSelector(args)

        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, ava_actions, coord, llm, test_mode=False):
        b, a, e = inputs.size()

        b1, a1, e1 = ava_actions.size()

        inputs = inputs.view(-1, e)
        ava_actions = ava_actions.view(-1, e1)

        # x = F.relu(self.fc1(inputs), inplace=True)
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # hh = self.rnn(x, h_in)

        # if getattr(self.args, "use_layer_norm", False):
        #     q = self.fc2(self.layer_norm(hh))
        # else:
        #     q = self.fc2(hh)

        coord = coord.reshape(-1, 8)
        q_LLM = self.inputs2action(inputs, ava_actions, coord).to(self.args.device)

        print("\n ## out_put_Q = ", q_LLM)

        return q_LLM.view(b, a, -1)
        # return q.view(b, a, -1), hh.view(b, a, -1), q_LLM.view(b, a, -1)

    def code2prompt(self, input, ava_action, coord_agent):
        n_agents = self.n_agents
        n_enemies = self.n_enemies
        all_agents_type = self.all_agents_type
        e_enemy_dim = int(coord_agent[5])
        e_ally_dim = int(coord_agent[6])
        unit_properties = self.agents_attr

        # Check if the agent is alive
        if input[4 + e_enemy_dim * n_enemies + e_ally_dim * (n_agents - 1)] == 0:
            return "dead", None, None
        else:
            state = "alive"

        # Environment and agent description
        part_1 = (
            f"### Environment Description\n"
            f"- **Map Name**: {self.map_name}\n"
            f"- **Size**: {coord_agent[2]}x{coord_agent[3]}\n"
            f"- **Allied Team**: {self.format_dict(self.agents)}\n"
            f"- **Enemy Team**: {self.format_dict(self.enemies)}\n"
            f"- **Unit Descriptions**: {self.generate_unit_descriptions(all_agents_type)}\n"
        )

        # Movement and observation rules
        part_2 = (
            "### Rules\n"
            "- Agents can observe other agents within a 9-unit distance from their position.\n"
            "- Agents can only attack enemy agents that are alive and within a 6-unit distance.\n"
            "- The agent's attack damage and effectiveness are not affected by the distance to the target or the agent's current health.\n"
            "- Agents can move within the confines of the map in the following directions: "
            f"{', '.join(['north', 'south', 'east', 'west'][i] for i in th.nonzero(input[:4] == 1).squeeze(-1))}.\n"
        )

        # My unit
        # Self unit type and health
        unit_type = self.get_unit_type(input, e_enemy_dim, e_ally_dim)
        my_health_max = unit_properties[unit_type]["maximal health"]
        my_health = my_health_max * input[4 + e_enemy_dim * n_enemies + (n_agents - 1) * e_ally_dim]
        my_description = f"### My Unit\n- **Type**: {unit_type[:-1].replace('colossu', 'colossi')}\n- **Health**: {my_health:.1f}/{my_health_max}"

        if self._agent_race == "P":
            my_shield_max = unit_properties[unit_type]["maximal shield"]
            my_shield = my_shield_max * input[4 + e_enemy_dim * n_enemies + (n_agents - 1) * e_ally_dim + 1]
            my_description += f"\n- **Shield**: {my_shield:.1f}/{my_shield_max}\n"

        my_description += (
            f"- **Objective**: Coordinate with allied agents to eliminate enemy agents based on current local observations.\n"
            f"- **Current Position**: ({coord_agent[0]}, {coord_agent[1]})\n"
            f"- **Relative Coordinate System**: Origin at my current position, axes pointing upwards and to the right.\n"
        )

        # Ally status
        ally_info = self.generate_ally_info(input, e_ally_dim, e_enemy_dim, n_agents)

        # Enemy status
        enemy_info = self.generate_enemy_info(input, e_enemy_dim, n_enemies)

        # Construct system prompt
        system_prompt = "\n".join([part_1, part_2, my_description, ally_info, enemy_info])

        # Action recommendation request
        action_example = [0] * self.args.n_actions
        action_example[4] = 1  # Example for moving east
        user_prompt = (
            "\n### Action Selection\n"
            "Based on the current situation of my unit, please provide the most optimal action recommendation. It should be represented as a binary vector of length {}, where only one element is set to 1 indicating the selected action, and all others are 0. Select one action from the following options:\n\n").format(6 + n_enemies)

        base_action = [
            "No-operation (if dead)",
            "Stay stationary",
            "Move 2 units north",
            "Move 2 units south",
            "Move 2 units east",
            "Move 2 units west"
        ]
        no_attack_action = base_action.copy()
        for k in range(len(base_action)):
            if ava_action[k] == 1:
                continue
            elif ava_action[k] == 0:
                no_attack_action[k] = "Not Available (N/A)"

        if unit_type == "medivacs":
            no_move_action = []
            n_allies = n_agents - 1
            for i in range(n_allies):
                if ava_action[6+i] == 1:
                    no_move_action.append(f"Heal the {self.ordinal(i + 1)} ally. ")
                else:
                    no_move_action.append("Not Available (N/A)")
            if n_allies < n_enemies:
                for _ in range(n_enemies - n_allies):
                    no_move_action.append("Not Available (N/A)")

        else:
            no_move_action = []
            for j in range(n_enemies):
                if ava_action[6+j] == 1:
                    no_move_action.append(f"Attack the {self.ordinal(j + 1)} enemy. ")
                else:
                    no_move_action.append("Not Available (N/A)")

        final_ava_action = no_attack_action + no_move_action

        for idx in range(self.args.n_actions):
            user_prompt += f"  - **{idx+1}. {final_ava_action[idx]}**\n"

        move_east = [0] * self.args.n_actions
        move_east[4] = 1

        attack_1st_enemy = [0] * self.args.n_actions
        attack_1st_enemy[6] = 1

        user_prompt += (
            "\nNote that the number starts from 1, representing the first position (not the zero-based index). Choose the action by setting the corresponding index to 1, while all other indices remain 0.\n"
            "- **Example 01**: To move 2 units east would be represented as {}.\n"
            "- **Example 02**: To attack the 1st enemy would be represented as {}.\n"
            "Please provide your recommendation in this format, with the binary vector enclosed in square brackets and the elements separated by commas."
        ).format(
            move_east,
            attack_1st_enemy
        )

        return state, system_prompt, user_prompt

    # Helper methods to generate ally and enemy info
    def generate_ally_info(self, input, e_ally_dim, e_enemy_dim, n_agents):
        ally_list = ["### Ally Status\n"]
        ally_dim_start = self.n_enemies * e_enemy_dim + 4
        visible_allies = th.sum(input[ally_dim_start: ally_dim_start + e_ally_dim * (n_agents - 1)]) > 0

        if not visible_allies:
            ally_list.append("- All allies are out of sight.\n")
        else:
            for i in range(n_agents - 1):
                ally_data = input[e_ally_dim * i + ally_dim_start: e_ally_dim * (i + 1) + ally_dim_start]
                if th.sum(ally_data) == 0:
                    ally_i = f"- The {self.ordinal(i + 1)} ally is out of vision range or dead.\n"
                else:
                    unit_type = self.get_ally_unit_type(input, e_ally_dim, i, ally_dim_start)
                    distance_a = 9 * ally_data[1]
                    relative_position_a = ((9 * ally_data[2]).cpu().item(), (9 * ally_data[3]).cpu().item())
                    health_a = self.agents_attr[unit_type]["maximal health"] * ally_data[4]

                    ally_i = (
                        f"- The {self.ordinal(i + 1)} ally:\n"
                        f"  - **Type**: {unit_type[:-1].replace('colossu', 'colossi')}\n"
                        f"  - **Distance**: {distance_a:.1f} units\n"
                        f"  - **Position**: Relative coordinates {relative_position_a}\n"
                        f"  - **Health**: {health_a:.1f}/{self.agents_attr[unit_type]['maximal health']}\n"
                    )
                    if self._agent_race == "P":
                        shield_a = self.agents_attr[unit_type]["maximal shield"] * ally_data[5]
                        ally_i += f"  - **Shield**: {shield_a:.1f}/{self.agents_attr[unit_type]['maximal shield']}\n"
                ally_list.append(ally_i)
        return ''.join(ally_list)

    def generate_enemy_info(self, input, e_enemy_dim, n_enemies):
        enemy_list = ["### Enemy Status\n"]
        visible_enemies = th.sum(input[4: e_enemy_dim * n_enemies + 4]) > 0

        if not visible_enemies:
            enemy_list.append("- All enemies are out of sight.\n")
        else:
            for i in range(n_enemies):
                enemy_data = input[e_enemy_dim * i + 4: e_enemy_dim * (i + 1) + 4]
                if th.sum(enemy_data) == 0:
                    enemy_i = f"- The {self.ordinal(i + 1)} enemy is out of vision range or dead.\n"
                else:
                    unit_type = self.get_enemy_unit_type(input, e_enemy_dim, i)
                    in_attack_range = "within" if enemy_data[0].to(th.int) else "not within"
                    distance = 9 * enemy_data[1]
                    relative_position = ((9 * enemy_data[2]).cpu().item(), (9 * enemy_data[3]).cpu().item())
                    health = self.agents_attr[unit_type]["maximal health"] * enemy_data[4]

                    enemy_i = (
                        f"- The {self.ordinal(i + 1)} enemy:\n"
                        f"  - **Type**: {unit_type[:-1].replace('colossu', 'colossi')}\n"
                        f"  - **Attack Range**: {in_attack_range}\n"
                        f"  - **Distance**: {distance:.1f} units\n"
                        f"  - **Position**: Relative coordinates {relative_position}\n"
                        f"  - **Health**: {health:.1f}/{self.agents_attr[unit_type]['maximal health']}\n"
                    )
                    if self._bot_race == "P":
                        shield = self.agents_attr[unit_type]["maximal shield"] * enemy_data[5]
                        enemy_i += f"  - **Shield**: {shield:.1f}/{self.agents_attr[unit_type]['maximal shield']}\n"
                enemy_list.append(enemy_i)
        return ''.join(enemy_list)

    # Helper method to get the unit type
    def get_unit_type(self, input, e_enemy_dim, e_ally_dim):
        if self.unit_type_bits == 0:
            return list(self.agents.keys())[0]
        else:
            type_index_start = (4 + self.n_enemies * e_enemy_dim + (self.n_agents - 1) * e_ally_dim + 1) if self._agent_race != "P" else (
                        4 + self.n_enemies * e_enemy_dim + (self.n_agents - 1) * e_ally_dim + 2)
            type_list = input[type_index_start: type_index_start + self.unit_type_bits]
            type_index = th.nonzero(type_list).squeeze(-1).to(th.int)
            return list(self.agents.keys())[type_index]

    # Helper method to get the ally unit type
    def get_ally_unit_type(self, input, e_ally_dim, i, ally_dim_start):
        if self.unit_type_bits == 0:
            return list(self.agents.keys())[0]
        else:
            type_index_start = e_ally_dim * i + ally_dim_start + (5 if self._agent_race != "P" else 6)
            type_list = input[type_index_start: type_index_start + self.unit_type_bits]
            type_index = th.nonzero(type_list).squeeze(-1).to(th.int)
            return list(self.agents.keys())[type_index]

    # Helper method to get the enemy unit type
    def get_enemy_unit_type(self, input, e_enemy_dim, i):
        if self.unit_type_bits == 0:
            return list(self.enemies.keys())[0]
        else:
            type_index_start = 9 + e_enemy_dim * i if self._bot_race != "P" else 10 + e_enemy_dim * i
            type_list = input[type_index_start: type_index_start + self.unit_type_bits]
            type_index = th.nonzero(type_list).squeeze(-1).to(th.int)
            return list(self.enemies.keys())[type_index]

    # Helper method to get ordinal suffix
    def ordinal(self, num):
        if 10 <= num % 100 <= 20:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(num % 10, 4)]
        return f"{num}{suffix}"

    def action2code(self, input_tensor, ava_action, coord_agent):

        state_plus_prompt = self.code2prompt(input_tensor, ava_action, coord_agent)

        state = state_plus_prompt[0]

        if state == "dead":
            action_tensor_ = th.zeros(self.args.n_actions)
            action_tensor_[0] = 1
            action_tensor = action_tensor_.to(self.args.device)
        else:

            system_prompt = state_plus_prompt[1]
            user_prompt = state_plus_prompt[2]

            print("\n ## prompt = ", system_prompt + user_prompt)

            
            try:
                #response = chat(
                #    model="qwen2.5:32b",
                #    messages=[{'role': 'system', 'content': system_prompt},
                #              {'role': 'user', 'content': user_prompt}]
                #)
                #answer = response["message"]["content"]

                # client = OpenAI(api_key="sk-41gBHOOMJVZflE9o83841661D400443eA0E88b108aBbDbA1",
                #                 base_url="https://aihubmix.com/v1")
                # use_model = "gpt-4"
                #
                # completion = client.chat.completions.create(
                #     model=use_model,
                #     messages=[{'role': 'system', 'content': system_prompt},
                #               {'role': 'user', 'content': user_prompt}]
                # )

                client = OpenAI(api_key="sk-a45a4e7df8e54d0aa7b6b864c99a31a7",
                                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                use_model = "qwen2.5-72b-instruct"
                completion = client.chat.completions.create(
                    model=use_model,
                    messages=[{'role': 'system', 'content': system_prompt},
                              {'role': 'user', 'content': user_prompt}]
                )
                answer = completion.choices[0].message.content

                print(answer)



                # print("\n ## 3.answer = ", answer)

                pattern = r"\[[^][]*\b(?:\d\s*,?\s*)\b[^][]*]"
                action_str = re.findall(pattern, answer)[-1]
                action_list = list(map(int, re.sub(r'\[|\]|\s+', '', action_str).split(',')))
                action_tensor = th.tensor(action_list).to(self.args.device)
                print("\n ## 4. action = ", action_tensor)

            except Exception as e:
                print(e)

                action_tensor = th.zeros(self.args.n_actions).to(self.args.device)

        # print(action_tensor)

        return action_tensor

    def inputs2action(self, inputs, ava_actions, coord):

        ba, e = inputs.size()

        q_LLM = th.zeros(ba, self.args.n_actions).to(self.args.device)

        # print(ba, epsilon, int(ba * epsilon * 0.1))

        # num_LLM = math.ceil(ba * epsilon * 0.4)
        # indices = th.randperm(ba)[:num_LLM]

        for ind in range(ba):
            try:
                q_LLM[ind] = self.action2code(inputs[ind], ava_actions[ind], coord[ind]).to(self.args.device)
            except Exception as e:
                print(e)
                continue

        return q_LLM

    def format_dict(self, d):
        # agent_dict --> agent_description

        if not d:
            return ""

        items = [f"{value} {key[:-1] if value == 1 else key}" for key, value in d.items()]

        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"


    def generate_unit_descriptions(self, d):

        unit_properties = self.agents_attr
        descriptions = []

        # print(d)

        for key in d:

            props = unit_properties[key]

            description = f"Each {key[:-1]} has an initial health of {props['maximal health']} points "

            if key == 'medivacs':
                description += f"and an initial energy of {props['maximal energy']} points, restoring {props['heal_amount']} health points to the ally and consuming {props['energy_cost']} energy points per heal. Medivacs cannot heal themselves or other flying units. "

            elif key == 'stalkers' or key == 'zealots' or key == 'colossus':
                description += f"and an initial shield of {props['maximal shield']}, "
                description += f"dealing {props['damage']} damage per attack. "

                description = description.replace("colossu", "colossi")

            else:
                description += f"and deals {props['damage']} damage per attack. "

            descriptions.append(description)

        if 'stalkers' in d or 'zealots' in d or 'colossus' in d:
            descriptions.append(
                f"The shields of Protoss units (stalkers, zealots, and colossus) must be depleted before their health can be affected. ")

        # print(descriptions)

        descriptions = "".join(descriptions)

        return descriptions

    def max_shield(self, unit):
        if unit == "stalkers":
            shield_max = 80
            return shield_max
        if unit == "zealots":
            shield_max = 50
            return shield_max
        if unit == "colossus":
            shield_max = 150
            return shield_max
        else:
            return None


