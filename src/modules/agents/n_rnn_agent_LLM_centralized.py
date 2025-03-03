import ast
import os
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
from llm_integration import LLMIntegration

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()

        map_name = args.env_args["map_name"]
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.agents_attr = get_map_params("agents_attr")

        self.map_type = map_params["map_type"]

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

        self.llm_client = LLMIntegration()
        
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, state, hidden_state, coord, llm, epsilon, test_mode=False):

        b, a, e = inputs.size()
        m, n = state.size()
        
        obs = inputs.clone()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        coord = coord.reshape(-1, 10)

        if llm == True and test_mode == False and epsilon > 0.98:

            q_LLM = self.inputs2action(obs, state, coord, epsilon).to(self.args.device)

            return q.view(b, a, -1), hh.view(b, a, -1), q_LLM.view(b, a, -1)

        else:

            q_LLM = th.zeros(b, a, self.args.n_actions).to(self.args.device)

            return q.view(b, a, -1), hh.view(b, a, -1), q_LLM.view(b, a, -1)

    def code2prompt(self, obs, input, coord_agent):
        n_agents = self.n_agents
        n_enemies = self.n_enemies
        max_dis_x = coord_agent[6]
        max_dis_y = coord_agent[7]
        map_x = coord_agent[2]
        map_y = coord_agent[3]
        center_x = map_x / 2
        center_y = map_y / 2
        e_ally_dim = coord_agent[4].to(th.int)
        e_enemy_dim = coord_agent[5].to(th.int)
        obs_e_ally_dim = coord_agent[8].to(th.int)
        obs_e_enemy_dim = coord_agent[9].to(th.int)
        unit_properties = self.agents_attr
        map_type = self.map_type
        attack_range = 6

        if th.sum(input[: n_agents * e_ally_dim]) == 0:
            state = "all_dead"
            system_prompt = None
            user_prompt = None

            print("\n\n-----------stage 0--------------\n\n")

            return state, system_prompt, user_prompt
        else:
            state = "alive"

            # Map and Objective
            part_1 = f"### Map and Objective\n"
            part_1 += f"- **Map**: '{self.map_name}' - A square map with {map_x} units by {map_y} units.\n"
            part_1 += f"- **Teams**: Allies (West) vs Enemies (East).\n"
            part_1 += f"- **Allies Composition**: {self.format_dict(self.agents)}.\n"
            part_1 += f"- **Enemies Composition**: {self.format_dict(self.enemies)}.\n"
            part_1 += f"- **Objective**: Coordinate to eliminate enemy agents efficiently, ensuring that allies do not waste attacks on enemies that can be defeated by fewer attacks. Prioritize targets based on their threat level or proximity.\n"

            # Unit Descriptions
            part_2 = "\n### Unit Descriptions\n" + self.generate_unit_descriptions(self.all_agents_type)

            # Game Mechanics
            part_3 = "\n\n### Game Mechanics\n"
            part_3 += "- **Observation Range**: 9 units from the agent's position.\n"
            part_3 += f"- **Attack Range**: {attack_range} units for targeting live enemies.\n"
            part_3 += f"- **Movement**: Agents can move exactly 2 units in any of four cardinal directions per time step.\n"
            part_3 += "- **Coordinate System**: Origin at the map center (0,0), with axes pointing upwards and rightwards.\n"
            part_3 += "- **Shields**: Protoss units have shields that must be depleted before health damage occurs.\n"

            # Additional Instructions for Coordination

            part_4 = "\n### Coordination Guidelines\n"
            part_4 += "- **Target Allocation**: Ensure that no more than the necessary number of allies target a single enemy. If an enemy can be defeated by one ally, only assign one ally to attack it unless there are tactical reasons to do otherwise.\n"
            part_4 += "- **Communication**: Agents should execute coordinated actions.\n"
            part_4 += "- **Efficiency**: Focus on maximizing the efficiency of attacks, minimizing wasted shots, and optimizing the movement of allies to ensure they can contribute effectively.\n"


            move_east = [0] * self.args.n_actions
            move_east[4] = 1

            attack_1st_enemy = [0] * self.args.n_actions
            attack_1st_enemy[6] = 1

            part_5 = "\n### Response Format\n"

            part_5 += (f"Each action should be represented as a binary vector of length {self.args.n_actions}, where only one element is set to 1 indicating the chosen action. "
            f"The response should be a list of {n_agents} such vectors, enclosed in square brackets and separated by commas. "
            f"\nFor example, if the optimal actions for the first two agents are to move 2 units east and attack the 1st enemy, respectively, the returned vector would be: "
            f"{move_east}, {attack_1st_enemy}.\n")

            if self.map_type == "MMM":

                heal_1st_ally = [0] * self.args.n_actions
                heal_1st_ally[6] = 1

                heal_4th_ally = [0] * self.args.n_actions
                heal_4th_ally[9] = 1

                part_5 += f"For a medivac, the action vector positions correspond to different actions. In this case:\n" \
                               f"(1) The vector {heal_1st_ally} sets the 7th position (representing the first position for healing actions) to 1, which means it will heal the first ally. \n" \
                               f"(2) The vector {heal_4th_ally} sets the 10th position to 1, indicating it will heal the fourth ally. \n"

            print("\n\n-----------stage 1--------------\n\n")

            # Ally Information
            ally_list = ["\n### Current Game State\n\n### Ally Status"]
            medivac_indices = []  # 记录 medivac 的索引
            agents_type_list = []
            for i in range(n_agents):
                suffix = ['st', 'nd', 'rd'][i % 10] if 0 <= i <= 2 or (i >= 20 and 0 <= i % 10 <= 2) else 'th'

                a_start_index_i = e_ally_dim * i
                a_end_index_i = a_start_index_i + e_ally_dim
                ally_input_i = input[a_start_index_i:a_end_index_i]
                if th.sum(ally_input_i) == 0:
                    ally_i = f"- **{i+1}{suffix} Ally**: Dead"
                    ally_list.append(ally_i)
                    agents_type_list.append(None)
                    continue

                if self.unit_type_bits == 0:
                    unit_type = list(self.agents.keys())[0]
                    shield_a = self.max_shield(unit_type) * input[
                        e_ally_dim * i + 4] if self._agent_race == "P" else None
                    agents_type_list.append(unit_type)
                else:
                    type_index_start = e_ally_dim * i + (5 if self._agent_race == "P" else 4)
                    type_list = input[type_index_start: type_index_start + self.unit_type_bits]
                    # print(f"type_list={type_list}")
                    type_index = th.nonzero(type_list).squeeze(-1).cpu().item()
                    agents_list = list(self.agents.keys())
                    unit_type = agents_list[type_index]
                    shield_a = self.max_shield(unit_type) * input[
                        e_ally_dim * i + 4] if self._agent_race == "P" else None
                    agents_type_list.append(unit_type)

                unit_type_a = unit_type[:-1].replace('colossu', 'colossi')
                relative_position_a = (round((max_dis_x * input[e_ally_dim * i + 2]).cpu().item(),2),
                                       round((max_dis_y * input[e_ally_dim * i + 3]).cpu().item(),2))
                my_attrs = unit_properties[unit_type]
                a_health_max = my_attrs["maximal health"]
                health_a = a_health_max * input[e_ally_dim * i]
                energy_a = input[e_ally_dim * i + 1]

                ally_i = f"- **{i + 1}{suffix} Ally**: {unit_type_a}, Position: {relative_position_a}, Health: {health_a:.2f}"
                if shield_a is not None:
                    ally_i += f", Shield: {shield_a:.2f}"
                if unit_type_a == "medivac":
                    ally_i += f", Energy: {energy_a:.2f}"
                    medivac_indices.append(i)
                else:
                    ally_i += f", Cooldown: {energy_a:.2f}"
                ally_list.append(ally_i)
                # ally_positions.append(relative_position_a)
            ally_info = '\n'.join(ally_list)

            print("\n\n-----------stage 2--------------\n\n")

            # Enemy Information
            enemy_list = ["\n\n### Enemy Status"]
            enemy_positions = [] 
            for i in range(n_enemies):
                suffix = ['st', 'nd', 'rd'][i % 10] if 0 <= i <= 2 or (i >= 20 and 0 <= i % 10 <= 2) else 'th'

                e_start_index_i = n_agents * e_ally_dim + e_enemy_dim * i
                e_end_index_i = e_start_index_i + e_enemy_dim
                enemy_input_i = input[e_start_index_i:e_end_index_i]
                if th.sum(enemy_input_i) == 0:
                    enemy_i = f"- **{i+1}{suffix} Enemy**: Dead"
                    enemy_list.append(enemy_i)
                    continue

                if self.unit_type_bits == 0:
                    unit_type = list(self.enemies.keys())[0]
                    shield = self.max_shield(unit_type) * input[
                        n_agents * e_ally_dim + e_enemy_dim * i + 3] if self._bot_race == "P" else None
                else:
                    type_index_start = n_agents * e_ally_dim + e_enemy_dim * i + (4 if self._bot_race == "P" else 3)
                    type_list = input[type_index_start: type_index_start + self.unit_type_bits]
                    type_index = th.nonzero(type_list).squeeze(-1).cpu().item()
                    enemies_list = list(self.enemies.keys())
                    unit_type = enemies_list[type_index]
                    shield = self.max_shield(unit_type) * input[
                        n_agents * e_ally_dim + e_enemy_dim * i + 3] if self._bot_race == "P" else None

                e_unit_type = unit_type[:-1].replace('colossu', 'colossi')
                relative_position = (round((max_dis_x * input[n_agents * e_ally_dim + e_enemy_dim * i + 1]).cpu().item(),2),
                                     round((max_dis_y * input[n_agents * e_ally_dim + e_enemy_dim * i + 2]).cpu().item(),2))
                my_attrs = unit_properties[unit_type]
                e_health_max = my_attrs["maximal health"]
                health = e_health_max * input[n_agents * e_ally_dim + e_enemy_dim * i]

                enemy_i = f"- **{i + 1}{suffix} Enemy**: {e_unit_type}, Position: {relative_position}, Health: {health:.2f}"
                if shield is not None:
                    enemy_i += f", Shield: {shield:.2f}"
                enemy_list.append(enemy_i)
                enemy_positions.append(relative_position)
            enemy_info = '\n'.join(enemy_list)

            print("\n\n-----------stage 3--------------\n\n")

            # Action Instructions
            action_instructions = "\n\n### Action Selection"
            action_instructions += f"\n- For each of the {n_agents} allied agents, select one action from the following options:\n"
            action_instructions += f"- Each number corresponds to the index in the action vector. The number starts from 1, representing the first position (not the zero-based index). Choose the action by setting the corresponding index to 1, while all other indices remain 0.\n"

            action_instructions += f"- 'Not Available (N/A)' indicates that action is not selectable.\n"

            base_actions = [
                "No-operation (if dead)",
                "Stay stationary",
                "Move 2 units north",
                "Move 2 units south",
                "Move 2 units east",
                "Move 2 units west"
            ]

            obs_ally_dim_start = self.n_enemies * obs_e_enemy_dim + 4

            if map_type == "MMM":
                if medivac_indices:
                    n_allies = n_agents - len(medivac_indices)  
                    medivac_actions = base_actions.copy()
                    for i in range(min(n_allies, n_enemies)):
                        distance_i = 9 * obs[medivac_indices[0]][obs_e_ally_dim * i + obs_ally_dim_start + 1]
                        if distance_i <= 9:
                            medivac_actions.append(f"Heal the {self.ordinal(i + 1)} ally. ")
                            # medivac_actions.append(
                            #     f"Heal {self.ordinal(i + 1)} ally, restoring {unit_properties['medivacs']['heal_amount']} health points to it. ")
                    if n_allies < n_enemies:
                        for _ in range(n_enemies - n_allies):
                            medivac_actions.append("Not Available (N/A)")
                else:
                    medivac_actions = []

                non_medivac_actions = base_actions.copy()
                max_action_length = len(base_actions)

                for i in range(n_agents):
                    suffix = ['st', 'nd', 'rd'][i % 10] if 0 <= i <= 2 or (i >= 20 and 0 <= i % 10 <= 2) else 'th'
                    agent_i_type = agents_type_list[i]

                    if th.sum(input[i * e_ally_dim: (i+1) * e_ally_dim]) == 0:
                        action_instructions += f"\n  - **For the {i + 1}{suffix} Ally**: Dead\n"
                        continue

                    available_actions = base_actions.copy()
                    for j in range(n_enemies):
                        NA = "Not Available (N/A)"

                        if agent_i_type and agent_i_type != "medivacs":
                            damage_i = unit_properties[agent_i_type]["damage"]
                            ATTACK = f"Attack the {self.ordinal(j + 1)} enemy. "
                            # ATTACK = f"Attack {self.ordinal(j + 1)} enemy, dealing {damage_i} damage points to it. "
                            attack_or_not = [NA, ATTACK][obs[i][obs_e_enemy_dim * j + 4].to(th.int)]
                        else:
                            attack_or_not = NA

                        available_actions.append(attack_or_not)
                        
                    if i in medivac_indices:
                        action_instructions += f"\n  - **For the {i + 1}{suffix} Medivac Ally:**\n"
                        for idx, action in enumerate(medivac_actions, start=1):
                            action_instructions += f"    - **{idx}. {action}**\n"
                    else:
                        action_instructions += f"\n  - **For the {i + 1}{suffix} Ally:**\n"
                        for idx, action in enumerate(available_actions, start=1):
                            action_instructions += f"    - **{idx}. {action}**\n"

                    max_action_length = max(max_action_length, len(available_actions))

                if len(medivac_actions) < max_action_length:
                    medivac_actions.extend(["Not Available (N/A)"] * (max_action_length - len(medivac_actions)))
                for i in range(n_agents):
                    if i in medivac_indices:
                        available_actions = medivac_actions
                    else:
                        available_actions = non_medivac_actions.copy()
                        agent_i_type = agents_type_list[i]
                        for j in range(n_enemies):

                            NA = "Not Available (N/A)"
                            if agent_i_type and agent_i_type != "medivacs":
                                damage_i = unit_properties[agent_i_type]["damage"]
                                ATTACK = f"Attack the {self.ordinal(j + 1)} enemy. "
                                # ATTACK = f"Attack {self.ordinal(j + 1)} enemy, dealing {damage_i} damage points to it. "
                                attack_or_not = [NA, ATTACK][obs[i][obs_e_enemy_dim * j + 4].to(th.int)]
                            else:
                                attack_or_not = NA

                            available_actions.append(attack_or_not)

                        if len(available_actions) < max_action_length:
                            available_actions.extend(["Not Available (N/A)"] * (max_action_length - len(available_actions)))

            else:
                non_medivac_actions = base_actions.copy()
                max_action_length = len(base_actions)

                for i in range(n_agents):
                    suffix = ['st', 'nd', 'rd'][i % 10] if 0 <= i <= 2 or (i >= 20 and 0 <= i % 10 <= 2) else 'th'
                    available_actions = base_actions.copy()
                    agent_i_type = agents_type_list[i]

                    for j in range(n_enemies):
                        NA = "Not Available (N/A)"
                        if agent_i_type and agent_i_type != "medivacs":
                            damage_i = unit_properties[agent_i_type]["damage"]
                            ATTACK = f"Attack the {self.ordinal(j + 1)} enemy. "
                            # ATTACK = f"Attack {self.ordinal(j + 1)} enemy, dealing {damage_i} damage points to it. "
                            attack_or_not = [NA, ATTACK][obs[i][obs_e_enemy_dim * j + 4].to(th.int)]
                        else:
                            attack_or_not = NA

                        available_actions.append(attack_or_not)
                    action_instructions += f"\n  - **For the {i + 1}{suffix} allied agent:**\n"
                    for idx, action in enumerate(available_actions, start=1):
                        action_instructions += f"    - **{idx}. {action}**\n"

                    max_action_length = max(max_action_length, len(available_actions))

                for i in range(n_agents):
                    available_actions = non_medivac_actions.copy()
                    agent_i_type = agents_type_list[i]
                    for j in range(n_enemies):
                        NA = "Not Available (N/A)"
                        if agent_i_type and agent_i_type != "medivacs":
                            damage_i = unit_properties[agent_i_type]["damage"]
                            ATTACK = f"Attack the {self.ordinal(j + 1)} enemy. "
                            # ATTACK = f"Attack {self.ordinal(j + 1)} enemy, dealing {damage_i} damage points to it. "
                            attack_or_not = [NA, ATTACK][obs[i][obs_e_enemy_dim * j + 4].to(th.int)]
                        else:
                            attack_or_not = NA
                        available_actions.append(attack_or_not)

                    if len(available_actions) < max_action_length:
                        available_actions.extend(["Not Available (N/A)"] * (max_action_length - len(available_actions)))

            print("\n\n-----------stage 4--------------\n\n")

            # User Prompt

            user_prompt_pre = f"Given the current game state, please provide the most optimal action recommendations for each of the {n_agents} allied agents. All agents should work together to achieve the best outcome, coordinating their actions for maximum effectiveness. Ensure that the actions chosen reflect the principles of coordination, communication, positioning, and efficiency as outlined in the system prompt.\n"

            system_prompt = part_1 + part_2 + part_3 + part_4 + part_5
            user_prompt = user_prompt_pre + ally_info + enemy_info + action_instructions

            return state, system_prompt, user_prompt


    def ordinal(self, n):
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def action2code(self, obs, state_tensor, coord_agent):
        state_plus_prompt = self.code2prompt(obs, state_tensor, coord_agent)
        state = state_plus_prompt[0]

        if state == "all_dead":
            action_tensor = th.zeros((self.n_agents, self.args.n_actions)).to(self.args.device)
            return action_tensor

        system_prompt = state_plus_prompt[1]
        user_prompt = state_plus_prompt[2]
        
        print("system prompt=\n", system_prompt)
        print("user prompt=\n", user_prompt)

        try:
            
            answer = self.llm_client.get_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=self.args.llm
            )

            print("\n## Answer = ", answer)

            # Define a pattern to match action vectors

            pattern = r"\[[^][]*\b(?:\d\s*,?\s*)\b[^][]*]"

            # Initialize an empty list to store the parsed action vectors
            actions_list = []

            # Split the response into lines and process each line
            for line in answer.splitlines():

                # print(f"line={line}")
                # Find all matches of the pattern in the current line
                match = re.search(pattern, line)
                if match:
                    matched_string = match.group(0)
                    try:
                        action_vector = literal_eval(matched_string)
                    # action_vector = [int(x) for x in match[0].split(',')]
                        if len(action_vector) == self.args.n_actions:
                            actions_list.append(action_vector)
                        if len(actions_list) == self.n_agents:
                            break
                    except:
                        actions_list.append([0]*self.args.n_actions)

            if len(actions_list) == self.n_agents:
                action_tensor = th.tensor(actions_list)
                return action_tensor

            # Check if we have collected the required number of action vectors
            if len(actions_list) != self.n_agents:

                try:

                    k = self.args.n_actions
                    pattern = r'\[\s*(?:\d+\s*,\s*){' + str(k - 1) + r'}\d+\s*\]'
                    matches = re.findall(pattern, answer)
                    action_list = [ast.literal_eval(lst) for lst in matches]
                    actions_tensor = th.tensor(action_list)[:self.n_agents]

                    if actions_tensor.size() == th.zeros([self.n_agents, self.args.n_actions]).size():
                        return actions_tensor
                    else:
                        return th.zeros(self.n_agents, self.args.n_actions)


                except:
                    print(answer)
                    print(f"\nNumber of action vectors ({len(actions_list)}) does not match the number of agents ({self.n_agents}).")
                    return th.zeros(self.n_agents, self.args.n_actions)

        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return th.zeros(self.n_agents, self.args.n_actions)


        # Combine all action tensors into a single n_agents * n_actions tensor
        # action_tensor = th.tensor(actions_list)
        #
        # except Exception as e:
        #     print(f"Error processing LLM response: {e}")
        #     action_tensor = th.zeros((self.n_agents, self.args.n_actions)).to(self.args.device)
        #
        # print(f"action_tensor={action_tensor}")

        # return action_tensor

    def inputs2action(self, obs, state, coord, epsilon):

        b, e = state.size()

        q_LLM = th.zeros(b, self.n_agents, self.args.n_actions).to(self.args.device)

        num_LLM = math.ceil(b * epsilon * 0.3)
        indices = th.randperm(b)[:num_LLM]
        # print(f"indices = {indices}")

        if th.tensor([0]) not in indices:
            indices = th.cat((indices, th.tensor([0])))

        # print(f"indices = {indices}")

        for ind in indices:
            action_i = self.action2code(obs[ind], state[ind], coord[ind]).to(self.args.device)
            # print("\n\n\n\n",action_i.size())
            q_LLM[ind] = action_i
            # print(f"action_i={action_i}")
            # except:
            # continue
        # print(f"q_LLM={q_LLM}")

        return q_LLM

    def inputs2action_2(self, obs, state, coord, epsilon):

        b, e = state.size()

        q_LLM = th.zeros(b, self.n_agents, self.args.n_actions).to(self.args.device)

        num_LLM = math.ceil(b * epsilon * 0.5)
        indices = th.randperm(b)[:num_LLM]
        
        for ind in indices:
            
            action_i = self.action2code(obs[ind], state[ind], coord[ind]).to(self.args.device)
                #print("\n\n\n\n",action_i.size())
            q_LLM[ind] = action_i
            #except:
                #continue

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
                description += f"and an initial shield of {props['maximal shield']} points, "
                description += f"dealing {props['damage']} points' damage per attack. "

                description = description.replace("colossu", "colossi")

            else:
                description += f"and deals {props['damage']} points' damage per attack. "

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



