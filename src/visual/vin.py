import os
import re
import ast
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
import chardet

filename = 'cout.txt'

def split_file_by_t0(filename):
    file_counter = 1
    lines = []

    with open(filename, 'r', encoding='utf-8') as file:  # 指定编码格式为 utf-8
        for line in file:
            if line.startswith('t: 0') and lines:
                output_filename = f'episode{file_counter}.txt'
                with open(output_filename, 'w', encoding='utf-8') as output_file:  # 同样指定写入时的编码格式
                    output_file.writelines(lines)
                lines = []
                file_counter += 1
            lines.append(line)

        # Write the last set of lines if any
        if lines:
            output_filename = f'episode{file_counter}.txt'
            with open(output_filename, 'w', encoding='utf-8') as output_file:
                output_file.writelines(lines)

split_file_by_t0(filename)

for existing_file in glob.glob('episode*_process.txt'):
    os.remove(existing_file)

input_files = sorted(glob.glob('episode*.txt'))
def detect_encoding(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

for input_file_name in input_files:
    # 构造输出文件名
    output_file_name = input_file_name.replace('.txt', '_process.txt')
    encoding = detect_encoding(input_file_name)

    with open(input_file_name, 'r', encoding=encoding) as file:
        lines = file.readlines()

    tensor_data = []
    inside_tensor = False
    tensor_name = ''
    mask_numbers = set()
    adj = []

    # 用于存储每个episode的最终处理结果
    processed_lines = []

    for line in lines:
        if 'state: tensor' in line:
            inside_tensor = True
            tensor_data.append(line.strip())
        elif 'visible_matrix: tensor' in line or 'top6_after: tensor' in line:
            inside_tensor = True
            tensor_data.append(line.strip())
            tensor_name = line.strip().split(':')[0]
        elif 'mask:' in line:
            line = line.replace(' ', '')
            mask_numbers = set()
            mask_numbers.update(re.findall(r'\d+', line))
            processed_lines.append(line)
        elif inside_tensor:
            tensor_data.append(line.strip())
            if '])' in line:
                inside_tensor = False
                full_tensor_data = ' '.join(tensor_data)
                if 'state: tensor' in full_tensor_data:
                    tensor_values = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', full_tensor_data)

                    my = tensor_values[:40]
                    my_lines = []
                    for i in range(10):
                        index = i * 4
                        if index + 3 < len(my):
                            x_coord = float(my[index + 2]) * 32 + 16
                            y_coord = float(my[index + 3]) * 32 + 16
                            my_vector = f'my{i + 1}: [{my[index]}, {x_coord}, {y_coord}]\n'
                            my_lines.append(my_vector)

                    enemy = tensor_values[40:73]
                    enemy_lines = []
                    for i in range(11):
                        index = i * 3
                        if index + 2 < len(enemy):
                            x_coord = float(enemy[index + 1]) * 32 + 16
                            y_coord = float(enemy[index + 2]) * 32 + 16
                            enemy_vector = f'enemy{i + 1}: [{enemy[index]}, {x_coord}, {y_coord}]\n'
                            enemy_lines.append(enemy_vector)

                    processed_lines.extend(my_lines)
                    processed_lines.extend(enemy_lines)
                elif tensor_name == 'visible_matrix':
                    full_tensor_data = full_tensor_data.replace('visible_matrix: tensor(', '').replace(')', '')
                    full_tensor_data = re.sub(r'\.', '', full_tensor_data)
                    full_tensor_data = full_tensor_data.replace(' ', '')
                    adj = ast.literal_eval(full_tensor_data)

                    processed_lines.append(f'visible_matrix:tensor({full_tensor_data})\n')
                elif tensor_name == 'top6_after':
                    full_tensor_data = full_tensor_data.replace('top6_after: tensor(', '').replace(')', '')
                    full_tensor_data = full_tensor_data.replace(' ', '')
                    processed = []

                    data = ast.literal_eval(full_tensor_data)
                    for c, node in enumerate(data):
                        cnt = []
                        for k in node:
                            if str(k) not in mask_numbers and adj[c][k] == 1:
                                cnt.append(k)
                        processed.append(cnt)
                    processed_lines.append(f'top6_after:tensor({processed})\n')
                tensor_data = []
                tensor_name = ''
        else:
            processed_lines.append(line)

    # 处理 top6_after 的输出格式并写入最终结果
    final_output = []
    for line in processed_lines:
        if 'top6_after:tensor' in line:
            tensor_data = line.strip().replace('top6_after:tensor(', '').replace(')', '')
            tensor_data = ast.literal_eval(tensor_data)
            for i, data in enumerate(tensor_data):
                final_output.append(f'c{i + 1}:{data}\n')
        else:
            final_output.append(line)

    # 将每个 episode 的最终结果写入对应的文件
    with open(output_file_name, 'w',encoding='utf-8') as file:
        file.writelines(final_output)

# 设置命令行参数
parser = argparse.ArgumentParser(description="Agent Simulation Visualization")
parser.add_argument('--selected_agent', type=int, required=True, help="The index of the selected agent")
parser.add_argument('--episode', type=int, required=True, help="The episode number to visualize")
args = parser.parse_args()

# 从命令行参数中获取值
selected_agent = args.selected_agent
episode_file = f'episode{args.episode}_process.txt'

# 动画参数
num_time_steps = 31
num_my_agents = 10
num_enemy_agents = 11

# 加载图片
my_img = mpimg.imread('my1.png')
enemy_img = mpimg.imread('enemy1.png')

fig, ax = plt.subplots(figsize=(8, 12))

my_agents_scatter = ax.scatter([], [], c='blue', label='My Agents', zorder=5, alpha=0)
enemy_agents_scatter = ax.scatter([], [], c='red', label='Enemy Agents', zorder=5, alpha=0)

agent_images = []
healthbars = []
crosses = []
communication_lines = []

max_x = -np.inf
max_y = -np.inf
min_x = np.inf
min_y = np.inf

def update(frame):
    global max_x, max_y, min_x, min_y

    my_agents_positions = []
    my_agents_health = []
    mask_vector = []

    enemy_agents_positions = []
    enemy_agents_health = []

    top6_after = []

    with open(episode_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith(f't: {frame}'):
                break
        current_frame_data = lines[lines.index(line) + 1:]
        for data_line in current_frame_data:
            if data_line.startswith('t:'):
                break
            if data_line.startswith('my'):
                parts = data_line.strip().split(': ')[1]
                pos_and_health = [float(x) for x in parts.strip('[]').split(', ')]
                my_agents_positions.append(pos_and_health[1:3])
                my_agents_health.append(pos_and_health[0])
            elif data_line.startswith('enemy'):
                parts = data_line.strip().split(': ')[1]
                pos_and_health = [float(x) for x in parts.strip('[]').split(', ')]
                enemy_agents_positions.append(pos_and_health[1:3])
                enemy_agents_health.append(pos_and_health[0])
            elif data_line.startswith('mask'):
                mask_vector = [int(x) for x in data_line.strip().split(':')[1].strip('[]').split(',')]
            elif data_line.startswith('c'):
                idx = int(data_line[1:data_line.index(':')]) - 1
                connections = [int(x) for x in data_line[data_line.index(':') + 1:].strip().strip('[]').split(',') if x.strip()]
                while len(top6_after) <= idx:
                    top6_after.append([])
                top6_after[idx] = connections

    for image in agent_images:
        image.remove()
    agent_images.clear()

    for bar in healthbars:
        bar.remove()
    healthbars.clear()

    for cross in crosses:
        cross.remove()
    crosses.clear()

    for line in communication_lines:
        line.remove()
    communication_lines.clear()

    for pos, health in zip(my_agents_positions, my_agents_health):
        if health > 0:
            image = OffsetImage(my_img, zoom=0.045)
            ab = AnnotationBbox(image, pos, frameon=False, zorder=10)
            agent_images.append(ab)
            ax.add_artist(ab)

            bar_length = health / 3.5
            bar_height = 0.03
            healthbar_bg = patches.Rectangle((pos[0] - 0.5, pos[1] + 0.3), 1, bar_height, linewidth=0.5, edgecolor='black', facecolor='white', zorder=9, transform=ax.transData)
            healthbar = patches.Rectangle((pos[0] - 0.5, pos[1] + 0.3), bar_length, bar_height, linewidth=0.5, edgecolor='black', facecolor='red', zorder=10, transform=ax.transData)
            ax.add_patch(healthbar_bg)
            ax.add_patch(healthbar)
            healthbars.append(healthbar_bg)
            healthbars.append(healthbar)

            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            max_x = max(max_x, pos[0])
            max_y = max(max_y, pos[1])

    for pos, health in zip(enemy_agents_positions, enemy_agents_health):
        if health > 0:
            image = OffsetImage(enemy_img, zoom=0.045)
            ab = AnnotationBbox(image, pos, frameon=False, zorder=10)
            agent_images.append(ab)
            ax.add_artist(ab)

            bar_length = health / 3.5
            bar_height = 0.03
            healthbar_bg = patches.Rectangle((pos[0] - 0.5, pos[1] + 0.3), 1, bar_height, linewidth=0.5, edgecolor='black', facecolor='white', zorder=9, transform=ax.transData)
            healthbar = patches.Rectangle((pos[0] - 0.5, pos[1] + 0.3), bar_length, bar_height, linewidth=0.5, edgecolor='black', facecolor='blue', zorder=10, transform=ax.transData)
            ax.add_patch(healthbar_bg)
            ax.add_patch(healthbar)
            healthbars.append(healthbar_bg)
            healthbars.append(healthbar)

            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            max_x = max(max_x, pos[0])
            max_y = max(max_y, pos[1])

    for idx in mask_vector:
        if idx < len(my_agents_positions) and my_agents_health[idx] > 0:
            pos = my_agents_positions[idx]
            cross = ax.scatter(pos[0], pos[1], color='black', marker='x', s=100, zorder=20)
            crosses.append(cross)

    if selected_agent < len(top6_after) and my_agents_health[selected_agent] > 0 and selected_agent not in mask_vector:
        connections = top6_after[selected_agent]
        for j in connections:
            if j < len(my_agents_positions) and my_agents_health[j] > 0 and j not in mask_vector:
                line = ax.plot([my_agents_positions[selected_agent][0], my_agents_positions[j][0]], [my_agents_positions[selected_agent][1], my_agents_positions[j][1]], 'g-', alpha=0.5, zorder=8)
                communication_lines.append(line[0])

    ax.set_xlim(min_x - 2, max_x + 2)
    ax.set_ylim(min_y - 2, max_y + 2)

    return my_agents_scatter, enemy_agents_scatter, *agent_images, *healthbars, *crosses, *communication_lines

ani = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=500, repeat=False)

is_paused = False
def on_click(event):
    global is_paused
    if is_paused:
        ani.resume()
        is_paused = False
    else:
        ani.pause()
        is_paused = True
cid = fig.canvas.mpl_connect('button_press_event', on_click)

if len(fig.axes) > 1:
    fig.delaxes(fig.axes[1])

my_agent_legend = plt.Line2D([], [], color='red', marker='o', markersize=5, linestyle='None')
enemy_agent_legend = plt.Line2D([], [], color='blue', marker='o', markersize=5, linestyle='None')
plt.legend([my_agent_legend, enemy_agent_legend], ['Ally', 'Enemy'], loc='upper right')

plt.title('Agent Simulation')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()
