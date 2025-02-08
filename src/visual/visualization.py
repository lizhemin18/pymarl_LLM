# Author: Icy
# Date  : 2025-1-17
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

# Delete any previously processed files
for existing_file in glob.glob('episode*_process.txt'):
    os.remove(existing_file)

input_files = sorted(glob.glob('episode*.txt'))

def detect_encoding(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

for input_file_name in input_files:
    # Construct output file name
    output_file_name = input_file_name.replace('.txt', '_process.txt')
    encoding = detect_encoding(input_file_name)

    with open(input_file_name, 'r', encoding=encoding) as file:
        lines = file.readlines()

    tensor_data = []
    inside_tensor = False
    tensor_name = ''
    mask_numbers = set()
    adj = []

    # Store processed data for each episode
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

    # Format and write final processed result for each episode
    final_output = []
    for line in processed_lines:
        if 'top6_after:tensor' in line:
            tensor_data = line.strip().replace('top6_after:tensor(', '').replace(')', '')
            tensor_data = ast.literal_eval(tensor_data)
            for i, data in enumerate(tensor_data):
                final_output.append(f'c{i + 1}:{data}\n')
        else:
            final_output.append(line)

    # Write to output file
    with open(output_file_name, 'w', encoding='utf-8') as file:
        file.writelines(final_output)

# Set up command-line arguments
parser = argparse.ArgumentParser(description="Agent Simulation Visualization")
parser.add_argument('--selected_agent', type=int, required=True, help="The index of the selected agent")
parser.add_argument('--episode', type=int, required=True, help="The episode number to visualize")
parser.add_argument('--time_steps', type=int, nargs='+', required=True, help="List of time steps to visualize communication at")
args = parser.parse_args()

# Get values from command-line arguments
selected_agent = args.selected_agent
episode_file = f'episode{args.episode}_process.txt'
time_steps = args.time_steps


# Animation parameters
num_time_steps = 31
num_my_agents = 10
num_enemy_agents = 11

# Load images
my_img = mpimg.imread('my2.png')
enemy_img = mpimg.imread('enemy2.png')

fig, ax = plt.subplots(figsize=(8, 12))

my_agents_scatter = ax.scatter([], [], c='blue', label='My Agents', zorder=5, alpha=0)
enemy_agents_scatter = ax.scatter([], [], c='red', label='Enemy Agents', zorder=5, alpha=0)

agent_images = []
healthbars = []
crosses = []
communication_lines = []
communication_circles = []


max_x = -np.inf
max_y = -np.inf
min_x = np.inf
min_y = np.inf

# 添加编号的全局变量，用于在每一帧清除之前的编号
agent_numbers = []

def update(frame):
    global max_x, max_y, min_x, min_y, agent_numbers, communication_circles

    my_agents_positions = []
    my_agents_health = []
    mask_vector = []

    enemy_agents_positions = []
    enemy_agents_health = []

    top6_after = []
    # 读取当前帧的数据
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


    # 清理之前的图像、健康条、编号、交互线
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

    for circle in communication_circles:
        circle.remove()
    communication_circles.clear()

    # 清除编号
    for label in agent_numbers:
        label.remove()
    agent_numbers.clear()

    # 添加我方智能体图像及健康条，并显示编号
    for i, (pos, health) in enumerate(zip(my_agents_positions, my_agents_health)):
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

            # 使用数据中的编号显示编号
            label = ax.text(pos[0] + 0.2, pos[1] + 0.2, f'my{i + 1}', fontsize=10, color='black', zorder=20)
            agent_numbers.append(label)

            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            max_x = max(max_x, pos[0])
            max_y = max(max_y, pos[1])

    # 添加敌方智能体图像及健康条，并显示编号
    for i, (pos, health) in enumerate(zip(enemy_agents_positions, enemy_agents_health)):
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

            # 使用数据中的编号显示编号
            label = ax.text(pos[0] + 0.2, pos[1] + 0.2, f'enemy{i + 1}', fontsize=10, color='black', zorder=20)
            agent_numbers.append(label)

            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            max_x = max(max_x, pos[0])
            max_y = max(max_y, pos[1])

    # 绘制mask
    for idx in mask_vector:
        if idx < len(my_agents_positions) and my_agents_health[idx] > 0:
            pos = my_agents_positions[idx]
            cross = ax.scatter(pos[0], pos[1], color='black', marker='x', s=100, zorder=20)
            crosses.append(cross)
    # 清理之前的绿色圆圈

    for circle in communication_circles:
        circle.remove()
    communication_circles.clear()

    # 检查选定的智能体是否被mask
    if selected_agent < len(my_agents_positions) and my_agents_health[selected_agent] > 0:
        pos = my_agents_positions[selected_agent]

        # 无论是否被mask，始终绘制黄色圆圈
        circle = patches.Circle(pos, radius=0.5, linewidth=2, edgecolor='goldenrod', facecolor='none', zorder=15)
        ax.add_patch(circle)
        communication_circles.append(circle)

        # 每个时间步绘制半径为6的灰色圆圈
        gray_circle = patches.Circle(pos, radius=6, linewidth=2, edgecolor='gray', facecolor='none', linestyle='--',
                                     zorder=10)
        ax.add_patch(gray_circle)
        communication_circles.append(gray_circle)

        # 绘制绿色连线和绿色圆圈，只有在当前智能体和通信智能体都没有被mask时
        connections = top6_after[selected_agent]
        for idx in connections:
            if idx < len(my_agents_positions) and my_agents_health[idx] > 0:
                pos = my_agents_positions[idx]

                # 绘制绿色圆圈，只有当通信智能体没有被mask时
                if selected_agent not in mask_vector and idx not in mask_vector:
                    circle = patches.Circle(pos, radius=0.5, linewidth=2, edgecolor='green', facecolor='none',
                                            zorder=15)
                    ax.add_patch(circle)
                    communication_circles.append(circle)

                # 绘制绿色连线，确保选定智能体和通信智能体都没有被mask
                if selected_agent not in mask_vector and idx not in mask_vector:
                    line = ax.plot([my_agents_positions[selected_agent][0], my_agents_positions[idx][0]],
                                   [my_agents_positions[selected_agent][1], my_agents_positions[idx][1]], 'g-',
                                   alpha=0.5, zorder=8)
                    communication_lines.append(line[0])

    # 为通信的智能体之间添加绿色连线
    if selected_agent < len(top6_after) and my_agents_health[selected_agent] > 0 and selected_agent not in mask_vector:
        connections = top6_after[selected_agent]
        for j in connections:
            if j < len(my_agents_positions) and my_agents_health[j] > 0 and j not in mask_vector:
                line = ax.plot([my_agents_positions[selected_agent][0], my_agents_positions[j][0]],
                               [my_agents_positions[selected_agent][1], my_agents_positions[j][1]], 'g-', alpha=0.5,
                               zorder=8)
                communication_lines.append(line[0])

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)

    # 显示选定智能体在该时间步的通信情况
    if frame in time_steps:
        if selected_agent < len(top6_after):
            connections = top6_after[selected_agent]
            # Check if the selected agent is in the mask
            if selected_agent in mask_vector:
                print(f"At time step {frame}, agent {selected_agent + 1} cannot communicate because it is masked.")
            elif connections:
                print(f"At time step {frame}, agent {selected_agent + 1} is communicating with agents: {connections}")
            else:
                print(f"At time step {frame}, agent {selected_agent + 1} is not communicating with any other agents.")

        plt.savefig(f"frame_at_{frame}.png")
        print(f"Saved frame at time step {frame} as 'frame_at_{frame}.png'")

    return my_agents_scatter, enemy_agents_scatter, *agent_images, *healthbars, *crosses, *communication_lines, *communication_circles


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_time_steps, interval=500, repeat=False)

# Add pause/resume functionality
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

# Remove extra axes if any
if len(fig.axes) > 1:
    fig.delaxes(fig.axes[1])

# Add legend
my_agent_legend = plt.Line2D([], [], color='red', marker='o', markersize=5, linestyle='None')
enemy_agent_legend = plt.Line2D([], [], color='blue', marker='o', markersize=5, linestyle='None')
plt.legend([my_agent_legend, enemy_agent_legend], ['Ally', 'Enemy'], loc='upper right')

# Title and labels
plt.title('Agent Simulation')
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Show the plot
plt.show()

