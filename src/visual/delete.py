# Author: Icy
# Date  : 2024-11-18
import os

# 获取当前目录
current_dir = os.getcwd()

# 遍历当前目录下的所有文件
for file_name in os.listdir(current_dir):
    # 检查文件是否以 "episode" 开头
    if file_name.startswith("episode"):
        file_path = os.path.join(current_dir, file_name)
        # 确认是文件而非目录
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_name}")

print("完成删除操作。")
