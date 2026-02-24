import os
import shutil

source_dir = r"F:\STATS"  # 源文件夹路径
target_dir1 = r"F:\data_structure"  #提取结构数据
target_dir2 = r"F:\data_dose"  #提取剂量数据
target_dir3=r"F:\data_plan"  #提取计划数据
target_dir4=r"F:\data_CT"  #提取CT数据

for d in (target_dir1, target_dir2, target_dir3, target_dir4):
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------
# 1) 处理 RS 开头的文件：扁平化到 target_dir1，按文件夹重命名
# -------------------------------------------------
for root, dirs, files in os.walk(source_dir):  #walk会返回绝对路径、文件、文件名
    for file in files:
        if file.startswith("RS"):
            src_path = os.path.join(root, file) #拼接查找文件的绝对路径
            folder_name = os.path.basename(root) #只保留路径的最后一段
            _, ext = os.path.splitext(file)#将文件名拆分成两个元组，名称以及格式
            new_name = f"{folder_name}{ext}"

            counter = 1
            dst_path = os.path.join(target_dir1, new_name)
            while os.path.exists(dst_path):
                new_name = f"{folder_name}_{counter}{ext}"
                dst_path = os.path.join(target_dir1, new_name)
                counter += 1

            shutil.copy2(src_path, dst_path)  
            print(f"已复制 RS：{src_path} -> {dst_path}")
for root, dirs, files in os.walk(source_dir):  
    for file in files:
        if file.startswith("RD"):
            src_path = os.path.join(root, file) 
            folder_name = os.path.basename(root) 
            _, ext = os.path.splitext(file)
            new_name = f"{folder_name}{ext}"

            counter = 1
            dst_path = os.path.join(target_dir2, new_name)
            while os.path.exists(dst_path):
                new_name = f"{folder_name}_{counter}{ext}"
                dst_path = os.path.join(target_dir2, new_name)
                counter += 1

            shutil.copy2(src_path, dst_path)  #将scr_path文件复制到dst_path
            print(f"已复制 RD：{src_path} -> {dst_path}")
for root, dirs, files in os.walk(source_dir):  #walk会返回绝对路径、文件、文件名
    for file in files:
        if file.startswith("RP"):
            src_path = os.path.join(root, file) #拼接查找文件的绝对路径
            folder_name = os.path.basename(root) #只保留路径的最后一段，例：root = "G:\datas\patient01" → folder_name = "patient01"
            _, ext = os.path.splitext(file)#将文件名拆分成两个元组，名称以及格式
            new_name = f"{folder_name}{ext}"

            counter = 1
            dst_path = os.path.join(target_dir3, new_name)
            while os.path.exists(dst_path):
                new_name = f"{folder_name}_{counter}{ext}"
                dst_path = os.path.join(target_dir3, new_name)
                counter += 1

            shutil.copy2(src_path, dst_path)  #将scr_path文件复制到dst_path
            print(f"已复制 RP：{src_path} -> {dst_path}")
# -------------------------------------------------
# 2) 处理 CT 开头的文件：保持目录结构到 target_dir2
# -------------------------------------------------
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.startswith("CT"):
            src_path = os.path.join(root, file)

            # 计算相对路径（相对于 source_dir）
            rel_dir = os.path.relpath(root, source_dir)      # 形如 'sub1\\sub2'
            dst_dir = os.path.join(target_dir4, rel_dir)     # 目标子文件夹

            os.makedirs(dst_dir, exist_ok=True)              # 确保子文件夹存在
            dst_path = os.path.join(dst_dir, file)           # 目标完整路径

            # 如果同名文件已存在，追加 _n
            counter = 1
            name, ext = os.path.splitext(file)
            while os.path.exists(dst_path):
                new_name = f"{name}_{counter}{ext}"
                dst_path = os.path.join(dst_dir, new_name)
                counter += 1

            shutil.copy2(src_path, dst_path)
            print(f"已复制 CT：{src_path} -> {dst_path}")
