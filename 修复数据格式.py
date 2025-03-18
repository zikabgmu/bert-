def fix_dataset(file_path):
    fixed_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 去掉前后空格
            if "\t" not in line:  # 检查是否包含制表符
                parts = line.split(" ", 1)  # 尝试用空格分割
                if len(parts) == 2 and parts[1].isdigit():
                    fixed_lines.append("\t".join(parts))  # 替换为空格
                    continue
            if len(line.split("\t")) == 2:  # 确保分隔符正确
                fixed_lines.append(line)
            else:
                print(f"跳过格式错误的行: {line}")

    # 写回修正后的文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines) + "\n")
    print(f"{file_path} 数据已修复！")

# 修复你的数据文件
file_path = "../ERNIE/datas/data/test.txt"  # 修改为你的文件路径
fix_dataset(file_path)
