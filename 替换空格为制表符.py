import os
def fix_spaces_to_tabs(file_path):
    fixed_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # 将第一个空格替换为制表符
            if " " in line and "\t" not in line:
                parts = line.split(" ", 1)  # 仅分割第一个空格
                if len(parts) == 2 and parts[1].isdigit():  # 确保第二部分是数字标签
                    fixed_lines.append("\t".join(parts))
                    continue
            fixed_lines.append(line)

    # 写回修正后的文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines) + "\n")
    print(f"{file_path} 的空格已修正为制表符！")

# 修正所有数据文件
base_path = "../ERNIE/datas/data"
for file_name in ["train.txt", "dev.txt", "test.txt"]:
    file_path = os.path.join(base_path, file_name)
    fix_spaces_to_tabs(file_path)
