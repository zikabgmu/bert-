import os
def clean_dataset(file_path):
    cleaned_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # 去除前后空格
            line = line.strip()
            # 检查行是否包含制表符并能正确分割
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2 and parts[1].isdigit():  # 确保有两部分且标签是数字
                    cleaned_lines.append(line)
                else:
                    print(f"跳过格式错误的行: {line}")
            else:
                print(f"跳过无效行: {line}")

    # 将清理后的数据写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines) + "\n")
    print(f"{file_path} 已清理完成！")


# 清理所有数据文件
base_path = "../ERNIE/datas/data"
for file_name in ["train.txt", "dev.txt", "test.txt"]:
    file_path = os.path.join(base_path, file_name)
    clean_dataset(file_path)
