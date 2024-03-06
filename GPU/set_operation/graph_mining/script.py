import os
import argparse


def generate_header_file(maxdegree, h, block_size, block_num):
    header_file = "constants.h"
    content = f"#ifndef CONSTANTS_HEADER\n"
    content = content + "#define CONSTANTS_HEADER\n"
    content = content + f"#define MaxDegree {maxdegree}\n"
    content = content + f"#define H {h}\n"
    content = content + f"#define BLOCK_SIZE {block_size}\n"
    content = content + f"#define BLOCK_NUM {block_num}\n"
    content = content + "#endif"
    try:
        with open(header_file, "r") as file:
            existing_content = file.read()
    except FileNotFoundError:
        existing_content = None

    if existing_content != content:
        with open(header_file, "w") as file:
            file.write(content)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument('--input_graph_folder', type=str, default="")
    parser.add_argument('--input_pattern', type=str, default='Q0')
    parser.add_argument('--block_size', type=str, default='216')
    parser.add_argument('--block_num', type=str, default='1024')

    # 解析命令行参数
    args = parser.parse_args()

    # 检查graph和pattern是否被提供
    if 'input_graph_folder' not in args:
        parser.error("--input_graph_folder is required.")
    if 'input_pattern' not in args:
        parser.error("--input_pattern is required.")

    input_graph_folder = args.input_graph_folder
    input_pattern = args.input_pattern
    block_size = args.block_size
    block_num = args.block_num

    if input_pattern == "Q0" or input_pattern == "Q3" or input_pattern == "Q5" or input_pattern == "Q7" or "clique" in input_pattern:
        maxdegree_file = input_graph_folder + "/md.bin"
    else:
        maxdegree_file = input_graph_folder + "/generic_md.bin"
    try:
        with open(maxdegree_file, "rb") as pFile:
            maxdegree = int.from_bytes(pFile.read(4), byteorder='little')
    except FileNotFoundError:
        print("error for max_degree_file")

    # 完整的系统需要增加一个解析pattern的功能，不然支持的pattern有限
    H = 4

    generate_header_file(maxdegree, H, block_size, block_num)

    command = f"make"
    os.system(command)
    command = f"./subgraphmatch.bin {input_graph_folder} {input_pattern} 1 0.1 8 216 1024 10"
    os.system(command)
