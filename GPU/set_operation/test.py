import subprocess
import re
import os




if __name__ == "__main__":
    
    s = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                                  sector                              311 sector 331"
    
    regex_pattern = r"sector\s+(\d+,*\d*)"
    
    match = re.search(regex_pattern, s)
    if match:
        graph_name = match.group(1)
        print(graph_name)
    result = []
    result.append(1)
    result.append(2)
    print(f"{result[0]}/{result[1]}")
            