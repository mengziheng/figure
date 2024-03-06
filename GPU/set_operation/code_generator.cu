#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

int stdout_fd;
int indentation = 0;

void print_statement(const char *fmt...) {
    va_list args;
    va_start(args, fmt);
    for(int _ = 0; _ < indentation; _++) putchar(' '); 
    vprintf(fmt, args);
}

int main(){
    stdout_fd = dup(STDOUT_FILENO);
    freopen("tmp", "w", stdout);
    print_statement("__global__ void gpu_pattern_matching_generated(e_index_t edge_num, uint32_t buffer_size, PatternMatchingDeviceContext *context) {\n");
    fflush(stdout);
    return 0;
}