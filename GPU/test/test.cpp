#include<iostream>
using namespace std;

void print_binary(unsigned int number) {
    if (number >> 1) {
        print_binary(number >> 1);
    }
    putc((number & 1) ? '1' : '0', stdout);
}

int main(){
    int a = 0xffffffff;
    int b = ~0;
    bool c = a==b;
    int t = 32;
    printf("%d",t&0x1f);
}