#include <cstdint>

__device__ uint32_t a, b, c;

__global__ void kernel()
{
    c = a / b;
}

int main(int argc, char *argv[])
{
    kernel<<<1, 1>>>();
    return 0;
}