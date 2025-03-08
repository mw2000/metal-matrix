#include <metal_stdlib>
using namespace metal;

kernel void matrix_transpose(device const float* A,
                            device float* B,
                            constant uint& rows,
                            constant uint& cols,
                            uint2 position [[thread_position_in_grid]])
{
    uint row = position.y;
    uint col = position.x;
    
    // Ensure we're within bounds
    if (row < rows && col < cols) {
        // B[col, row] = A[row, col]
        B[col * rows + row] = A[row * cols + col];
    }
} 