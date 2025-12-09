
#define TILE 16

__kernel void linear_kernel(
    __global const float* input,   // [tokens * in_features]
    __global const float* weight,  // [out_features * in_features]
    __global const float* bias,    // [out_features]
    __global float* output,        // [tokens * out_features]
    int tokens,
    int in_features,
    int out_features
) {
    const int blockRow = get_group_id(0);   // token block
    const int blockCol = get_group_id(1);   // out_feature block

    const int localRow = get_local_id(0);   // 0..TILE-1
    const int localCol = get_local_id(1);   // 0..TILE-1

    const int row = blockRow * TILE + localRow; // token idx
    const int col = blockCol * TILE + localCol; // out_feature idx

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float sum = 0.0f;

    // loop over k tiles
    for (int kBase = 0; kBase < in_features; kBase += TILE) {
        int a_k = kBase + localCol;   // column within input tile
        int b_k = kBase + localRow;   // row within weight tile

        // load A (input) element -> As[localRow][localCol]
        if (row < tokens && a_k < in_features) {
            As[localRow][localCol] = input[row * in_features + a_k];
        } else {
            As[localRow][localCol] = 0.0f;
        }

        // load B (weight) element -> Bs[localRow][localCol]
        if (col < out_features && b_k < in_features) {
            // weight layout: [out_feature * in_features + k]
            Bs[localRow][localCol] = weight[col * in_features + b_k];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // compute partial
        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[localRow][kk] * Bs[kk][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result only if in-bounds
    if (row < tokens && col < out_features) {
        output[row * out_features + col] = sum + bias[col];
    }
}

__kernel void gelu_kernel_inplace(__global float* data, int N) {
    int gid = get_global_id(0);
    if (gid >= N) return;

    float x = data[gid];
    float y = 0.5f * x * (1.0f + erf(x / sqrt((float)2)));
    data[gid] = y;
}

