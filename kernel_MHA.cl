__kernel void gemm(	
	__global float* A,
	__global float* B,
	__global float* C,
	int M, int N, int K) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = 0;
	
	float sum = 0;
	for (k = 0; k < K; k++) {
		sum += A[j * K + k] * B[k * N + i];
	}
	C[j * N + i] = sum;
}
__kernel void transpose(
    __global float* src,
    __global float* dst,
    int row, int col) {
    int r = get_global_id(0);
    int c = get_global_id(1);

    dst[c * row + r] = src[r * col + c];
}
__kernel void add_bias(	
	__global float* M,
	__global float* B,
	int embed_dim) {
    int i = get_global_id(0);
    int t = get_global_id(1);

    M[t * embed_dim + i] += B[i];
}	
__kernel void divide_head(
    __global float* src,
    __global float* dst,
    int embed_dim,
    int head_dim,
    int head_idx) {
    int t = get_global_id(0);
    int d = get_global_id(1);
    
    int offset = head_idx * head_dim;
    dst[t * head_dim + d] = src[t * embed_dim + offset + d];
}
__kernel void scale_score(
    __global float* src,
    float scaler) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    src[i * get_global_size(0) + j] /= sqrt(scaler);
}
__kernel void softmax_score(
    __global float* src,
    __local float* tmp,
    int padding) {
/*
각 i에 대해서
맥스값 구하기
exp(원본값-맥스) 대입
exp의 합 구하기
exp합으로 나누기
*/
    int i = get_global_id(0);
    int j = get_global_id(1);
    int tokens = get_global_size(0);

    // 로컬 값 대입
    tmp[j] = src[i * tokens + j];
	barrier(CLK_LOCAL_MEM_FENCE);

    // 맥스 구하기
	for (int p = padding / 2; p >= 1; p = p >> 1) {
        int shiftedIdx = j + p;
        float shiftedValue = -INFINITY;
        if(shiftedIdx < tokens) shiftedValue = tmp[shiftedIdx];
        if (j < p && tmp[j] < shiftedValue) tmp[j] = shiftedValue;				
		barrier(CLK_LOCAL_MEM_FENCE);					
	}	
    float max = tmp[0];
	barrier(CLK_LOCAL_MEM_FENCE);
    
    // exp 대입
    tmp[j] = exp(src[i * tokens + j] - max);
	barrier(CLK_LOCAL_MEM_FENCE);
    
    // exp 합 구하기
	for (int p = padding / 2; p >= 1; p = p >> 1) {		
        int shiftedIdx = j + p;
        float shiftedValue = 0;
        if(shiftedIdx < tokens) shiftedValue = tmp[shiftedIdx];
        if (j < p) tmp[j] += shiftedValue;				
		barrier(CLK_LOCAL_MEM_FENCE);					
	}	
    float sum = tmp[0];
	barrier(CLK_LOCAL_MEM_FENCE);

    // exp합으로 나눠서 최종값 도출
    src[i * tokens + j] = exp(src[i * tokens + j] - max) / sum;
}

__kernel void copy_head_output(
    __global float* attn,
    __global float* head,
    int embed_dim,
    int head_offset) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    attn[i * embed_dim + head_offset + j] = head[i * get_global_size(1) + j];
}
////////////////////////////////////////////////////////////////////////////////////////

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

#define M_SQRT2PI 0.7978845608028654f  // sqrt(2/pi)

__kernel void gelu_kernel_inplace(__global float* data, int N) {
    int gid = get_global_id(0);
    if (gid >= N) return;

    float x = data[gid];
    // approximate GELU
    float x3 = x * x * x;
    float t = 0.044715f * x3 + x;
    float y = 0.5f * x * (1.0f + tanh(M_SQRT2PI * t));
    data[gid] = y;
}
