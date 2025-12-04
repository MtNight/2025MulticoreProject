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

__kernel void linear_kernel(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    int tokens,
    int in_features,
    int out_features
) {
    int t = get_global_id(0); // token
    int o = get_global_id(1); // output feature

    if (t >= tokens || o >= out_features) return;

    float sum = bias[o];
    int input_offset = t * in_features;
    int weight_offset = o * in_features;

    for (int i = 0; i < in_features; i++) {
        sum += input[input_offset + i] * weight[weight_offset + i];
    }

    output[t * out_features + o] = sum;
}
