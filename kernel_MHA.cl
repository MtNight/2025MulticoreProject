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
    int rows, int cols) {
    int r = get_global_id(0);
    int c = get_global_id(1);

    dst[c * rows + r] = src[r * cols + c];
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