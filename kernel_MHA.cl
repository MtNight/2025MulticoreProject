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
    int ROW, int COL) {
    int r = get_global_id(0);
    int c = get_global_id(1);

    dst[c * ROW + r] = src[r * COL + c];
}
__kernel void add_bias(	
	__global float* M,
	__global float* B,
	int ROW) {
    int i = get_global_id(0);
    int t = get_global_id(1);

    M[t * ROW + i] += B[i];
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
    float scaler,
    int tokens) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    src[i * tokens + j] /= sqrt(scaler);
}
__kernel void softmax_score(
    __global float* src,
    __local float* tmp,
    int padding,
    int tokens) {
    int i = get_global_id(0);
    int j = get_global_id(1);

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
    float e = tmp[j];
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
    src[i * tokens + j] = e / sum;
}
__kernel void copy_head_output(
    __global float* attn,
    __global float* head,
    int embed_dim,
    int head_offset,
    int head_dim) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    attn[i * embed_dim + head_offset + j] = head[i * head_dim + j];
}
__kernel void layer_normalize(
    __global float* src,
    __global float* weight, 
    __global float* bias,
    __global float* dst,
    __local float* tmp1,
    __local float* tmp2,
    __local float* statistic,
    int tokens,
    int embed_dim) {
    int t = get_global_id(0);
    int d = get_global_id(1);
    
    float sum = 0;
    float sum_sq = 0;
    for(int block = 0; block < 3; block++) {
        int idx = t * embed_dim + block*256 + d;
        float origin = src[idx];
        tmp1[d] = origin;
        tmp2[d] = origin * origin;
	    barrier(CLK_LOCAL_MEM_FENCE);

	    for (int p = 128; p >= 1; p = p >> 1) {
            if (d < p) {
               tmp1[d] += tmp1[d + p];
               tmp2[d] += tmp2[d + p];
            }			
		    barrier(CLK_LOCAL_MEM_FENCE);			
	    }	
        if (d == 0) {
            sum += tmp1[0];
            sum_sq += tmp2[0];	
        }	
	    barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (d == 0) {
        statistic[0] = sum / embed_dim;
        statistic[1] = sum_sq / embed_dim - (statistic[0] * statistic[0]);
        statistic[2] = 1.0f / sqrt(statistic[1] + 1e-6f);
    }
	barrier(CLK_LOCAL_MEM_FENCE);	

    float mean = statistic[0];
    float var = statistic[1];
    float inv_std = statistic[2];

    for(int block = 0; block < 3; block++){
        int idx_base = t * embed_dim + block*256 + d;
        float v = src[idx_base];
        dst[idx_base] = (v - mean) * inv_std * weight[d + block*256] + bias[d + block*256];
    }
}
__kernel void residual(
    __global float* src,
    __global float* add, 
    __global float* dst,
    int embed_dim) {
    int t = get_global_id(0);
    int d = get_global_id(1);
    int idx = t * embed_dim + d;

    dst[idx] = src[idx] + add[idx];
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

__kernel void softmax_kernel(
    __global const float* logits,  // [batch * classes]
    __global float* probs,         // [batch * classes]
    const int classes)
{
    int batch_id = get_global_id(0);
    int row_start = batch_id * classes;

    // 1) max(logits[row])
    float m = logits[row_start];
    for (int c = 1; c < classes; ++c) {
        float v = logits[row_start + c];
        if (v > m) m = v;
    }

    // 2) exp(logit - m) 및 합
    float s = 0.0f;
    for (int c = 0; c < classes; ++c) {
        float e = exp(logits[row_start + c] - m);
        probs[row_start + c] = e;
        s += e;
    }

    // 3) 정규화
    for (int c = 0; c < classes; ++c) {
        probs[row_start + c] /= s;
    }
}
