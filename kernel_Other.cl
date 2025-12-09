
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
__kernel void softmax_kernel(
    __global const float* logits, // [batch * classes]
    __global float* probs, // [batch * classes]
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