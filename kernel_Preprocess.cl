__kernel void im2col(	
	__global float* src,
	__global float* dst,
	int img_size,
    int patch_size,
    int output_size) {

	int k = get_global_id(0);
    int p = get_global_id(1);

    int ic = k / (patch_size * patch_size);
    int rest = k % (patch_size * patch_size);
    int kh = rest / patch_size;
    int kw = rest % patch_size;

    int oh = p / output_size;
    int ow = p % output_size;

    int ih = oh * patch_size + kh;
    int iw = ow * patch_size + kw;

    dst[k * (output_size * output_size) + 
        p] = 
    src[ic * img_size * img_size + 
        ih * img_size + 
        iw];
}
__kernel void conv_add_bias(
    __global float* src,
    __global float* bias,
    __global float* dst,
    int embed_dim,
    int output_size) {
    int oh = get_global_id(0);
    int ow = get_global_id(1);
    int oc = get_global_id(2);
    
    int patch_idx = oh * output_size + ow;

    int input_idx = oc * (output_size * output_size) + patch_idx;
    int output_idx = embed_dim + patch_idx * embed_dim + oc;

    dst[output_idx] = src[input_idx] + bias[oc];
}
__kernel void add_classToken_posEmbed(
    __global float* src,
    __global float* token, 
    __global float* posEmbed,
    int embed_dim) {
    int i = get_global_id(0);

    if (i < embed_dim) src[i] = token[i];
    
    src[i] += posEmbed[i];
}