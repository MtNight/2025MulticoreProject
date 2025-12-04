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
    int oc = get_global_id(0);
    int p  = get_global_id(1);

    int oh = p / output_size;
    int ow = p % output_size;

    dst[oc * output_size * output_size + 
        oh * output_size + 
        ow] 
    = 
    src[oc * output_size * output_size + 
        p]
    + 
    bias[oc];
}