#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "Network.h"
#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define TILE 16
#define NUM_IMAGE 100

// OpenCL 설정 관련 변수들
cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue writeQueue, excuteQueue;
cl_event imageWriteEvent[NUM_IMAGE];
#define NUM_KERNEL_FILES 4
size_t kernel_source_size[NUM_KERNEL_FILES];
char* sources[NUM_KERNEL_FILES];
cl_program program;

// 이미지, 가중치 인덱싱용 전역변수들
int image_count = 0;
int encoder_count = 0;

// 자주 쓰는 상수(주소 필요)
int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
int embed_dim = 768;
int hidden_dim = 3072;

// Preprocessing에 필요한 커널
cl_kernel kernel_im2col;
cl_kernel kernel_convAddBias;
cl_kernel kernel_addClassToken_posEmbeding;
// Preprocessing에 필요한 버퍼
cl_mem convInputBuffer[NUM_IMAGE], convArrangedInputBuffer;
cl_mem convTmpBuffer, preprocessingOutputBuffer;
cl_mem convWeightBuffer, convBiasBuffer;
cl_mem classTokenBuffer, posEmbedBuffer;

// Layer Normalize에 필요한 커널
cl_kernel kernel_layer_normalize;
// Layer Normalize에 필요한 버퍼
cl_mem layerNormalizeWeightBuffer[depth][2];
cl_mem layerNormalizeBiasBuffer[depth][2];
cl_mem finalLayerNormalizeWeightBuffer;
cl_mem finalLayerNormalizeBiasBuffer;
cl_mem layerNormalizeOutputBuffer;

// Residual에 필요한 커널
cl_kernel kernel_residual;
// Residual에 필요한 버퍼
cl_mem residualOutputBuffer;

// MHA에 필요한 커널
cl_kernel kernel_transpose;
cl_kernel kernel_gemm;
cl_kernel kernel_mhaAddBias;
cl_kernel kernel_divide;
cl_kernel kernel_scale;
cl_kernel kernel_softmax;
cl_kernel kernel_concat;
// MHA에 필요한 버퍼
cl_mem qkvBuffer[3], mhaInWeightBuffer[depth][3], transposedInWeightBuffer[3], mhaInBiasBuffer[depth][3], dividedQkvBuffer[3];
cl_mem transposedKeyBuffer, scoreBuffer, headOutputBuffer, attnOutputBuffer;
cl_mem mhaOutWeightBuffer[depth], transposedOutWeightBuffer, mhaOutBiasBuffer[depth];
cl_mem mhaOutputBuffer;

//mlp
cl_kernel linear_kernel;
cl_kernel gelukernel;
cl_mem weight1Buffer[depth];
cl_mem weight2Buffer[depth];
cl_mem bias1Buffer[depth];
cl_mem bias2Buffer[depth];
cl_mem fc1_outBuffer;
cl_mem finalTmpOutputBuffer;
cl_mem outputBuffer_mlp;
cl_mem finalLinearWeightBuffer[1];
cl_mem finalLinearBiasBuffer[1];

//softmax
cl_kernel softmax_kernel;
cl_mem logits_buf;
cl_mem probs_buf;

// 실행 시간 체크용
clock_t startTime, endTime, ecnTime, mhaTime, mlpTime;

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        

        log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

////////////////////////////////////// ViT function //////////////////////////////////////

void ImagePreprocessing(cl_mem* output) {
    int imgSize = img_size;                     //224
    int patchSize = patch_size;                 //16
    int output_size = img_size / patch_size;    //14
    int M = embed_dim;                          //768
    int K = in_chans * patch_size * patch_size; //768
    int N = output_size * output_size;          //196

    // Conv2D에 필요한 글로벌 메모리 사이즈
    size_t img2col_size[2] = { K, N };
    size_t gemm_size[2] = { N, M };
    size_t add_bias_size[3] = { output_size, output_size, M };

    // 원본 이미지를 행렬에 맞게 배열
    clSetKernelArg(kernel_im2col, 0, sizeof(cl_mem), &convInputBuffer[image_count]);
    clSetKernelArg(kernel_im2col, 1, sizeof(cl_mem), &convArrangedInputBuffer);
    clSetKernelArg(kernel_im2col, 2, sizeof(int), &imgSize);
    clSetKernelArg(kernel_im2col, 3, sizeof(int), &patchSize);
    clSetKernelArg(kernel_im2col, 4, sizeof(int), &output_size);
    clEnqueueNDRangeKernel(excuteQueue, kernel_im2col, 2, NULL, img2col_size, NULL, 1, &imageWriteEvent[image_count], NULL);

    // 이미지*가중치 행렬곱을 활용해서 컨볼루션 곱 실행
    clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &convWeightBuffer);
    clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &convArrangedInputBuffer);
    clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &convTmpBuffer);
    clSetKernelArg(kernel_gemm, 3, sizeof(int), &M);
    clSetKernelArg(kernel_gemm, 4, sizeof(int), &N);
    clSetKernelArg(kernel_gemm, 5, sizeof(int), &K);
    clEnqueueNDRangeKernel(excuteQueue, kernel_gemm, 2, NULL, gemm_size, NULL, 0, NULL, NULL);

    // bias 합 및 인덱스 정리
    clSetKernelArg(kernel_convAddBias, 0, sizeof(cl_mem), &convTmpBuffer);
    clSetKernelArg(kernel_convAddBias, 1, sizeof(cl_mem), &convBiasBuffer);
    clSetKernelArg(kernel_convAddBias, 2, sizeof(cl_mem), &preprocessingOutputBuffer);
    clSetKernelArg(kernel_convAddBias, 3, sizeof(int), &M);
    clSetKernelArg(kernel_convAddBias, 4, sizeof(int), &output_size);
    clEnqueueNDRangeKernel(excuteQueue, kernel_convAddBias, 3, NULL, add_bias_size, NULL, 0, NULL, NULL);

    // 클래스 토큰 붙이기, 포지션 임베딩
    clSetKernelArg(kernel_addClassToken_posEmbeding, 0, sizeof(cl_mem), &preprocessingOutputBuffer);
    clSetKernelArg(kernel_addClassToken_posEmbeding, 1, sizeof(cl_mem), &classTokenBuffer);
    clSetKernelArg(kernel_addClassToken_posEmbeding, 2, sizeof(cl_mem), &posEmbedBuffer);
    clSetKernelArg(kernel_addClassToken_posEmbeding, 3, sizeof(int), &M);
    size_t ctNpe_size = embed_dim * tokens;
    clEnqueueNDRangeKernel(excuteQueue, kernel_addClassToken_posEmbeding, 1, NULL, &ctNpe_size, NULL, 0, NULL, NULL);
    
    *output = preprocessingOutputBuffer;
}

void layer_norm(cl_mem input, cl_mem* output, int lnIndex) {
    int dvdEd = embed_dim / 3;
    cl_mem wBuffer, bBuffer;

    if (lnIndex != 3) {
        wBuffer = layerNormalizeWeightBuffer[encoder_count][lnIndex];
        bBuffer = layerNormalizeBiasBuffer[encoder_count][lnIndex];
    }
    else {
        wBuffer = finalLayerNormalizeWeightBuffer;
        bBuffer = finalLayerNormalizeBiasBuffer;
    }

    clSetKernelArg(kernel_layer_normalize, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel_layer_normalize, 1, sizeof(cl_mem), &wBuffer);
    clSetKernelArg(kernel_layer_normalize, 2, sizeof(cl_mem), &bBuffer);
    clSetKernelArg(kernel_layer_normalize, 3, sizeof(cl_mem), &layerNormalizeOutputBuffer);
    clSetKernelArg(kernel_layer_normalize, 4, sizeof(float) * dvdEd, NULL);
    clSetKernelArg(kernel_layer_normalize, 5, sizeof(float) * dvdEd, NULL);
    clSetKernelArg(kernel_layer_normalize, 6, sizeof(float) * 3, NULL);
    clSetKernelArg(kernel_layer_normalize, 7, sizeof(int), &tokens);
    clSetKernelArg(kernel_layer_normalize, 8, sizeof(int), &embed_dim);
    size_t norm_size[2] = { tokens , dvdEd };
    size_t norm_local_size[2] = { 1 , dvdEd };
    clEnqueueNDRangeKernel(excuteQueue, kernel_layer_normalize, 2, NULL, &norm_size, &norm_local_size, 0, NULL, NULL);
    
    *output = layerNormalizeOutputBuffer;
}
void residual(cl_mem input, cl_mem add, cl_mem* output) {

    clSetKernelArg(kernel_residual, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel_residual, 1, sizeof(cl_mem), &add);
    clSetKernelArg(kernel_residual, 2, sizeof(cl_mem), &residualOutputBuffer);
    clSetKernelArg(kernel_residual, 3, sizeof(int), &embed_dim);
    size_t resi_size[2] = { tokens , embed_dim };
    clEnqueueNDRangeKernel(excuteQueue, kernel_residual, 2, NULL, &resi_size, NULL, 0, NULL, NULL);

    *output = residualOutputBuffer;
}

void multihead_attn(cl_mem input, cl_mem* output) {

    // MHA 계산에 필요한 변수들
    int head_dim = embed_dim / num_heads;
    float hd_float = head_dim;  // float head_dim for scaling
    int padding = 256;  // padded tokens for softmax

    // 계산 사이즈 정의
    size_t weight_traspose_size[2] = { embed_dim, embed_dim };
    size_t weight_size[2] = { embed_dim, tokens };
    //size_t weight_size_padded[2] = { 768, 224 };  //embed_dim, tokens_padded 
    size_t head_size[2] = { tokens, head_dim };
    size_t rev_head_size[2] = { head_dim, tokens };
    size_t score_size[2] = { tokens, tokens };
    //size_t score_size_padded[2] = { 224 , 224 };
    size_t softmax_local_size[2] = { 1, tokens };
    //size_t gemm_local_size[2] = { TILE_SIZE, TILE_SIZE };     //tiling?

    for (int i = 0; i < 3; i++) {
        // in_weight 전치 - gemm 계산을 위한 전처리
        clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mhaInWeightBuffer[encoder_count][i]);
        clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        clSetKernelArg(kernel_transpose, 2, sizeof(int), &embed_dim);
        clSetKernelArg(kernel_transpose, 3, sizeof(int), &embed_dim);
        clEnqueueNDRangeKernel(excuteQueue, kernel_transpose, 2, NULL, weight_traspose_size, NULL, 0, NULL, NULL);
    }

    // Set QKV (Excute gemm)
    clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
    clSetKernelArg(kernel_gemm, 4, sizeof(int), &embed_dim);
    clSetKernelArg(kernel_gemm, 5, sizeof(int), &embed_dim);
    for (int i = 0; i < 3; i++) {
        clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &qkvBuffer[i]);
        clEnqueueNDRangeKernel(excuteQueue, kernel_gemm, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    }

    // Excute add_bias
    clSetKernelArg(kernel_mhaAddBias, 2, sizeof(int), &embed_dim);
    for (int i = 0; i < 3; i++) {
        clSetKernelArg(kernel_mhaAddBias, 0, sizeof(cl_mem), &qkvBuffer[i]);
        clSetKernelArg(kernel_mhaAddBias, 1, sizeof(cl_mem), &mhaInBiasBuffer[encoder_count][i]);
        clEnqueueNDRangeKernel(excuteQueue, kernel_mhaAddBias, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    }

    /*head별로 attn 수행*/
    for (int h = 0; h < num_heads; h++) {
        // 헤드 분리
        for (int i = 0; i < 3; i++) {
            clSetKernelArg(kernel_divide, 0, sizeof(cl_mem), &qkvBuffer[i]);
            clSetKernelArg(kernel_divide, 1, sizeof(cl_mem), &dividedQkvBuffer[i]);
            clSetKernelArg(kernel_divide, 2, sizeof(int), &embed_dim);
            clSetKernelArg(kernel_divide, 3, sizeof(int), &head_dim);
            clSetKernelArg(kernel_divide, 4, sizeof(int), &h);
            clEnqueueNDRangeKernel(excuteQueue, kernel_divide, 2, NULL, head_size, NULL, 0, NULL, NULL);
        }
        // key 전치
        clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &dividedQkvBuffer[1]);
        clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedKeyBuffer);
        clSetKernelArg(kernel_transpose, 2, sizeof(int), &tokens);
        clSetKernelArg(kernel_transpose, 3, sizeof(int), &head_dim);
        clEnqueueNDRangeKernel(excuteQueue, kernel_transpose, 2, NULL, head_size, NULL, 0, NULL, NULL);

        // score(QxKt) 행렬 생성
        clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &dividedQkvBuffer[0]);
        clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedKeyBuffer);
        clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &scoreBuffer);
        clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
        clSetKernelArg(kernel_gemm, 4, sizeof(int), &tokens);
        clSetKernelArg(kernel_gemm, 5, sizeof(int), &head_dim);
        clEnqueueNDRangeKernel(excuteQueue, kernel_gemm, 2, NULL, score_size, NULL, 0, NULL, NULL);

        // 스코어 스케일링
        clSetKernelArg(kernel_scale, 0, sizeof(cl_mem), &scoreBuffer);
        clSetKernelArg(kernel_scale, 1, sizeof(float), &hd_float);
        clSetKernelArg(kernel_scale, 2, sizeof(int), &tokens);
        clEnqueueNDRangeKernel(excuteQueue, kernel_scale, 2, NULL, score_size, NULL, 0, NULL, NULL);

        // softmax 적용
        clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &scoreBuffer);
        clSetKernelArg(kernel_softmax, 1, sizeof(float) * tokens, NULL);
        clSetKernelArg(kernel_softmax, 2, sizeof(int), &padding);
        clSetKernelArg(kernel_softmax, 3, sizeof(int), &tokens);
        clEnqueueNDRangeKernel(excuteQueue, kernel_softmax, 2, NULL, score_size, softmax_local_size, 0, NULL, NULL);

        // Score*V 계산
        clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &scoreBuffer);
        clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &dividedQkvBuffer[2]);
        clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &headOutputBuffer);
        clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
        clSetKernelArg(kernel_gemm, 4, sizeof(int), &head_dim);
        clSetKernelArg(kernel_gemm, 5, sizeof(int), &tokens);
        clEnqueueNDRangeKernel(excuteQueue, kernel_gemm, 2, NULL, rev_head_size, NULL, 0, NULL, NULL);

        // head 계산 결과 concat
        clSetKernelArg(kernel_concat, 0, sizeof(cl_mem), &attnOutputBuffer);
        clSetKernelArg(kernel_concat, 1, sizeof(cl_mem), &headOutputBuffer);
        clSetKernelArg(kernel_concat, 2, sizeof(int), &embed_dim);
        int headOffset = h * head_dim;
        clSetKernelArg(kernel_concat, 3, sizeof(int), &headOffset);
        clSetKernelArg(kernel_concat, 4, sizeof(int), &head_dim);
        clEnqueueNDRangeKernel(excuteQueue, kernel_concat, 2, NULL, head_size, NULL, 0, NULL, NULL);
    }

    // out_weight 전치 - gemm 계산을 위한 전처리
    clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mhaOutWeightBuffer[encoder_count]);
    clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedOutWeightBuffer);
    clSetKernelArg(kernel_transpose, 2, sizeof(int), &embed_dim);
    clSetKernelArg(kernel_transpose, 3, sizeof(int), &embed_dim);
    clEnqueueNDRangeKernel(excuteQueue, kernel_transpose, 2, NULL, weight_traspose_size, NULL, 0, NULL, NULL);

    // out_weight 적용
    clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &attnOutputBuffer);
    clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedOutWeightBuffer);
    clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &mhaOutputBuffer);
    clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
    clSetKernelArg(kernel_gemm, 4, sizeof(int), &embed_dim);
    clSetKernelArg(kernel_gemm, 5, sizeof(int), &embed_dim);
    clEnqueueNDRangeKernel(excuteQueue, kernel_gemm, 2, NULL, weight_size, NULL, 0, NULL, NULL);

    // out_bias 값 더하기
    clSetKernelArg(kernel_mhaAddBias, 0, sizeof(cl_mem), &mhaOutputBuffer);
    clSetKernelArg(kernel_mhaAddBias, 1, sizeof(cl_mem), &mhaOutBiasBuffer[encoder_count]);
    clSetKernelArg(kernel_mhaAddBias, 2, sizeof(int), &embed_dim);
    clEnqueueNDRangeKernel(excuteQueue, kernel_mhaAddBias, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    
    *output = mhaOutputBuffer;
}
void linear_layer_opencl(
    cl_mem input, cl_mem* output,           // CPU output
    int num_used_token,
    int in_features,
    int out_features,
    int isFinalLinear,
    cl_mem* weightBuffer_ll,
    cl_mem* biasBuffer_ll,
    cl_mem outputBuffer_ll
) {
    cl_int err;

    int input_size = num_used_token * in_features;
    int weight_size = in_features * out_features;
    int bias_size = out_features;
    int output_size = num_used_token * out_features;

    // 1) Check isFinalLinear
    cl_mem weightBuffer;
    cl_mem biasBuffer;
    if (isFinalLinear == 1) {
        weightBuffer = finalLinearWeightBuffer[0];
        biasBuffer = finalLinearBiasBuffer[0];
    }
    else {
        weightBuffer = weightBuffer_ll[encoder_count];
        biasBuffer = biasBuffer_ll[encoder_count];
    }

    // 2) Set kernel args
    clSetKernelArg(linear_kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(linear_kernel, 1, sizeof(cl_mem), &weightBuffer);
    clSetKernelArg(linear_kernel, 2, sizeof(cl_mem), &biasBuffer);
    clSetKernelArg(linear_kernel, 3, sizeof(cl_mem), &outputBuffer_ll);
    clSetKernelArg(linear_kernel, 4, sizeof(int), &num_used_token);
    clSetKernelArg(linear_kernel, 5, sizeof(int), &in_features);
    clSetKernelArg(linear_kernel, 6, sizeof(int), &out_features);

    // 3) Kernel launch
    size_t local[2] = { TILE, TILE };
    size_t global[2] = {
        ((size_t)((num_used_token + TILE - 1) / TILE)) * TILE,
        ((size_t)((out_features + TILE - 1) / TILE)) * TILE
    };
    clEnqueueNDRangeKernel(excuteQueue, linear_kernel, 2, NULL, global, local, 0, NULL, NULL);

    // 4) Read output
    *output = outputBuffer_ll;
}

void mlp_block_opencl(cl_mem input, cl_mem* output) {

    cl_mem fc1_gpu;
    cl_mem fc2_gpu;

    fc1_gpu = fc1_outBuffer;
    fc2_gpu = outputBuffer_mlp;

    linear_layer_opencl(
        input,                   // CPU input
        &fc1_gpu,                 // CPU output buffer
        tokens,
        embed_dim,               // in
        hidden_dim,              // out
        0,
        weight1Buffer,
        bias1Buffer,
        fc1_outBuffer
    );

    //GELU활성화
    int N = tokens * hidden_dim;
    size_t local = 256;
    size_t global = ((size_t)N + local - 1) / local * local;
    clSetKernelArg(gelukernel, 0, sizeof(cl_mem), &fc1_gpu); 
    clSetKernelArg(gelukernel, 1, sizeof(int), &N); 
    clEnqueueNDRangeKernel(excuteQueue, gelukernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    linear_layer_opencl(
        fc1_gpu,
        &fc2_gpu,
        tokens,
        hidden_dim,             // in
        embed_dim,              // out
        0,
        weight2Buffer,
        bias2Buffer,
        outputBuffer_mlp
    );

    *output = fc2_gpu;
}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////

void Encoder(cl_mem input, cl_mem* output) {
    //clock_t s = clock();
    cl_mem ln1_gpu, attn_gpu, resi_gpu, ln2_gpu, mlp_gpu;

    /*LN1*/
    layer_norm(input, &ln1_gpu, 0);

    /*Attn*/
    //startTime = clock();
    multihead_attn(ln1_gpu, &attn_gpu);
    //endTime = clock();
    //mhaTime += endTime - startTime;

    /*Residual1*/
    residual(input, attn_gpu, &resi_gpu);

    /*LN2*/
    layer_norm(resi_gpu, &ln2_gpu, 1);

    /*MLP*/
    //startTime = clock();
    mlp_block_opencl(ln2_gpu, &mlp_gpu);
    //endTime = clock();
    //mlpTime += endTime - startTime;

    /*Residual2*/
    residual(resi_gpu, mlp_gpu, output);
    //clock_t e = clock();
    //ecnTime += (e - s);

    encoder_count++;
}

void Softmax_opencl(cl_mem logits, float* probs, int batch, int classes)
{
    // 1) 커널 인자 설정
    clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &logits);
    clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &probs_buf);
    clSetKernelArg(softmax_kernel, 2, sizeof(int), &classes);
    
    // 2) 커널 실행
    size_t gws[1] = { (size_t)batch };
    size_t lws[1] = { 1 };
    clEnqueueNDRangeKernel(excuteQueue, softmax_kernel, 1, NULL, gws, lws, 0, NULL, NULL);
    
    // 3) 결과 읽기
    clEnqueueReadBuffer(excuteQueue, probs_buf, CL_TRUE, 0, sizeof(float) * batch * classes, probs, 0, NULL, NULL);
}

void InitOpenCLElements(ImageData* image, Network* networks) {

    // 기기 설정
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    writeQueue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    excuteQueue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    sources[0] = get_source_code("kernel_Preprocess.cl", &kernel_source_size[0]);
    sources[1] = get_source_code("kernel_MHA.cl", &kernel_source_size[1]);
    sources[2] = get_source_code("kernel_MLP.cl", &kernel_source_size[2]);
    sources[3] = get_source_code("kernel_Other.cl", &kernel_source_size[3]);
    program = clCreateProgramWithSource(context, NUM_KERNEL_FILES, (const char**)sources, &kernel_source_size, &err);
    clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    
    // preprocessing 커널
    kernel_im2col = clCreateKernel(program, "im2col", &err);
    kernel_convAddBias = clCreateKernel(program, "conv_add_bias", &err);
    kernel_addClassToken_posEmbeding = clCreateKernel(program, "add_classToken_posEmbed", &err);
    
    // preprocessing 버퍼
    int output_size = img_size / patch_size;
    for (int i = 0; i < NUM_IMAGE; i++) {
        convInputBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * image->c * image->w * image->h, NULL, &err);
    }
    convArrangedInputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * image->c * image->w * image->h, NULL, &err);
    convTmpBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * output_size * output_size, NULL, &err);
    preprocessingOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * tokens, NULL, &err);
    convWeightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * in_chans * patch_size * patch_size, NULL, &err);
    convBiasBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
    classTokenBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    posEmbedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * tokens, NULL, &err);
    
    // preprocessing 가중치 전송
    clEnqueueWriteBuffer(writeQueue, convWeightBuffer, CL_TRUE, 0, sizeof(float) * embed_dim * in_chans * patch_size * patch_size, networks[1].data, 0, NULL, NULL);
    clEnqueueWriteBuffer(writeQueue, convBiasBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[2].data, 0, NULL, NULL);    
    clEnqueueWriteBuffer(writeQueue, classTokenBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[0].data, 0, NULL, NULL);    
    clEnqueueWriteBuffer(writeQueue, posEmbedBuffer, CL_TRUE, 0, sizeof(float) * embed_dim * (output_size * output_size + 1), networks[3].data, 0, NULL, NULL);    

    // Layer Normalize 커널
    kernel_layer_normalize = clCreateKernel(program, "layer_normalize", &err);    
    // Layer Normalize 버퍼
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < 2; j++) {
            layerNormalizeWeightBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);            
            layerNormalizeBiasBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);            
        }
    }
    finalLayerNormalizeWeightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);    
    finalLayerNormalizeBiasBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);    
    layerNormalizeOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);    

    // Layer Normalize 가중치 전송
    for (int i = 0; i < depth; i++) {
        int idx = 4 + i * 12;
        clEnqueueWriteBuffer(writeQueue, layerNormalizeWeightBuffer[i][0], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, layerNormalizeBiasBuffer[i][0], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 1].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, layerNormalizeWeightBuffer[i][1], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 6].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, layerNormalizeBiasBuffer[i][1], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 7].data, 0, NULL, NULL);        
    }
    clEnqueueWriteBuffer(writeQueue, finalLayerNormalizeWeightBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[148].data, 0, NULL, NULL);    
    clEnqueueWriteBuffer(writeQueue, finalLayerNormalizeBiasBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[149].data, 0, NULL, NULL);    

    // Residual 커널
    kernel_residual = clCreateKernel(program, "residual", &err);    
    // Residual 버퍼
    residualOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);    

    // MHA 커널
    kernel_transpose = clCreateKernel(program, "transpose", &err);    
    kernel_gemm = clCreateKernel(program, "gemm", &err);    
    kernel_mhaAddBias = clCreateKernel(program, "add_bias", &err);    
    kernel_divide = clCreateKernel(program, "divide_head", &err);    
    kernel_scale = clCreateKernel(program, "scale_score", &err);    
    kernel_softmax = clCreateKernel(program, "softmax_score", &err);    
    kernel_concat = clCreateKernel(program, "copy_head_output", &err);    

    // MHA 버퍼
    int head_dim = embed_dim / num_heads;
    for (int i = 0; i < 3; i++) {
        qkvBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);        
        transposedInWeightBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);        
        dividedQkvBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);        
    }
    transposedKeyBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);    
    scoreBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * tokens, NULL, &err);    
    headOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);    
    attnOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);    
    transposedOutWeightBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);    
    mhaOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);    
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < in_chans; j++) {
            mhaInWeightBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);            
            mhaInBiasBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);            
        }
        mhaOutWeightBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);        
        mhaOutBiasBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);        
    }
    // in_weight, in_bias, out_weight, out_bias 전송
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < in_chans; j++) {
            clEnqueueWriteBuffer(writeQueue, mhaInWeightBuffer[i][j], CL_TRUE, 0, sizeof(float) * embed_dim * embed_dim, (networks[6 + i * depth].data + j * embed_dim * embed_dim), 0, NULL, NULL);            
            clEnqueueWriteBuffer(writeQueue, mhaInBiasBuffer[i][j], CL_TRUE, 0, sizeof(float) * embed_dim, (networks[7 + i * depth].data + j * embed_dim), 0, NULL, NULL);            
        }
        clEnqueueWriteBuffer(writeQueue, mhaOutWeightBuffer[i], CL_TRUE, 0, sizeof(float) * embed_dim * embed_dim, networks[8 + i * depth].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, mhaOutBiasBuffer[i], CL_TRUE, 0, sizeof(float) * embed_dim, networks[9 + i * depth].data, 0, NULL, NULL);        
    }

    // MLP 커널
    int Hidden_dim = hidden_dim;
    linear_kernel = clCreateKernel(program, "linear_kernel", &err);    
    gelukernel = clCreateKernel(program, "gelu_kernel_inplace", &err);   
    // MLP(Linear) 버퍼
    size_t max_in = (embed_dim > Hidden_dim) ? embed_dim : Hidden_dim;
    fc1_outBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * Hidden_dim, NULL, &err);    
    outputBuffer_mlp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);    
    finalTmpOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_classes, NULL, &err);    
    for (int i = 0; i < depth; i++) {
        int idx = 12 + i * 12;
        weight1Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * Hidden_dim * embed_dim, NULL, &err);        
        bias1Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * Hidden_dim, NULL, &err);        
        weight2Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * Hidden_dim, NULL, &err);        
        bias2Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);        
    }
    finalLinearWeightBuffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_classes * embed_dim, NULL, &err);    
    finalLinearBiasBuffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_classes, NULL, &err);
    
    // Linear 가중치 전송
    for (int i = 0; i < depth; i++) {
        int idx = 12 + i * 12;
        clEnqueueWriteBuffer(writeQueue, weight1Buffer[i], CL_TRUE, 0, sizeof(float) * Hidden_dim * embed_dim, networks[idx].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, bias1Buffer[i], CL_TRUE, 0, sizeof(float) * Hidden_dim, networks[idx + 1].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, weight2Buffer[i], CL_TRUE, 0, sizeof(float) * embed_dim * Hidden_dim, networks[idx + 2].data, 0, NULL, NULL);        
        clEnqueueWriteBuffer(writeQueue, bias2Buffer[i], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 3].data, 0, NULL, NULL);        
    }
    clEnqueueWriteBuffer(writeQueue, finalLinearWeightBuffer[0], CL_TRUE, 0, sizeof(float) * num_classes * embed_dim, networks[150].data, 0, NULL, NULL);    
    clEnqueueWriteBuffer(writeQueue, finalLinearBiasBuffer[0], CL_TRUE, 0, sizeof(float) * num_classes, networks[151].data, 0, NULL, NULL);    

    // Softmax 커널
    softmax_kernel = clCreateKernel(program, "softmax_kernel", &err);    
    // Softmax 버퍼
    int max_batch = 1;
    int max_classes = num_classes;
    logits_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * max_batch * max_classes, NULL, &err);    
    probs_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * max_batch * max_classes, NULL, &err);    

    clFinish(writeQueue);
    // 이미지 인풋 전송
    for (int i = 0; i < NUM_IMAGE; i++) {
        clEnqueueWriteBuffer(writeQueue, convInputBuffer[i], CL_FALSE, 0, sizeof(float) * image->c * image->w * image->h, image[i].data, 0, NULL, &imageWriteEvent[i]);        
        clFlush(writeQueue);
    }
}
void ReleaseOpenCLElements() {
    // preprocessing 커널
    clReleaseKernel(kernel_im2col);
    clReleaseKernel(kernel_convAddBias);
    clReleaseKernel(kernel_addClassToken_posEmbeding);
    // preprocessing 버퍼
    int output_size = img_size / patch_size;
    for (int i = 0; i < NUM_IMAGE; i++) {
        clReleaseMemObject(convInputBuffer[i]);
    }
    clReleaseMemObject(convArrangedInputBuffer);
    clReleaseMemObject(convTmpBuffer);
    clReleaseMemObject(preprocessingOutputBuffer);
    clReleaseMemObject(convWeightBuffer);
    clReleaseMemObject(convBiasBuffer);
    clReleaseMemObject(classTokenBuffer);
    clReleaseMemObject(posEmbedBuffer);

    // Layer Normalize 커널
    clReleaseKernel(kernel_layer_normalize);
    // Layer Normalize 버퍼
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < 2; j++) {
            clReleaseMemObject(layerNormalizeWeightBuffer[i][j]);
            clReleaseMemObject(layerNormalizeBiasBuffer[i][j]);
        }
    }
    clReleaseMemObject(finalLayerNormalizeWeightBuffer);
    clReleaseMemObject(finalLayerNormalizeBiasBuffer);
    clReleaseMemObject(layerNormalizeOutputBuffer);

    // MHA 커널
    clReleaseKernel(kernel_divide);
    clReleaseKernel(kernel_transpose);
    clReleaseKernel(kernel_gemm);
    clReleaseKernel(kernel_mhaAddBias);
    // MHA 버퍼
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < in_chans; j++) {
            clReleaseMemObject(mhaInWeightBuffer[i][j]);
            clReleaseMemObject(mhaInBiasBuffer[i][j]);
        }
        clReleaseMemObject(mhaOutWeightBuffer[i]);
        clReleaseMemObject(mhaOutBiasBuffer[i]);
    }
    for (int i = 0; i < 3; i++) {
        clReleaseMemObject(qkvBuffer[i]);
        clReleaseMemObject(transposedInWeightBuffer[i]);
        clReleaseMemObject(dividedQkvBuffer[i]);
    }
    clReleaseMemObject(transposedKeyBuffer);
    clReleaseMemObject(scoreBuffer);
    clReleaseMemObject(headOutputBuffer);
    clReleaseMemObject(attnOutputBuffer);
    clReleaseMemObject(transposedOutWeightBuffer);
    clReleaseMemObject(mhaOutputBuffer);

    // MLP 커널
    clReleaseKernel(linear_kernel);
    clReleaseKernel(gelukernel);

    // MLP 버퍼
    clReleaseMemObject(fc1_outBuffer);
    clReleaseMemObject(outputBuffer_mlp);
    clReleaseMemObject(finalTmpOutputBuffer);
    for (int i = 0; i < depth; i++) {
        clReleaseMemObject(weight1Buffer[i]);
        clReleaseMemObject(weight2Buffer[i]);
        clReleaseMemObject(bias2Buffer[i]);
        clReleaseMemObject(bias1Buffer[i]);
    }
    clReleaseMemObject(finalLinearWeightBuffer[0]);
    clReleaseMemObject(finalLinearBiasBuffer[0]);

    // Softmax 커널
    clReleaseKernel(softmax_kernel);
    // Softmax 버퍼
    clReleaseMemObject(logits_buf);
    clReleaseMemObject(probs_buf);

    // 기기 설정 해제
    clReleaseProgram(program);
    clReleaseCommandQueue(excuteQueue);
    clReleaseContext(context);
    clReleaseDevice(device);
    for (int i = 0; i < NUM_KERNEL_FILES; i++) free(sources[i]);
}

////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_prl(ImageData* image, Network* networks, float** probabilities) {

    InitOpenCLElements(image, networks);

    cl_mem pre_layer;
    cl_mem enc_layer[12];
    cl_mem enc_output;
    cl_mem final_output;

    for (image_count = 0; image_count < image->n; image_count++) {

        //clock_t imgStartTime = clock();
        /*patch embedding*/
        //startTime = clock();
        ImagePreprocessing(&pre_layer);
        //endTime = clock();
        //printf("ImagePreprocessing: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

        /*Encoder - 12 Layers*/
        encoder_count = 0;
        //clock_t encStartTime = clock();
        mhaTime = 0;
        mlpTime = 0;
        ecnTime = 0;
        Encoder(pre_layer, &enc_layer[0]);
        Encoder(enc_layer[0], &enc_layer[1]);
        Encoder(enc_layer[1], &enc_layer[2]);
        Encoder(enc_layer[2], &enc_layer[3]);
        Encoder(enc_layer[3], &enc_layer[4]);
        Encoder(enc_layer[4], &enc_layer[5]);
        Encoder(enc_layer[5], &enc_layer[6]);
        Encoder(enc_layer[6], &enc_layer[7]);
        Encoder(enc_layer[7], &enc_layer[8]);
        Encoder(enc_layer[8], &enc_layer[9]);
        Encoder(enc_layer[9], &enc_layer[10]);
        Encoder(enc_layer[10], &enc_layer[11]);

        //endTime = clock();
        //printf("mha avg: %lf\n", (double)(mhaTime / 12.0f) / CLOCKS_PER_SEC);
        //printf("mlp avg: %lf\n", (double)(mlpTime / 12.0f) / CLOCKS_PER_SEC);
        //printf("Encoder avg: %lf\n", (double)(ecnTime / 12.0f) / CLOCKS_PER_SEC);
        //printf("Encoder 12: %lf\n", (double)(endTime - encStartTime) / CLOCKS_PER_SEC);

        layer_norm(enc_layer[11], &enc_output, 3);

        linear_layer_opencl(enc_output, &final_output, 1, embed_dim, num_classes, 1, finalLinearWeightBuffer, finalLinearBiasBuffer, finalTmpOutputBuffer);
        /* 확률분포 추출 */
        Softmax_opencl(final_output, probabilities[image_count], 1, num_classes);

        //endTime = clock();
        //printf("image %d: %lf\n\n", image_count, (double)(endTime - imgStartTime) / CLOCKS_PER_SEC);
    }

    ReleaseOpenCLElements();
}
