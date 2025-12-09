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
#define dropout 0.0         //?? 이거 포함 아래 3개는 미사용?
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6
#define TILE 16
#define NUM_IMAGE 1     //for test. 100개로 고쳐야 함

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

// OpenCL 설정 관련 변수들
cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES,
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    0 };
#define NUM_KERNEL_FILES 3
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
cl_mem layerNormalizeInputBuffer, layerNormalizeOutputBuffer;
cl_mem layerNormalizeWeightBuffer[depth][2];
cl_mem layerNormalizeBiasBuffer[depth][2];
cl_mem finalLayerNormalizeWeightBuffer;
cl_mem finalLayerNormalizeBiasBuffer;

// Residual에 필요한 커널
cl_kernel kernel_residual;
// Residual에 필요한 버퍼
cl_mem residualInputBuffer;
cl_mem residualAddBuffer;
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
cl_mem mhaInputBuffer, mhaOutputBuffer;
cl_mem qkvBuffer[3], mhaInWeightBuffer[depth][3], transposedInWeightBuffer[3], mhaInBiasBuffer[depth][3], dividedQkvBuffer[3];
cl_mem transposedKeyBuffer, scoreBuffer, headOutputBuffer, attnOutputBuffer;
cl_mem mhaOutWeightBuffer[depth], transposedOutWeightBuffer, mhaOutBiasBuffer[depth];

//mlp
cl_kernel linear_kernel;
cl_kernel gelukernel;
cl_mem inputBuffer_mlp;
cl_mem weight1Buffer[depth];
cl_mem weight2Buffer[depth];
cl_mem bias1Buffer[depth];
cl_mem bias2Buffer[depth];
cl_mem fc1_outBuffer;
cl_mem outputBuffer_mlp;
cl_mem finalLinearWeightBuffer;
cl_mem finalLinearBiasBuffer;

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

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

////////////////////////////////////// ViT function //////////////////////////////////////

void ImagePreprocessing(float* input, float* output) {
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
    err = clSetKernelArg(kernel_im2col, 0, sizeof(cl_mem), &convInputBuffer[image_count]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_im2col, 1, sizeof(cl_mem), &convArrangedInputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_im2col, 2, sizeof(int), &imgSize);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_im2col, 3, sizeof(int), &patchSize);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_im2col, 4, sizeof(int), &output_size);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_im2col, 2, NULL, img2col_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 이미지*가중치 행렬곱을 활용해서 컨볼루션 곱 실행
    err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &convWeightBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &convArrangedInputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &convTmpBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &K);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, gemm_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // bias 합 및 인덱스 정리
    err = clSetKernelArg(kernel_convAddBias, 0, sizeof(cl_mem), &convTmpBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convAddBias, 1, sizeof(cl_mem), &convBiasBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convAddBias, 2, sizeof(cl_mem), &preprocessingOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convAddBias, 3, sizeof(int), &M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convAddBias, 4, sizeof(int), &output_size);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_convAddBias, 3, NULL, add_bias_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 클래스 토큰 붙이기, 포지션 임베딩
    err = clSetKernelArg(kernel_addClassToken_posEmbeding, 0, sizeof(cl_mem), &preprocessingOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_addClassToken_posEmbeding, 1, sizeof(cl_mem), &classTokenBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_addClassToken_posEmbeding, 2, sizeof(cl_mem), &posEmbedBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_addClassToken_posEmbeding, 3, sizeof(int), &M);
    CHECK_ERROR(err);
    size_t ctNpe_size = embed_dim * tokens;
    err = clEnqueueNDRangeKernel(queue, kernel_addClassToken_posEmbeding, 1, NULL, &ctNpe_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 최종 계산 결과를 out에 읽어오기
    clFinish(queue);
    err = clEnqueueReadBuffer(queue, preprocessingOutputBuffer, CL_TRUE, 0, sizeof(float) * embed_dim * (output_size * output_size + 1), output, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void layer_norm(float* input, float* output, int lnIndex) {
    int dvdEd = embed_dim / 3;
    cl_mem iBuffer, wBuffer, bBuffer;

    err = clEnqueueWriteBuffer(queue, layerNormalizeInputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, input, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);
    iBuffer = layerNormalizeInputBuffer;

    if (lnIndex != 3) {
        wBuffer = layerNormalizeWeightBuffer[encoder_count][lnIndex];
        bBuffer = layerNormalizeBiasBuffer[encoder_count][lnIndex];
    }
    else {
        wBuffer = finalLayerNormalizeWeightBuffer;
        bBuffer = finalLayerNormalizeBiasBuffer;
    }

    err = clSetKernelArg(kernel_layer_normalize, 0, sizeof(cl_mem), &iBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 1, sizeof(cl_mem), &wBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 2, sizeof(cl_mem), &bBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 3, sizeof(cl_mem), &layerNormalizeOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 4, sizeof(float) * dvdEd, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 5, sizeof(float) * dvdEd, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 6, sizeof(float) * 3, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 7, sizeof(int), &tokens);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_layer_normalize, 8, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    size_t norm_size[2] = { tokens , dvdEd };
    size_t norm_local_size[2] = { 1 , dvdEd };
    err = clEnqueueNDRangeKernel(queue, kernel_layer_normalize, 2, NULL, &norm_size, &norm_local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);

    // 최종 계산 결과를 out에 읽어오기
    err = clEnqueueReadBuffer(queue, layerNormalizeOutputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, output, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);
}
void residual(float* input, float* add, float* output) {

    err = clEnqueueWriteBuffer(queue, residualInputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, input, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, residualAddBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, add, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);

    err = clSetKernelArg(kernel_residual, 0, sizeof(cl_mem), &residualInputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_residual, 1, sizeof(cl_mem), &residualAddBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_residual, 2, sizeof(cl_mem), &residualOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_residual, 3, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    size_t resi_size[2] = { tokens , embed_dim };
    err = clEnqueueNDRangeKernel(queue, kernel_residual, 2, NULL, &resi_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);

    // 최종 계산 결과를 out에 읽어오기
    err = clEnqueueReadBuffer(queue, residualOutputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, output, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);
}

void multihead_attn(float* input, float* output) {

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

    // input 전송
    err = clEnqueueWriteBuffer(queue, mhaInputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, input, 0, NULL, NULL);
    CHECK_ERROR(err);

    for (int i = 0; i < 3; i++) {
        // in_weight 전치 - gemm 계산을 위한 전처리
        err = clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mhaInWeightBuffer[encoder_count][i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 2, sizeof(int), &embed_dim);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 3, sizeof(int), &embed_dim);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_transpose, 2, NULL, weight_traspose_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // Set QKV (Excute gemm)
    err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &mhaInputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &embed_dim);
    CHECK_ERROR(err);

    for (int i = 0; i < 3; i++) {
        err = clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &qkvBuffer[i]);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, weight_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // Excute add_bias
    err = clSetKernelArg(kernel_mhaAddBias, 2, sizeof(int), &embed_dim);
    CHECK_ERROR(err);

    for (int i = 0; i < 3; i++) {
        err = clSetKernelArg(kernel_mhaAddBias, 0, sizeof(cl_mem), &qkvBuffer[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_mhaAddBias, 1, sizeof(cl_mem), &mhaInBiasBuffer[encoder_count][i]);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_mhaAddBias, 2, NULL, weight_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    /*head별로 attn 수행*/
    for (int h = 0; h < num_heads; h++) {
        // 헤드 분리
        for (int i = 0; i < 3; i++) {
            err = clSetKernelArg(kernel_divide, 0, sizeof(cl_mem), &qkvBuffer[i]);
            CHECK_ERROR(err);
            err = clSetKernelArg(kernel_divide, 1, sizeof(cl_mem), &dividedQkvBuffer[i]);
            CHECK_ERROR(err);
            err = clSetKernelArg(kernel_divide, 2, sizeof(int), &embed_dim);
            CHECK_ERROR(err);
            err = clSetKernelArg(kernel_divide, 3, sizeof(int), &head_dim);
            CHECK_ERROR(err);
            err = clSetKernelArg(kernel_divide, 4, sizeof(int), &h);
            CHECK_ERROR(err);

            err = clEnqueueNDRangeKernel(queue, kernel_divide, 2, NULL, head_size, NULL, 0, NULL, NULL);
            CHECK_ERROR(err);
        }
        // key 전치
        err = clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &dividedQkvBuffer[1]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedKeyBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 2, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 3, sizeof(int), &head_dim);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_transpose, 2, NULL, head_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // score(QxKt) 행렬 생성
        err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &dividedQkvBuffer[0]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedKeyBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &head_dim);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, score_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // 스코어 스케일링
        err = clSetKernelArg(kernel_scale, 0, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_scale, 1, sizeof(float), &hd_float);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_scale, 2, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_scale, 2, NULL, score_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // softmax 적용
        err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 1, sizeof(float) * tokens, NULL);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 2, sizeof(int), &padding);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 3, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_softmax, 2, NULL, score_size, softmax_local_size, 0, NULL, NULL);
        CHECK_ERROR(err);

        // Score*V 계산
        err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &dividedQkvBuffer[2]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &headOutputBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &head_dim);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &tokens);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, rev_head_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // head 계산 결과 concat
        err = clSetKernelArg(kernel_concat, 0, sizeof(cl_mem), &attnOutputBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_concat, 1, sizeof(cl_mem), &headOutputBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_concat, 2, sizeof(int), &embed_dim);
        CHECK_ERROR(err);
        int headOffset = h * head_dim;
        err = clSetKernelArg(kernel_concat, 3, sizeof(int), &headOffset);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_concat, 4, sizeof(int), &head_dim);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_concat, 2, NULL, head_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // out_weight 전치 - gemm 계산을 위한 전처리
    err = clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mhaOutWeightBuffer[encoder_count]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedOutWeightBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_transpose, 2, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_transpose, 3, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_transpose, 2, NULL, weight_traspose_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // out_weight 적용
    err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &attnOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedOutWeightBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &mhaOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // out_bias 값 더하기
    err = clSetKernelArg(kernel_mhaAddBias, 0, sizeof(cl_mem), &mhaOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_mhaAddBias, 1, sizeof(cl_mem), &mhaOutBiasBuffer[encoder_count]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_mhaAddBias, 2, sizeof(int), &embed_dim);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_mhaAddBias, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 최종 결과 읽어와서 output에 저장
    err = clEnqueueReadBuffer(queue, mhaOutputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, output, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void gelu_opencl() {
    int N = 197 * hidden_dim; // or tokens*out_features depending on 대상
    size_t local = 256;
    size_t global = ((size_t)N + local - 1) / local * local;

    err = clSetKernelArg(gelukernel, 0, sizeof(cl_mem), &fc1_outBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(gelukernel, 1, sizeof(int), &N);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, gelukernel, 1, NULL, &global, &local, 0, NULL, NULL);
    CHECK_ERROR(err);

}
void linear_layer_opencl(
    float* input,            // CPU input
    float* output,           // CPU output
    int num_used_token,
    int in_features,
    int out_features,
    int isFinalLinear,
    cl_mem inputBuffer_ll,
    cl_mem* weightBuffer_ll,
    cl_mem* biasBuffer_ll,
    cl_mem outputBuffer_ll
) {
    cl_int err;

    int input_size = num_used_token * in_features;
    int weight_size = in_features * out_features;
    int bias_size = out_features;
    int output_size = num_used_token * out_features;

    // 1) Upload input
    if (input != NULL) {
        err = clEnqueueWriteBuffer(queue, inputBuffer_ll, CL_TRUE, 0,
            sizeof(float) * input_size, input, 0, NULL, NULL);
    }
    cl_mem weightBuffer;
    cl_mem biasBuffer;
    if (isFinalLinear == 1) {
        weightBuffer = finalLinearWeightBuffer;
        biasBuffer = finalLinearBiasBuffer;
    }
    else {
        weightBuffer = weightBuffer_ll[encoder_count];
        biasBuffer = biasBuffer_ll[encoder_count];
    }

    // 2) Set kernel args
    clSetKernelArg(linear_kernel, 0, sizeof(cl_mem), &inputBuffer_ll);
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

    err = clEnqueueNDRangeKernel(queue, linear_kernel, 2, NULL, global, local, 0, NULL, NULL);

    // 4) Read output
    err = clEnqueueReadBuffer(queue, outputBuffer_ll, CL_TRUE, 0, sizeof(float) * output_size, output, 0, NULL, NULL);
}

void mlp_block_opencl(float* input, float* output) {

    float* fc1_cpu = (float*)malloc(sizeof(float) * tokens * hidden_dim);

    linear_layer_opencl(
        input,                   // CPU input
        fc1_cpu,                 // CPU output buffer
        tokens,
        embed_dim,               // in
        hidden_dim,              // out
        0,
        inputBuffer_mlp,
        weight1Buffer,
        bias1Buffer,
        fc1_outBuffer
    );


    //GELU활성화
    int N = tokens * hidden_dim;
    size_t local = 256;
    size_t global = ((size_t)N + local - 1) / local * local;
    err = clSetKernelArg(gelukernel, 0, sizeof(cl_mem), &fc1_outBuffer); CHECK_ERROR(err);
    err = clSetKernelArg(gelukernel, 1, sizeof(int), &N); CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, gelukernel, 1, NULL, &global, &local, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue, fc1_outBuffer, CL_TRUE, 0, sizeof(float) * tokens * hidden_dim, fc1_cpu, 0, NULL, NULL);
    CHECK_ERROR(err);

    linear_layer_opencl(
        fc1_cpu,
        output,
        tokens,
        hidden_dim,             // in
        embed_dim,              // out
        0,
        inputBuffer_mlp,
        weight2Buffer,
        bias2Buffer,
        outputBuffer_mlp
    );

    free(fc1_cpu);
}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////
void Encoder(float* input, float* output) {
    clock_t s = clock();

    float* ln1_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* attn_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* resi = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* ln2_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* mlp_out = (float*)malloc(sizeof(float) * tokens * embed_dim);

    /*LN1*/
    layer_norm(input, ln1_out, 0);

    /*Attn*/
    startTime = clock();
    multihead_attn(ln1_out, attn_out);
    clFinish(queue);
    endTime = clock();
    mhaTime += endTime - startTime;

    /*Residual1*/
    residual(input, attn_out, resi);

    /*LN2*/
    layer_norm(resi, ln2_out, 1);

    /*MLP*/
    startTime = clock();
    mlp_block_opencl(ln2_out, mlp_out);
    clFinish(queue);
    endTime = clock();
    mlpTime += endTime - startTime;

    /*Residual2*/
    residual(resi, mlp_out, output);

    free(ln1_out); free(attn_out); free(resi); free(ln2_out); free(mlp_out);

    clFinish(queue);
    clock_t e = clock();
    ecnTime += (e - s);

    encoder_count++;
}

void Softmax_opencl(float* logits, float* probs, int batch, int classes)
{
    // 1) logits 업로드
    err = clEnqueueWriteBuffer(queue, logits_buf, CL_TRUE, 0,
        sizeof(float) * batch * classes, logits, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 2) 커널 인자 설정
    err = clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &logits_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &probs_buf);
    CHECK_ERROR(err);
    err = clSetKernelArg(softmax_kernel, 2, sizeof(int), &classes);
    CHECK_ERROR(err);

    // 3) 커널 실행: batch 개의 work-item, 각 work-item이 한 row 처리
    size_t gws[1] = { (size_t)batch };
    size_t lws[1] = { 1 };

    err = clEnqueueNDRangeKernel(queue, softmax_kernel, 1,
        NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 4) 결과 읽기
    err = clEnqueueReadBuffer(queue, probs_buf, CL_TRUE, 0,
        sizeof(float) * batch * classes, probs, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void InitOpenCLElements(ImageData* image, Network* networks) {

    // 기기 설정
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    CHECK_ERROR(err);
    sources[0] = get_source_code("kernel_Preprocess.cl", &kernel_source_size[0]);
    sources[1] = get_source_code("kernel_MHA.cl", &kernel_source_size[1]);
    sources[2] = get_source_code("kernel_MLP.cl", &kernel_source_size[2]);
    program = clCreateProgramWithSource(context, NUM_KERNEL_FILES, (const char**)sources, &kernel_source_size, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

    // preprocessing 커널
    kernel_im2col = clCreateKernel(program, "im2col", &err);
    CHECK_ERROR(err);
    kernel_convAddBias = clCreateKernel(program, "conv_add_bias", &err);
    CHECK_ERROR(err);
    kernel_addClassToken_posEmbeding = clCreateKernel(program, "add_classToken_posEmbed", &err);
    CHECK_ERROR(err);
    // preprocessing 버퍼
    int output_size = img_size / patch_size;
    for (int i = 0; i < NUM_IMAGE; i++) {
        convInputBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * image->c * image->w * image->h, NULL, &err);
        CHECK_ERROR(err);
    }
    convArrangedInputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * image->c * image->w * image->h, NULL, &err);
    CHECK_ERROR(err);
    convTmpBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * output_size * output_size, NULL, &err);
    CHECK_ERROR(err);
    preprocessingOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * tokens, NULL, &err);
    CHECK_ERROR(err);
    convWeightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * in_chans * patch_size * patch_size, NULL, &err);
    CHECK_ERROR(err);
    convBiasBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    classTokenBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    posEmbedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * tokens, NULL, &err);
    CHECK_ERROR(err);
    // preprocessing 인풋 및 가중치 전송
    for (int i = 0; i < NUM_IMAGE; i++) {
        err = clEnqueueWriteBuffer(queue, convInputBuffer[i], CL_TRUE, 0, sizeof(float) * image->c * image->w * image->h, image[i].data, 0, NULL, NULL);
        CHECK_ERROR(err);
    }
    err = clEnqueueWriteBuffer(queue, convWeightBuffer, CL_TRUE, 0, sizeof(float) * embed_dim * in_chans * patch_size * patch_size, networks[1].data, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, convBiasBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[2].data, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, classTokenBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[0].data, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, posEmbedBuffer, CL_TRUE, 0, sizeof(float) * embed_dim * (output_size * output_size + 1), networks[3].data, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Layer Normalize 커널
    kernel_layer_normalize = clCreateKernel(program, "layer_normalize", &err);
    CHECK_ERROR(err);
    // Layer Normalize 버퍼
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < 2; j++) {
            layerNormalizeWeightBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
            CHECK_ERROR(err);
            layerNormalizeBiasBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
            CHECK_ERROR(err);
        }
    }
    finalLayerNormalizeWeightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    finalLayerNormalizeBiasBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    layerNormalizeInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    layerNormalizeOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);

    // Layer Normalize 가중치 전송
    for (int i = 0; i < depth; i++) {
        int idx = 4 + i * 12;
        err = clEnqueueWriteBuffer(queue, layerNormalizeWeightBuffer[i][0], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, layerNormalizeBiasBuffer[i][0], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 1].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, layerNormalizeWeightBuffer[i][1], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 6].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, layerNormalizeBiasBuffer[i][1], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 7].data, 0, NULL, NULL);
        CHECK_ERROR(err);
    }
    err = clEnqueueWriteBuffer(queue, finalLayerNormalizeWeightBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[148].data, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, finalLayerNormalizeBiasBuffer, CL_TRUE, 0, sizeof(float) * embed_dim, networks[149].data, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Residual 커널
    kernel_residual = clCreateKernel(program, "residual", &err);
    CHECK_ERROR(err);
    // Residual 버퍼
    residualInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    residualAddBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    residualOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);

    // MHA 커널
    kernel_transpose = clCreateKernel(program, "transpose", &err);
    CHECK_ERROR(err);
    kernel_gemm = clCreateKernel(program, "gemm", &err);
    CHECK_ERROR(err);
    kernel_mhaAddBias = clCreateKernel(program, "add_bias", &err);
    CHECK_ERROR(err);
    kernel_divide = clCreateKernel(program, "divide_head", &err);
    CHECK_ERROR(err);
    kernel_scale = clCreateKernel(program, "scale_score", &err);
    CHECK_ERROR(err);
    kernel_softmax = clCreateKernel(program, "softmax_score", &err);
    CHECK_ERROR(err);
    kernel_concat = clCreateKernel(program, "copy_head_output", &err);
    CHECK_ERROR(err);

    // MHA 버퍼
    int head_dim = embed_dim / num_heads;
    mhaInputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    for (int i = 0; i < 3; i++) {
        qkvBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        transposedInWeightBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        dividedQkvBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);
        CHECK_ERROR(err);
    }
    transposedKeyBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);
    CHECK_ERROR(err);
    scoreBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * tokens, NULL, &err);
    CHECK_ERROR(err);
    headOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);
    CHECK_ERROR(err);
    attnOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    transposedOutWeightBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    mhaOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < in_chans; j++) {
            mhaInWeightBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);
            CHECK_ERROR(err);
            mhaInBiasBuffer[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
            CHECK_ERROR(err);
        }
        mhaOutWeightBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        mhaOutBiasBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);
        CHECK_ERROR(err);
    }
    // in_weight, in_bias, out_weight, out_bias 전송
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < in_chans; j++) {
            err = clEnqueueWriteBuffer(queue, mhaInWeightBuffer[i][j], CL_TRUE, 0, sizeof(float) * embed_dim * embed_dim, (networks[6 + i * depth].data + j * embed_dim * embed_dim), 0, NULL, NULL);
            CHECK_ERROR(err);
            err = clEnqueueWriteBuffer(queue, mhaInBiasBuffer[i][j], CL_TRUE, 0, sizeof(float) * embed_dim, (networks[7 + i * depth].data + j * embed_dim), 0, NULL, NULL);
            CHECK_ERROR(err);
        }
        err = clEnqueueWriteBuffer(queue, mhaOutWeightBuffer[i], CL_TRUE, 0, sizeof(float) * embed_dim * embed_dim, networks[8 + i * depth].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, mhaOutBiasBuffer[i], CL_TRUE, 0, sizeof(float) * embed_dim, networks[9 + i * depth].data, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // MLP 커널
    int Hidden_dim = hidden_dim;
    linear_kernel = clCreateKernel(program, "linear_kernel", &err);
    CHECK_ERROR(err);
    gelukernel = clCreateKernel(program, "gelu_kernel_inplace", &err);
    CHECK_ERROR(err);

    // MLP(Linear) 버퍼
    size_t max_in = (embed_dim > Hidden_dim) ? embed_dim : Hidden_dim;
    inputBuffer_mlp = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * tokens * max_in, NULL, &err);
    CHECK_ERROR(err);
    fc1_outBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * Hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    outputBuffer_mlp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    for (int i = 0; i < depth; i++) {
        int idx = 12 + i * 12;
        weight1Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * Hidden_dim * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        bias1Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * Hidden_dim, NULL, &err);
        CHECK_ERROR(err);
        weight2Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * Hidden_dim, NULL, &err);
        CHECK_ERROR(err);
        bias2Buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
        CHECK_ERROR(err);
    }
    finalLinearWeightBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_classes * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    finalLinearBiasBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_classes, NULL, &err);
    CHECK_ERROR(err);
    // Linear 가중치 전송
    for (int i = 0; i < depth; i++) {
        int idx = 12 + i * 12;
        err = clEnqueueWriteBuffer(queue, weight1Buffer[i], CL_TRUE, 0, sizeof(float) * Hidden_dim * embed_dim, networks[idx].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, bias1Buffer[i], CL_TRUE, 0, sizeof(float) * Hidden_dim, networks[idx + 1].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, weight2Buffer[i], CL_TRUE, 0, sizeof(float) * embed_dim * Hidden_dim, networks[idx + 2].data, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, bias2Buffer[i], CL_TRUE, 0, sizeof(float) * embed_dim, networks[idx + 3].data, 0, NULL, NULL);
        CHECK_ERROR(err);
    }
    err = clEnqueueWriteBuffer(queue, finalLinearWeightBuffer, CL_TRUE, 0, sizeof(float) * num_classes * embed_dim, networks[150].data, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, finalLinearBiasBuffer, CL_TRUE, 0, sizeof(float) * num_classes, networks[151].data, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Softmax 커널
    softmax_kernel = clCreateKernel(program, "softmax_kernel", &err);
    CHECK_ERROR(err);
    // Softmax 버퍼
    int max_batch = 1;
    int max_classes = num_classes;
    logits_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * max_batch * max_classes, NULL, &err);
    CHECK_ERROR(err);
    probs_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * max_batch * max_classes, NULL, &err);
    CHECK_ERROR(err);

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
    clReleaseMemObject(layerNormalizeInputBuffer);
    clReleaseMemObject(layerNormalizeOutputBuffer);

    // MHA 커널
    clReleaseKernel(kernel_divide);
    clReleaseKernel(kernel_transpose);
    clReleaseKernel(kernel_gemm);
    clReleaseKernel(kernel_mhaAddBias);
    // MHA 버퍼
    clReleaseMemObject(mhaInputBuffer);
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < in_chans; j++) {
            clReleaseMemObject(mhaInWeightBuffer[i][j]);
            clReleaseMemObject(mhaInBiasBuffer[i][j]);
        }
        clReleaseMemObject(mhaOutWeightBuffer[i]);
        clReleaseMemObject(mhaOutBiasBuffer[i]);
    }
    for (int i = 0; i < 3; i++) {
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
    clReleaseMemObject(inputBuffer_mlp);
    clReleaseMemObject(fc1_outBuffer);
    clReleaseMemObject(outputBuffer_mlp);
    for (int i = 0; i < depth; i++) {
        clReleaseMemObject(weight1Buffer[i]);
        clReleaseMemObject(weight2Buffer[i]);
        clReleaseMemObject(bias2Buffer[i]);
        clReleaseMemObject(bias1Buffer[i]);
    }
    clReleaseMemObject(finalLinearWeightBuffer);
    clReleaseMemObject(finalLinearBiasBuffer);

    // Softmax 커널
    clReleaseKernel(softmax_kernel);
    // Softmax 버퍼
    clReleaseMemObject(logits_buf);
    clReleaseMemObject(probs_buf);

    // 기기 설정 해제
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
    for (int i = 0; i < NUM_KERNEL_FILES; i++) free(sources[i]);
}

////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_prl(ImageData* image, Network* networks, float** probabilities) {

    InitOpenCLElements(image, networks);

    float* pre_layer;
    float* enc_layer[12];
    float* enc_output;

    pre_layer = (float*)malloc(sizeof(float) * embed_dim * tokens);
    for (int i = 0; i < 12; i++) enc_layer[i] = (float*)malloc(sizeof(float) * embed_dim * tokens);
    enc_output = (float*)malloc(sizeof(float) * embed_dim * tokens);

    float* cls_token = (float*)malloc(sizeof(float) * embed_dim);
    float* cls_output = (float*)malloc(sizeof(float) * num_classes);

    for (image_count = 0; image_count < image->n; image_count++) {
        clock_t imgStartTime = clock();
        /*patch embedding*/
        startTime = clock();
        ImagePreprocessing(image[image_count].data, pre_layer);
        clFinish(queue);
        endTime = clock();
        printf("ImagePreprocessing: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

        /*Encoder - 12 Layers*/
        encoder_count = 0;
        clock_t encStartTime = clock();
        mhaTime = 0;
        mlpTime = 0;
        ecnTime = 0;
        Encoder(pre_layer, enc_layer[0]);
        Encoder(enc_layer[0], enc_layer[1]);
        Encoder(enc_layer[1], enc_layer[2]);
        Encoder(enc_layer[2], enc_layer[3]);
        Encoder(enc_layer[3], enc_layer[4]);
        Encoder(enc_layer[4], enc_layer[5]);
        Encoder(enc_layer[5], enc_layer[6]);
        Encoder(enc_layer[6], enc_layer[7]);
        Encoder(enc_layer[7], enc_layer[8]);
        Encoder(enc_layer[8], enc_layer[9]);
        Encoder(enc_layer[9], enc_layer[10]);
        Encoder(enc_layer[10], enc_layer[11]);
        clFinish(queue);

        endTime = clock();
        printf("mha avg: %lf\n", (double)(mhaTime / 12.0f) / CLOCKS_PER_SEC);
        printf("mlp avg: %lf\n", (double)(mlpTime / 12.0f) / CLOCKS_PER_SEC);
        printf("Encoder avg: %lf\n", (double)(ecnTime / 12.0f) / CLOCKS_PER_SEC);
        printf("Encoder 12: %lf\n", (double)(endTime - encStartTime) / CLOCKS_PER_SEC);

        layer_norm(enc_layer[11], enc_output, 3);

        /* Token 값 추출 */
        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);

        linear_layer_opencl(cls_token, cls_output, 1, embed_dim, num_classes, 1, inputBuffer_mlp, finalLinearWeightBuffer, finalLinearBiasBuffer, fc1_outBuffer);

        /* 확률분포 추출 */
        Softmax_opencl(cls_output, probabilities[image_count], 1, num_classes);
        clFinish(queue);

        endTime = clock();
        printf("image %d: %lf\n\n", image_count, (double)(endTime - imgStartTime) / CLOCKS_PER_SEC);
    }

    ReleaseOpenCLElements();
    free(pre_layer);
    for (int i = 0; i < 12; i++) free(enc_layer[i]);
    free(enc_output);
    free(cls_token);
    free(cls_output);
}