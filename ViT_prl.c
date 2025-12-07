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
#define embed_dim 768
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define dropout 0.0         //?? 이거 포함 아래 3개는 미사용?
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6
#define TILE 16
#define NUM_IMAGE 1     //for test. 100개로 고쳐야 함
#define hidden_dim 3072

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
#define NUM_KERNEL_FILES 2
size_t kernel_source_size[NUM_KERNEL_FILES];
char* sources[NUM_KERNEL_FILES];
cl_program program;

// 이미지, 가중치 인덱싱용 전역변수들
int image_count = 0;
int encoder_count = 0;

// Preprocessing에 필요한 커널
cl_kernel kernel_im2col;
cl_kernel kernel_convAddBias;
cl_kernel kernel_addClassToken_posEmbeding;
// Preprocessing에 필요한 버퍼
cl_mem convInputBuffer[NUM_IMAGE], convArrangedInputBuffer;
cl_mem convTmpBuffer, preprocessingOutputBuffer;
cl_mem convWeightBuffer, convBiasBuffer;
cl_mem classTokenBuffer, posEmbedBuffer;

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
cl_mem qkvBuffer[3], mhaInWeightBuffer[12][3], transposedInWeightBuffer[3], mhaInBiasBuffer[12][3], dividedQkvBuffer[3];
cl_mem transposedKeyBuffer, scoreBuffer, headOutputBuffer, attnOutputBuffer;
cl_mem mhaOutWeightBuffer[12], transposedOutWeightBuffer, mhaOutBiasBuffer[12];

//mlp
cl_kernel linear_kernel;
cl_mem inputBuffer_mlp;
cl_mem weight1Buffer;
cl_mem weight2Buffer;
cl_mem bias1Buffer;
cl_mem bias2Buffer;
cl_mem fc1_outBuffer;
cl_mem outputBuffer_mlp;

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
    size_t ctNpe_size = embed_dim * (output_size * output_size + 1);
    err = clEnqueueNDRangeKernel(queue, kernel_addClassToken_posEmbeding, 1, NULL, &ctNpe_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 최종 계산 결과를 out에 읽어오기
    clFinish(queue);
    err = clEnqueueReadBuffer(queue, preprocessingOutputBuffer, CL_TRUE, 0, sizeof(float) * embed_dim * (output_size * output_size + 1), output, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void layer_norm(float* input, float* output, Network weight, Network bias) {
    int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;

    for (int t = 0; t < token; t++) {
        float sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < embed_dim; i++) {
            float val = input[t * embed_dim + i];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / embed_dim;
        float var = sum_sq / embed_dim - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < embed_dim; i++) {
            int idx = t * embed_dim + i;
            output[idx] = (input[idx] - mean) * inv_std * weight.data[i] + bias.data[i];
        }
    }
}

void multihead_attn(float* input, float* output) {

    // MHA 계산에 필요한 변수들
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int embedDim = embed_dim;
    int head_dim = embed_dim / num_heads;
    float hd_float = head_dim;  // float head_dim for scaling
    int padding = 256;  // padded tokens for softmax

    // 계산 사이즈 정의
    size_t weight_traspose_size[2] = { embedDim, embedDim };
    size_t weight_size[2] = { embedDim, tokens };
    //size_t weight_size_padded[2] = { 768, 224 };  //embedDim, tokens_padded 
    size_t head_size[2] = { tokens, head_dim };
    size_t rev_head_size[2] = { head_dim, tokens };
    size_t score_size[2] = { tokens, tokens };
    //size_t score_size_padded[2] = { 224 , 224 };
    size_t softmax_local_size[2] = { 1, tokens };
    //size_t gemm_local_size[2] = { TILE_SIZE, TILE_SIZE };     //tiling?

    // input 전송
    err = clEnqueueWriteBuffer(queue, mhaInputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embedDim, input, 0, NULL, NULL);
    CHECK_ERROR(err);

    for (int i = 0; i < 3; i++) {
        // in_weight 전치 - gemm 계산을 위한 전처리
        err = clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mhaInWeightBuffer[encoder_count][i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 2, sizeof(int), &embedDim);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_transpose, 3, sizeof(int), &embedDim);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_transpose, 2, NULL, weight_traspose_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // Set QKV (Excute gemm)
    err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &mhaInputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &embedDim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &embedDim);
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
    err = clSetKernelArg(kernel_mhaAddBias, 2, sizeof(int), &embedDim);
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
            err = clSetKernelArg(kernel_divide, 2, sizeof(int), &embedDim);
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
        err = clEnqueueNDRangeKernel(queue, kernel_scale, 2, NULL, score_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // softmax 적용
        err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 1, sizeof(float) * tokens, NULL);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 2, sizeof(int), &padding);
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
        err = clSetKernelArg(kernel_concat, 2, sizeof(int), &embedDim);
        CHECK_ERROR(err);
        int headOffset = h * head_dim;
        err = clSetKernelArg(kernel_concat, 3, sizeof(int), &headOffset);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_concat, 2, NULL, head_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // out_weight 전치 - gemm 계산을 위한 전처리
    err = clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &mhaOutWeightBuffer[encoder_count]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedOutWeightBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_transpose, 2, sizeof(int), &embedDim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_transpose, 3, sizeof(int), &embedDim);
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
    err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &embedDim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &embedDim);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // out_bias 값 더하기
    err = clSetKernelArg(kernel_mhaAddBias, 0, sizeof(cl_mem), &mhaOutputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_mhaAddBias, 1, sizeof(cl_mem), &mhaOutBiasBuffer[encoder_count]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_mhaAddBias, 2, sizeof(int), &embedDim);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel_mhaAddBias, 2, NULL, weight_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 최종 결과 읽어와서 output에 저장
    err = clEnqueueReadBuffer(queue, mhaOutputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embedDim, output, 0, NULL, NULL);
    CHECK_ERROR(err);
}

float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}
void gelu_activation(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = gelu(input[i]);
    }
}

void linear_layer(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias) {
    for (int t = 0; t < tokens; t++) {
        for (int o = 0; o < out_features; o++) {
            float sum = bias.data[o];
            for (int i = 0; i < in_features; i++) {
                sum += input[t * in_features + i] * weight.data[o * in_features + i];
            }
            output[t * out_features + o] = sum;
        }
    }
}
/*
void mlp_block(float* input, float* output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
    int Embed_dim = embed_dim; //768

    float* fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);

    linear_layer(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
    // GELU 활성화
    for (int i = 0; i < tokens * hidden_dim; i++) {
        fc1_out[i] = gelu(fc1_out[i]);
    }
    // fc2: (tokens, in_dim)
    linear_layer(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);
    free(fc1_out);
}
*/
void mlp_block_opencl(float* input, float* output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {

    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
    int Embed_dim = embed_dim; //768
    int Hidden_dim = hidden_dim;
    err = clEnqueueWriteBuffer(queue, inputBuffer_mlp, CL_TRUE, 0,
        sizeof(float) * tokens * Embed_dim, input, 0, NULL, NULL);
    CHECK_ERROR(err);
    float* fc1_cpu = (float*)malloc(sizeof(float) * tokens * Hidden_dim);
    float* fc2_cpu = (float*)malloc(sizeof(float) * tokens * Embed_dim);

    size_t global_fc1[2] = { tokens, Hidden_dim };


    // fc1 weight, bias 업로드
    err = clEnqueueWriteBuffer(queue, weight1Buffer, CL_TRUE, 0,
        sizeof(float) * Hidden_dim * Embed_dim,
        fc1_weight.data, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bias1Buffer, CL_TRUE, 0,
        sizeof(float) * Hidden_dim,
        fc1_bias.data, 0, NULL, NULL);
    CHECK_ERROR(err);


    clSetKernelArg(linear_kernel, 0, sizeof(cl_mem), &inputBuffer_mlp);
    clSetKernelArg(linear_kernel, 1, sizeof(cl_mem), &weight1Buffer);
    clSetKernelArg(linear_kernel, 2, sizeof(cl_mem), &bias1Buffer);
    clSetKernelArg(linear_kernel, 3, sizeof(cl_mem), &fc1_outBuffer);
    clSetKernelArg(linear_kernel, 4, sizeof(int), &tokens);
    clSetKernelArg(linear_kernel, 5, sizeof(int), &Embed_dim);
    clSetKernelArg(linear_kernel, 6, sizeof(int), &Hidden_dim);


    size_t global[2] = { tokens, Hidden_dim };
    err = clEnqueueNDRangeKernel(queue, linear_kernel, 2, NULL, global_fc1, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, fc1_outBuffer, CL_TRUE, 0,
        sizeof(float) * tokens * Hidden_dim, fc1_cpu, 0, NULL, NULL);
    CHECK_ERROR(err);

    //GELU활성화
    for (int i = 0; i < tokens * Hidden_dim; i++) {
        fc1_cpu[i] = gelu(fc1_cpu[i]);
    }
    err = clEnqueueWriteBuffer(queue, fc1_outBuffer, CL_TRUE, 0,
        sizeof(float) * tokens * Hidden_dim, fc1_cpu, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, weight2Buffer, CL_TRUE, 0,
        sizeof(float) * Embed_dim * Hidden_dim,
        fc2_weight.data, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bias2Buffer, CL_TRUE, 0,
        sizeof(float) * Embed_dim,
        fc2_bias.data, 0, NULL, NULL);
    CHECK_ERROR(err);


    clSetKernelArg(linear_kernel, 0, sizeof(cl_mem), &fc1_outBuffer);
    clSetKernelArg(linear_kernel, 1, sizeof(cl_mem), &weight2Buffer);
    clSetKernelArg(linear_kernel, 2, sizeof(cl_mem), &bias2Buffer);
    clSetKernelArg(linear_kernel, 3, sizeof(cl_mem), &outputBuffer_mlp);
    clSetKernelArg(linear_kernel, 4, sizeof(int), &tokens);
    clSetKernelArg(linear_kernel, 5, sizeof(int), &Hidden_dim);
    clSetKernelArg(linear_kernel, 6, sizeof(int), &Embed_dim);

    size_t global_fc2[2] = { tokens, Embed_dim };
    err = clEnqueueNDRangeKernel(queue, linear_kernel, 2, NULL, global_fc2, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 최종 output 읽기
    err = clEnqueueReadBuffer(queue, outputBuffer_mlp, CL_TRUE, 0,
        sizeof(float) * tokens * Embed_dim, output, 0, NULL, NULL);
    CHECK_ERROR(err);

    free(fc1_cpu);
    free(fc2_cpu);

}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////
void Encoder(float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {
    clock_t s = clock();

    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    float* ln1_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* attn_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* residual = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* ln2_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* mlp_out = (float*)malloc(sizeof(float) * tokens * embed_dim);

    /*LN1*/
    layer_norm(input, ln1_out, ln1_w, ln1_b);

    /*Attn*/
    startTime = clock();
    multihead_attn(ln1_out, attn_out);
    clFinish(queue);
    endTime = clock();
    mhaTime += endTime - startTime;

    /*Residual1*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        residual[i] = input[i] + attn_out[i];
    }

    /*LN2*/
    layer_norm(residual, ln2_out, ln2_w, ln2_b);

    /*MLP*/
    startTime = clock();
    mlp_block_opencl(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    //mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    clFinish(queue);
    endTime = clock();
    mlpTime += endTime - startTime;

    /*Residual2*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        output[i] = residual[i] + mlp_out[i];
    }

    free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);

    clFinish(queue);
    clock_t e = clock();
    ecnTime += (e - s);

    encoder_count++;
}

void Softmax(float* logits, float* probabilities, int length) {
    // 수치 안정성을 위한 최대값 계산
    float max_val = logits[0];
    for (int i = 1; i < length; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // 각 원소에 대해 exp(logit - max_val)을 계산하고 합산
    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        probabilities[i] = expf(logits[i] - max_val);
        sum_exp += probabilities[i];
    }

    // 확률값으로 정규화
    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum_exp;
    }
}

////////////////////////////////////// layer별 size //////////////////////////////////////

const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1);

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
    sources[0] = get_source_code("kernel_MHA.cl", &kernel_source_size[0]);
    //sources[1] = get_source_code("kernel_MLP.cl", &kernel_source_size);
    sources[1] = get_source_code("kernel_Preprocess.cl", &kernel_source_size[1]);
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
    preprocessingOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * (output_size * output_size + 1), NULL, &err);
    CHECK_ERROR(err);
    convWeightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * in_chans * patch_size * patch_size, NULL, &err);
    CHECK_ERROR(err);
    convBiasBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    classTokenBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    posEmbedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * (output_size * output_size + 1), NULL, &err);
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
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
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

    int Hidden_dim = hidden_dim;

    linear_kernel = clCreateKernel(program, "linear_kernel", &err);
    CHECK_ERROR(err);

    inputBuffer_mlp = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    fc1_outBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * tokens * Hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    outputBuffer_mlp = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    weight1Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * Hidden_dim * embed_dim, NULL, &err);
    CHECK_ERROR(err);
    bias1Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * Hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    weight2Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * embed_dim * Hidden_dim, NULL, &err);
    CHECK_ERROR(err);
    bias2Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * embed_dim, NULL, &err);
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

    clReleaseMemObject(inputBuffer_mlp);
    clReleaseMemObject(fc1_outBuffer);
    clReleaseMemObject(outputBuffer_mlp);
    clReleaseMemObject(weight1Buffer);
    clReleaseMemObject(weight2Buffer);
    clReleaseMemObject(bias2Buffer);
    clReleaseMemObject(bias1Buffer);
    clReleaseKernel(linear_kernel);
    // 기기 설정 해제
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
    free(sources[0]);
}

////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_prl(ImageData* image, Network* networks, float** probabilities) {

    InitOpenCLElements(image, networks);

    float* pre_layer;
    float* enc_layer[12];
    float* enc_output;

    pre_layer = (float*)malloc(sizeof(float) * enc_size);
    for (int i = 0; i < 12; i++) {
        enc_layer[i] = (float*)malloc(sizeof(float) * enc_size);
    }
    enc_output = (float*)malloc(sizeof(float) * enc_size);

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
        Encoder(pre_layer, enc_layer[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

        Encoder(enc_layer[0], enc_layer[1],
            networks[16], networks[17], networks[18], networks[19],
            networks[20], networks[21], networks[22], networks[23],
            networks[24], networks[25], networks[26], networks[27]);

        Encoder(enc_layer[1], enc_layer[2],
            networks[28], networks[29], networks[30], networks[31],
            networks[32], networks[33], networks[34], networks[35],
            networks[36], networks[37], networks[38], networks[39]);

        Encoder(enc_layer[2], enc_layer[3],
            networks[40], networks[41], networks[42], networks[43],
            networks[44], networks[45], networks[46], networks[47],
            networks[48], networks[49], networks[50], networks[51]);

        Encoder(enc_layer[3], enc_layer[4],
            networks[52], networks[53], networks[54], networks[55],
            networks[56], networks[57], networks[58], networks[59],
            networks[60], networks[61], networks[62], networks[63]);

        Encoder(enc_layer[4], enc_layer[5],
            networks[64], networks[65], networks[66], networks[67],
            networks[68], networks[69], networks[70], networks[71],
            networks[72], networks[73], networks[74], networks[75]);

        Encoder(enc_layer[5], enc_layer[6],
            networks[76], networks[77], networks[78], networks[79],
            networks[80], networks[81], networks[82], networks[83],
            networks[84], networks[85], networks[86], networks[87]);

        Encoder(enc_layer[6], enc_layer[7],
            networks[88], networks[89], networks[90], networks[91],
            networks[92], networks[93], networks[94], networks[95],
            networks[96], networks[97], networks[98], networks[99]);

        Encoder(enc_layer[7], enc_layer[8],
            networks[100], networks[101], networks[102], networks[103],
            networks[104], networks[105], networks[106], networks[107],
            networks[108], networks[109], networks[110], networks[111]);

        Encoder(enc_layer[8], enc_layer[9],
            networks[112], networks[113], networks[114], networks[115],
            networks[116], networks[117], networks[118], networks[119],
            networks[120], networks[121], networks[122], networks[123]);

        Encoder(enc_layer[9], enc_layer[10],
            networks[124], networks[125], networks[126], networks[127],
            networks[128], networks[129], networks[130], networks[131],
            networks[132], networks[133], networks[134], networks[135]);

        Encoder(enc_layer[10], enc_layer[11],
            networks[136], networks[137], networks[138], networks[139],
            networks[140], networks[141], networks[142], networks[143],
            networks[144], networks[145], networks[146], networks[147]);
        clFinish(queue);
        endTime = clock();
        printf("mha avg: %lf\n", (double)(mhaTime / 12.0f) / CLOCKS_PER_SEC);
        printf("mlp avg: %lf\n", (double)(mlpTime / 12.0f) / CLOCKS_PER_SEC);
        printf("Encoder avg: %lf\n", (double)(ecnTime / 12.0f) / CLOCKS_PER_SEC);
        printf("Encoder 12: %lf\n", (double)(endTime - encStartTime) / CLOCKS_PER_SEC);

        layer_norm(enc_layer[11], enc_output, networks[148], networks[149]);

        /* Token 값 추출 */
        float* cls_token = (float*)malloc(sizeof(float) * embed_dim);
        float* cls_output = (float*)malloc(sizeof(float) * num_classes);
        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);

        linear_layer(cls_token, cls_output, 1, embed_dim, num_classes, networks[150], networks[151]);
        /* 확률분포 추출 */
        Softmax(cls_output, probabilities[image_count], num_classes);
        clFinish(queue);
        endTime = clock();
        printf("image %d: %lf\n\n", image_count, (double)(endTime - imgStartTime) / CLOCKS_PER_SEC);
    }

    ReleaseOpenCLElements();
}