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
#define dropout 0.0
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES,
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    0 };
size_t kernel_source_size;
char* sources[3];
cl_program program;

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

void Conv2d(float* input, float* output, Network weight, Network bias) {
    int output_size = img_size / patch_size;

    for (int oc = 0; oc < embed_dim; ++oc) {
        for (int oh = 0; oh < output_size; ++oh) {
            for (int ow = 0; ow < output_size; ++ow) {
                float sum = bias.data[oc];

                for (int ic = 0; ic < in_chans; ++ic) {
                    for (int kh = 0; kh < patch_size; ++kh) {
                        for (int kw = 0; kw < patch_size; ++kw) {
                            int ih = oh * patch_size + kh;
                            int iw = ow * patch_size + kw;
                            int input_idx = (ic * img_size + ih) * img_size + iw;
                            int kernel_idx = ((oc * in_chans + ic) * patch_size + kh) * patch_size + kw;

                            sum += input[input_idx] * weight.data[kernel_idx];
                        }
                    }
                }

                output[(oc * output_size + oh) * output_size + ow] = sum;
            }
        }
    }
}

void flatten_transpose(float* input, float* output) {
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 각 공간 위치(oh, ow)를 하나의 패치로 취급하여 patch index 계산
    for (int oh = 0; oh < output_size; oh++) {
        for (int ow = 0; ow < output_size; ow++) {
            int patch_idx = oh * output_size + ow;
            for (int oc = 0; oc < embed_dim; oc++) {
                // 기존 입력은 (oc, oh, ow)
                int idx_input = (oc * output_size + oh) * output_size + ow;
                // 원하는 출력은 (patch_idx, oc)
                int idx_output = patch_idx * embed_dim + oc;
                output[idx_output] = input[idx_input];
                //printf("%f ",output[idx_output]);
            }
        }
    }
}

void class_token(float* patch_tokens, float* final_tokens, Network cls_tk) {
    // 이미지의 패치 수 계산: output_size = img_size / patch_size, num_patches = output_size^2
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 1. 첫 번째 토큰에 class token 복사 (networks[0].data에 저장됨, embed_dim 길이)
    for (int j = 0; j < embed_dim; j++) {
        final_tokens[j] = cls_tk.data[j];
    }

    // 2. 이후 patch_tokens를 이어붙임
    // final_tokens의 인덱스 embed_dim부터, patch_tokens 전체(embed_dim * num_patches) 복사
    memcpy(final_tokens + embed_dim, patch_tokens, sizeof(float) * embed_dim * num_patches);

    int total_tokens = num_patches + 1; // class token + patch tokens
    for (int i = 0; i < total_tokens * embed_dim; i++) {
        //("%f ", final_tokens[i]);
    }
    //printf("\n");
}

void pos_emb(float* input, float* output, Network pos_emb) {
    // output_size: 한 변의 패치 수, num_patches: 전체 패치 수, total_tokens: class token + patch tokens
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;
    int total_tokens = num_patches + 1;
    int total_elements = total_tokens * embed_dim;
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] + pos_emb.data[i];
    }
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

void multihead_attn(float* input, float* output, Network in_weight, Network in_bias, Network out_weight, Network out_bias) {
    // 기기 세팅

    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    float* Q = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* K = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* V = (float*)malloc(sizeof(float) * tokens * embed_dim);

    // 버퍼 생성
    cl_mem inputBuffer, qkvBuffer[3], inWeightBuffer[3], transposedInWeightBuffer[3], inBiasBuffer[3];
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * tokens * embed_dim, NULL, &err);
    CHECK_ERROR(err);

    for (int i = 0; i < 3; i++) {
        qkvBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        inWeightBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        transposedInWeightBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * embed_dim * embed_dim, NULL, &err);
        CHECK_ERROR(err);
        inBiasBuffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * embed_dim, NULL, &err);
        CHECK_ERROR(err);
    }

    // 버퍼에 메모리 전송
    // input 전송
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, input, 0, NULL, NULL);
    CHECK_ERROR(err);

    cl_kernel kernel_transpose = clCreateKernel(program, "transpose", &err);
    int ed = embed_dim;
    size_t trans_size[2] = { embed_dim, embed_dim };
    for (int i = 0; i < 3; i++) {
        // in_weight 전송
        err = clEnqueueWriteBuffer(queue, inWeightBuffer[i], CL_TRUE, 0, sizeof(float) * embed_dim * embed_dim, (in_weight.data + i * embed_dim * embed_dim), 0, NULL, NULL);
        CHECK_ERROR(err);
        // in_weight 전치 - gemm 계산을 위한 전처리
        clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &inWeightBuffer[i]);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 2, sizeof(int), &ed);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 3, sizeof(int), &ed);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_transpose, 2, NULL, trans_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
        // in_bias 전송
        err = clEnqueueWriteBuffer(queue, inBiasBuffer[i], CL_TRUE, 0, sizeof(float) * embed_dim, (in_bias.data + i * embed_dim), 0, NULL, NULL);
        CHECK_ERROR(err);
    }
    clFinish(queue);
    for (int i = 0; i < 3; i++) clReleaseMemObject(inWeightBuffer[i]);

    // Set QKV (Excute gemm)
    cl_kernel kernel_gemm = clCreateKernel(program, "gemm", &err);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &inputBuffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 3, sizeof(int), &tokens);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 4, sizeof(int), &ed);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_gemm, 5, sizeof(int), &ed);
    CHECK_ERROR(err);
    size_t weight_global_size[2] = { embed_dim, tokens };

    for (int i = 0; i < 3; i++) {
        err = clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &transposedInWeightBuffer[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &qkvBuffer[i]);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, weight_global_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    // Excute add_bias
    cl_kernel kernel_add = clCreateKernel(program, "add_bias", &err);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_add, 2, sizeof(int), &ed);
    CHECK_ERROR(err);

    for (int i = 0; i < 3; i++) {
        err = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), &qkvBuffer[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), &inBiasBuffer[i]);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_add, 2, NULL, weight_global_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    clFinish(queue);
    for (int i = 0; i < 3; i++) clReleaseMemObject(transposedInWeightBuffer[i]);
    for (int i = 0; i < 3; i++) clReleaseMemObject(inBiasBuffer[i]);

    //for test
    err = clEnqueueReadBuffer(queue, qkvBuffer[0], CL_TRUE, 0, sizeof(float) * tokens * embed_dim, Q, 0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, qkvBuffer[1], CL_TRUE, 0, sizeof(float) * tokens * embed_dim, K, 0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, qkvBuffer[2], CL_TRUE, 0, sizeof(float) * tokens * embed_dim, V, 0, NULL, NULL);

    int head_dim = embed_dim / num_heads;   //group size

    /*Attn 결과를 저장할 버퍼*/
    float* attn_output = (float*)malloc(sizeof(float) * tokens * embed_dim);
    for (int i = 0; i < tokens * embed_dim; i++) attn_output[i] = 0.0f;


    /*head별로 attn 수행*/
    cl_kernel kernel_divide = clCreateKernel(program, "divide_head", &err);
    size_t head_size[2] = { tokens, head_dim };
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim; //group id?

        // 헤드 분리
        cl_mem dividedQkvBuffer[3];
        for (int i = 0; i < 3; i++) {
            dividedQkvBuffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);
            CHECK_ERROR(err);

            clSetKernelArg(kernel_divide, 0, sizeof(cl_mem), &qkvBuffer[i]);
            CHECK_ERROR(err);
            clSetKernelArg(kernel_divide, 1, sizeof(cl_mem), &dividedQkvBuffer[i]);
            CHECK_ERROR(err);
            clSetKernelArg(kernel_divide, 2, sizeof(int), &ed);
            CHECK_ERROR(err);
            clSetKernelArg(kernel_divide, 3, sizeof(int), &head_dim);
            CHECK_ERROR(err);
            clSetKernelArg(kernel_divide, 4, sizeof(int), &h);
            CHECK_ERROR(err);

            err = clEnqueueNDRangeKernel(queue, kernel_divide, 2, NULL, head_size, NULL, 0, NULL, NULL);
            CHECK_ERROR(err);
        }
        // key 전치
        cl_mem transposedKeyBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * head_dim, NULL, &err);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &dividedQkvBuffer[1]);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &transposedKeyBuffer);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 2, sizeof(int), &tokens);
        CHECK_ERROR(err);
        clSetKernelArg(kernel_transpose, 3, sizeof(int), &head_dim);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_transpose, 2, NULL, head_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
        clFinish(queue);

        // score(QxKt) 행렬 생성
        cl_mem scoreBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * tokens * tokens, NULL, &err);
        CHECK_ERROR(err);
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
        size_t score_global_size[2] = { tokens, tokens };
        err = clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, score_global_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // 스코어 스케일링
        cl_kernel kernel_scale = clCreateKernel(program, "scale", &err);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_scale, 0, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        float hd = head_dim;
        err = clSetKernelArg(kernel_scale, 1, sizeof(float), &hd);
        CHECK_ERROR(err);
        err = clEnqueueNDRangeKernel(queue, kernel_scale, 2, NULL, score_global_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        // softmax 적용
        printf("%d\n", tokens);
        cl_kernel kernel_softmax = clCreateKernel(program, "softmax_score", &err);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &scoreBuffer);
        CHECK_ERROR(err);
        err = clSetKernelArg(kernel_softmax, 1, sizeof(float) * tokens, NULL);
        CHECK_ERROR(err);
        size_t softmax_local_size[2] = { 1, tokens };
        err = clEnqueueNDRangeKernel(queue, kernel_softmax, 2, NULL, score_global_size, softmax_local_size, 0, NULL, NULL);
        CHECK_ERROR(err);

        /*for (int i = 0; i < tokens; i++) {
            float max_val = scores[i * tokens];
            for (int j = 1; j < tokens; j++) {
                if (scores[i * tokens + j] > max_val) max_val = scores[i * tokens + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < tokens; j++) {
                scores[i * tokens + j] = expf(scores[i * tokens + j] - max_val);
                sum_exp += scores[i * tokens + j];
            }

            for (int j = 0; j < tokens; j++) {
                scores[i * tokens + j] /= sum_exp;
            }
        }*/

        //for test
        // attn_score 저장 공간
        float* scores = (float*)malloc(sizeof(float) * tokens * tokens);
        err = clEnqueueReadBuffer(queue, scoreBuffer, CL_TRUE, 0, sizeof(float) * tokens * tokens, scores, 0, NULL, NULL);
        CHECK_ERROR(err);

        // scores와 V를 곱해 head output 계산
        float* head_out = (float*)malloc(sizeof(float) * tokens * head_dim);
        for (int i = 0; i < tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < tokens; j++) {
                    sum += scores[i * tokens + j] * V[j * embed_dim + head_offset + d];
                }
                head_out[i * head_dim + d] = sum;
            }
        }

        // head_out를 attn_output의 해당 부분에 복사
        for (int i = 0; i < tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                attn_output[i * embed_dim + head_offset + d] = head_out[i * head_dim + d];
            }
        }

        free(scores);
        free(head_out);
    }

    free(Q); free(K); free(V);

    // 최종 선형 프로젝션
    for (int t = 0; t < tokens; t++) {
        for (int i = 0; i < embed_dim; i++) {
            float sum = out_bias.data[i];
            for (int j = 0; j < embed_dim; j++) {
                sum += attn_output[t * embed_dim + j] * out_weight.data[i * embed_dim + j];
            }
            output[t * embed_dim + i] = sum;
        }
    }
    free(attn_output);

    clReleaseKernel(kernel_gemm);
    clReleaseKernel(kernel_add);
    clReleaseKernel(kernel_transpose);
    clReleaseKernel(kernel_divide);
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
void mlp_block(float* input, float* output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
    int Embed_dim = embed_dim; //768
    int hidden_dim = ((int)(embed_dim * mlp_ratio)); //3072



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
    multihead_attn(ln1_out, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);
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
    mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    endTime = clock();
    mlpTime += endTime - startTime;

    /*Residual2*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        output[i] = residual[i] + mlp_out[i];
    }

    free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);

    clock_t e = clock();
    ecnTime += (e - s);
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
const int size[] = {
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // conv2D
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1), // class token
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1) // position embedding
};

const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1);

////////////////////////////////////// Model Architecture //////////////////////////////////////
void ViT_prl(ImageData* image, Network* networks, float** probabilities) {

    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    CHECK_ERROR(err);
    sources[0] = get_source_code("kernel_MHA.cl", &kernel_source_size);
    //sources[1] = get_source_code("kernel_MLP.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)sources, &kernel_source_size, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1);
    float* layer[4];
    float* enc_layer[12];
    float* enc_output;
    int  hidden_dim = ((int)(embed_dim * mlp_ratio));
    //printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);

    for (int i = 0; i < 4; i++) {
        layer[i] = (float*)malloc(sizeof(float) * size[i]);
    }
    for (int i = 0; i < 12; i++) {
        enc_layer[i] = (float*)malloc(sizeof(float) * enc_size);
    }
    enc_output = (float*)malloc(sizeof(float) * enc_size);

    for (int i = 0; i < image->n; i++) {
        clock_t imgStartTime = clock();
        /*patch embedding*/
        startTime = clock();
        Conv2d(image[i].data, layer[0], networks[1], networks[2]);
        endTime = clock();
        printf("Cov2D: %lf\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
        /*flatten and transpose*/
        flatten_transpose(layer[0], layer[1]);
        /*prepend class token*/
        class_token(layer[1], layer[2], networks[0]);
        /*position embedding*/
        pos_emb(layer[2], layer[3], networks[3]);

        /*Encoder - 12 Layers*/
        clock_t encStartTime = clock();
        ecnTime = 0;
        Encoder(layer[3], enc_layer[0],
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
        Softmax(cls_output, probabilities[i], num_classes);
        endTime = clock();
        printf("%d image: %lf\n", i, (double)(endTime - imgStartTime) / CLOCKS_PER_SEC);
    }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
    free(sources[0]);
}