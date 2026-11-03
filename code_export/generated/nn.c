#include "nn.h"
#include <stdio.h>
#include <math.h>

static void fc_layer(float *input, float *output, const float *kernel, const float *bias, int input_size, int output_size, int activation) {
    for (int i = 0; i < output_size; i++) {
        output[i] = bias[i];
        for (int j = 0; j < input_size; j++) {
            output[i] += input[j] * kernel[i * input_size + j];
        }
        if (activation == 1) { // SILU
         output[i] = output[i] / (1 + expf(-output[i]));   //  output[i] > 0 ? output[i]:0.0f; // ReLU activation
        }    
    }
}

void forward_(float *input, float *output) {
    float buffer1[32];
    float buffer2[32];
    float *current_input = buffer1;
    float *current_output = buffer2;
    float *temp;
    fc_layer(input, buffer1, hidden_0_kernel, hidden_0_bias, 134, 32, 1);
    fc_layer(current_input, current_output, hidden_1_kernel, hidden_1_bias, 32, 32, 1);
    temp = current_input;
    current_input = current_output;
    current_output = temp;
    fc_layer(current_input, current_output, hidden_2_kernel, hidden_2_bias, 32, 32, 1);
    temp = current_input;
    current_input = current_output;
    current_output = temp;
    fc_layer(current_input, current_output, hidden_3_kernel, hidden_3_bias, 32, 32, 1);
    temp = current_input;
    current_input = current_output;
    current_output = temp;
    fc_layer(current_input, output, hidden_4_kernel, hidden_4_bias, 32, 16, 0);



    for (int i = 0; i < 4; i++) {
        output[i] = 1 - 2 / (expf(2*output[i]) + 1);
    }
}
