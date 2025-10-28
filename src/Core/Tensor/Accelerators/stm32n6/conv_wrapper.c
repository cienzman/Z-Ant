#include "conv_wrapper.h"

bool wrapper_arm_convolve_s8(
    const float *input,
    const size_t *input_shape,
    const float *weights,
    const size_t *weight_shape,
    float *output,
    const size_t *output_shape,
    const float *bias,
    size_t bias_len,
    const size_t *stride,
    const size_t *pads,
    const size_t *dilations,
    size_t group,
    size_t filters_per_group,
    size_t channels_per_group){
        // call CMSIS function
        bool res = arm_convolve_s8(
            NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL
        );

        //printf("%d",res);
        return res;
    }
