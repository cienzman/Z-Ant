#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Only include math.h and ARM headers when actually compiling for target, not
// during codegen
#ifndef ZANT_CODEGEN_PHASE
#include <math.h>

// Forward declarations for memory functions
extern void *malloc(size_t size);
extern void free(void *ptr);

#if defined(ZANT_HAS_CMSIS_DSP)
#include <arm_math.h>
#if defined(__has_include)
#if __has_include(<arm_nnfunctions.h>)
#define ZANT_HAS_CMSIS_NN 1
#include <arm_nnfunctions.h>
#endif
#endif
#endif
#endif

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
    size_t channels_per_group);