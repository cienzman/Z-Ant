// ============================================================================
// File: test_op_conv_cmsis.zig
// Description:
//   This test is useful to verify basic integration of the CMSIS-NN library within
//   the Z-Ant tensor math module. This test runs a simple 2D convolution
//   (Conv2D) using the ARM CMSIS-NN `arm_convolve_s8` function and validates
//   linkage between Zig and the C-based CMSIS-NN runtime.
//
//   For now the purpose of this test is not numerical accuracy but functional validation:
//   - correct inclusion of CMSIS-NN headers
//   - correct allocation of intermediate buffers
//   - correct passing of quantization parameters
//   - correct structure mapping between Zig and C
// ============================================================================

const zant = @import("zant");
const std = @import("std");

// Import the C CMSIS-NN API into Zig.
// Only the main header `arm_nnfunctions.h` is needed for most NN operators.
const c = @cImport({
    @cInclude("arm_nnfunctions.h");
});

// ---------------------------------------------------------------------------
// TEST: "cmsis conv2d simple"
// ---------------------------------------------------------------------------
// The following test constructs a minimal example of an integer (int8) convolution
// to verify that the CMSIS-NN library functions are callable and behave
// as expected from Zig code.

test "cmsis conv2d simple" {
    var gpa = std.heap.page_allocator; // can we manage it better?

    // Convolution layer parameters
    var conv_params: c.cmsis_nn_conv_params = .{
        .input_offset = 0,
        .output_offset = 0,
        .stride = .{ .w = 1, .h = 1 },
        .padding = .{ .w = 0, .h = 0 },
        .dilation = .{ .w = 1, .h = 1 },
        .activation = .{ .min = -128, .max = 127 },
    };

    // Quantization parameters
    var mult: [1]i32 = [_]i32{1};
    var shift: [1]i32 = [_]i32{0};

    var quant_params: c.cmsis_nn_per_channel_quant_params = .{
        .multiplier = &mult,
        .shift = &shift,
    };

    // Tensor dimensions:
    // n = batch size
    // h = rows
    // W = columns
    // c = channles
    var input_dims: c.cmsis_nn_dims = .{ .n = 1, .w = 3, .h = 3, .c = 1 };
    var filter_dims: c.cmsis_nn_dims = .{ .w = 2, .h = 2, .c = 1 };
    var bias_dims: c.cmsis_nn_dims = .{ .w = 1, .h = 1, .c = 1 };
    var output_dims: c.cmsis_nn_dims = .{ .w = 2, .h = 2, .c = 1 };

    // Test Input Data
    var input_data: [9]i8 = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var filter_data: [4]i8 = [_]i8{ 1, 0, 0, 1 };
    var bias_data: [1]i32 = [_]i32{0};
    var output_data: [4]i8 = undefined;

    // Determine and allocate the required scratch buffer. You cannot use a null pointer to the buffer since CMSIS function arm_convolve_s8 would return error with code -1
    const buf_size = c.arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    std.log.info("Buffer size needed = {}", .{buf_size});

    const buffer = try gpa.alloc(u8, @intCast(buf_size));

    var ctx: c.cmsis_nn_context = .{
        .buf = buffer.ptr, //pointer to the buffer
        .size = 0,
    };

    // Run CMSIS-NN convolution
    const result = c.arm_convolve_s8(
        &ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        &input_data,
        &filter_dims,
        &filter_data,
        &bias_dims,
        &bias_data,
        null,
        &output_dims,
        &output_data,
    );

    // From CMSIS doc we know that arm_convolve_s8 return ARM_CMSIS_NN_SUCCESS(0) on successful completion
    std.log.info("arm_convolve_s8 result = {}", .{result});
    std.log.info("output_data = {any}", .{output_data});
}
