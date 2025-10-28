const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;
const lowerConv2d = zant.core.tensor.math_standard.lowerConv2d;

const tests_log = std.log.scoped(.test_conv_relu);

// Struct for params
pub const AutoPadMode = enum {
    notset,
    valid,
    same_upper,
    same_lower,
};

pub const ConvPreparedParams = struct {
    stride: [2]usize,
    dilations: [2]usize,
    pads: [4]usize,
    group: usize,
    filters_per_group: usize,
    channels_per_group: usize,
    auto_pad: AutoPadMode,
};

// CMSIS function declaration
pub const Dims = extern struct {
    n: i32,
    h: i32,
    w: i32,
    c: i32,
};

pub const Context = extern struct {
    buf: ?*anyopaque,
    size: i32,
};

pub const ConvParams = extern struct {
    input_offset: i32,
    output_offset: i32,
    stride: extern struct { h: i32, w: i32 },
    padding: extern struct { h: i32, w: i32 },
    dilation: extern struct { h: i32, w: i32 },
    activation: extern struct { min: i32, max: i32 },
};

pub const PerChannelQuantParams = extern struct {
    multiplier: [*]i32,
    shift: [*]i32,
};

pub const DwConvParams = extern struct {
    input_offset: i32,
    output_offset: i32,
    ch_mult: i32,
    stride: extern struct { h: i32, w: i32 },
    padding: extern struct { h: i32, w: i32 },
    dilation: extern struct { h: i32, w: i32 },
    activation: extern struct { min: i32, max: i32 },
};

extern fn wrapper_arm_convolve_s8(
    input_ptr: [*c]const f32,
    input_shape: [*c]const usize,
    weight_ptr: [*c]const f32,
    weight_shape: [*c]const usize,
    output_ptr: [*c]f32,
    output_shape: [*c]const usize,
    bias_ptr: ?*const f32,
    bias_len: usize,
    stride_ptr: [*c]const usize,
    pads_ptr: [*c]const usize,
    dilations_ptr: [*c]const usize,
    group: usize,
    filters_per_group: usize,
    channels_per_group: usize,
) callconv(.C) bool;

//Tutti valori positivi (ReLU non fa nulla)
test "ConvCMSIS" {
    tests_log.info("\n     CMSIS-conv-s8\n", .{});

    const allocator = pkgAllocator.allocator;

    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data: [1][1][3][3]f32 = .{
        .{
            .{
                .{ 1.0, 2.0, 3.0 },
                .{ 4.0, 5.0, 6.0 },
                .{ 7.0, 8.0, 9.0 },
            },
        },
    };

    var weight_shape = [_]usize{ 1, 1, 2, 2 };
    var weight_data: [1][1][2][2]f32 = .{
        .{
            .{
                .{ 1.0, 0.0 },
                .{ 0.0, 1.0 },
            },
        },
    };

    var output_shape = [_]usize{ 1, 1, 2, 2 };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input_tensor.deinit();

    var weight_tensor = try Tensor(f32).fromArray(&allocator, &weight_data, &weight_shape);
    defer weight_tensor.deinit();

    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    const params = ConvPreparedParams{
        .stride = .{ 1, 1 },
        .dilations = .{ 1, 1 },
        .pads = .{ 0, 0, 0, 0 },
        .group = 1,
        .filters_per_group = 1,
        .channels_per_group = 1,
        .auto_pad = .notset,
    };

    const expected: [1][1][2][2]f32 = .{
        .{
            .{
                .{ 6.0, 8.0 },
                .{ 12.0, 14.0 },
            },
        },
    };

    // !!!

    const result = wrapper_arm_convolve_s8(
        @as([*c]const f32, @ptrCast(input_tensor.data.ptr)),
        @as([*c]const usize, @ptrCast(input_shape[0..].ptr)),
        @as([*c]const f32, @ptrCast(weight_tensor.data.ptr)),
        @as([*c]const usize, @ptrCast(weight_shape[0..].ptr)),
        @as([*c]f32, @ptrCast(output_tensor.data.ptr)),
        @as([*c]const usize, @ptrCast(output_shape[0..].ptr)),
        null,
        0,
        @as([*c]const usize, @ptrCast(params.stride[0..].ptr)),
        @as([*c]const usize, @ptrCast(params.pads[0..].ptr)),
        @as([*c]const usize, @ptrCast(params.dilations[0..].ptr)),
        params.group,
        params.filters_per_group,
        params.channels_per_group,
    );

    std.debug.print("Zig dice: {}", .{result});
    // Ogni output è 4 (somma di 2x2 kernel di 1)
    // for (result) |val| {
    //     std.debug.print("{d}", .{val});
    // }
    std.debug.print("Expected: {}\n", .{result});
    for (expected) |val| {
        std.debug.print("{d}", .{val});
    }

    tests_log.info("Passed\n", .{});
}
