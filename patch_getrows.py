import os

ultra_root = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra"
f_getrows = os.path.join(ultra_root, r"ggml\src\ggml-cuda\getrows.cu")

with open(f_getrows, 'r', encoding='utf-8') as fp:
    c = fp.read()

tq_kernel = """
// Trellis types: TQ_LANES threads per 128-weight block, codebook read through the cache
// (a get_rows CTA touches too few blocks to amortize a shared-memory copy).
#define CUDA_GET_ROWS_TQ_BLOCK_SIZE 256

template<ggml_type type, typename dst_t>
static __global__ void k_get_rows_tq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    typedef typename tq_block_info<type>::block_t block_t;

    ggml_cuda_pdl_sync();
    const int64_t nb   = ne00/QK_TQK; // blocks per row
    const int     lane = threadIdx.x % TQ_LANES;
    const uint32_t * lut = tq_lut_global();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const block_t * src0_row = (const block_t *) ((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t ib = (int64_t) blockIdx.y*(blockDim.x/TQ_LANES) + threadIdx.x/TQ_LANES; ib < nb;
                ib += (int64_t) gridDim.y*(blockDim.x/TQ_LANES)) {
            float v[16];
            tq_dequant_lane<type>(src0_row + ib, lane, lut, v);
            dst_t * y = dst_row + ib*QK_TQK + 16*lane;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                y[j] = ggml_cuda_cast<dst_t>(v[j]);
            }
        }
    }
}

template<ggml_type type, typename dst_t>
static void get_rows_cuda_tq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQK == 0);
    const int64_t nb = ne00/QK_TQK;
    const int64_t blocks_per_cta = CUDA_GET_ROWS_TQ_BLOCK_SIZE/TQ_LANES;

    const dim3 block_dims(CUDA_GET_ROWS_TQ_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(ne10, MIN((nb + blocks_per_cta - 1)/blocks_per_cta, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_tq<type, dst_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12_fdv, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}
"""

tq_cases = """        case GGML_TYPE_TQ2_T:
            get_rows_cuda_tq<GGML_TYPE_TQ2_T, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQK6:
            get_rows_cuda_tq<GGML_TYPE_TQK6, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQK7:
            get_rows_cuda_tq<GGML_TYPE_TQK7, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQP5:
            get_rows_cuda_tq<GGML_TYPE_TQP5, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQP6:
            get_rows_cuda_tq<GGML_TYPE_TQP6, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TQP7:
            get_rows_cuda_tq<GGML_TYPE_TQP7, dst_t>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
"""

# Insert tq_kernel before template<typename dst_t> static void ggml_cuda_get_rows_switch_src0_type
target_fn = "template<typename dst_t>\nstatic void ggml_cuda_get_rows_switch_src0_type("
c = c.replace(target_fn, tq_kernel + "\n" + target_fn, 1)

# Insert tq_cases before default: in ggml_cuda_get_rows_switch_src0_type
target_default = "        default:\n            GGML_ABORT(\"%s: unsupported src0 type: %s\\n\", __func__, ggml_type_name(src0_type));"
c = c.replace(target_default, tq_cases + target_default, 1)

with open(f_getrows, 'w', encoding='utf-8') as fp:
    fp.write(c)

print("SUCCESS: getrows.cu patched")
