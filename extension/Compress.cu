__global__ void optimizedKernel(unsigned char* __restrict__ cmpData, 
                                const int* __restrict__ absQuant, 
                                const int* __restrict__ fixed_rate, 
                                bool encoding_selection,
                                int num_chunks)
{
    constexpr int CHUNK_SIZE = 32;
    constexpr int SUBCHUNK_SIZE = 8;

    __shared__ int s_fixed_rate[32];  // Assuming max 32 different fixed rates

    int tid = threadIdx.x;
    int chunk_idx = blockIdx.x * blockDim.x + tid;

    if (tid < 32) {
        s_fixed_rate[tid] = fixed_rate[tid];
    }
    __syncthreads();

    if (chunk_idx < num_chunks) {
        int chunk_idx_start = chunk_idx * CHUNK_SIZE;
        int cur_byte_ofs = chunk_idx * 4;  // Assuming 4 bytes per chunk
        int cmp_byte_ofs = cur_byte_ofs;

        uchar4 tmp_char;
        unsigned int mask = 1;

        for (int j = 0; j < 4; ++j) {
            int rate = s_fixed_rate[j];
            int subchunk_start = chunk_idx_start + j * SUBCHUNK_SIZE;

            for (int i = 0; i < rate - 1; ++i) {
                tmp_char = make_uchar4(0, 0, 0, 0);

                // Optimize bit extraction for each subchunk
                #pragma unroll
                for (int k = 0; k < SUBCHUNK_SIZE; ++k) {
                    int shift = 7 - k;
                    tmp_char.x |= ((absQuant[subchunk_start + k + 0] & mask) >> i) << shift;
                    tmp_char.y |= ((absQuant[subchunk_start + k + 8] & mask) >> i) << shift;
                    tmp_char.z |= ((absQuant[subchunk_start + k + 16] & mask) >> i) << shift;
                    tmp_char.w |= ((absQuant[subchunk_start + k + 24] & mask) >> i) << shift;
                }

                // Handle encoding_selection outside the main loop to reduce divergence
                if (encoding_selection && j == 0) {
                    tmp_char.w = (((absQuant[chunk_idx_start] & mask) >> i) << 7) | (tmp_char.w >> 1);
                }

                // Write back to global memory
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs / 4] = tmp_char;
                cmp_byte_ofs += 4;
                mask <<= 1;
            }

            // Handle padding part
            unsigned char padding[3] = {0, 0, 0};
            #pragma unroll
            for (int k = 0; k < SUBCHUNK_SIZE; ++k) {
                int shift = 7 - k;
                padding[0] |= ((absQuant[subchunk_start + k + 8] & mask) >> (rate - 1)) << shift;
                padding[1] |= ((absQuant[subchunk_start + k + 16] & mask) >> (rate - 1)) << shift;
                padding[2] |= ((absQuant[subchunk_start + k + 24] & mask) >> (rate - 1)) << shift;
            }
            
            // Write padding back to global memory
            cmpData[cmp_byte_ofs++] = padding[0];
            cmpData[cmp_byte_ofs++] = padding[1];
            cmpData[cmp_byte_ofs++] = padding[2];
        }

        // Update the final byte offset
        if (tid == blockDim.x - 1) {
            atomicMax(reinterpret_cast<int*>(cmpData), cmp_byte_ofs);
        }
    }
}
