#include "cusingler.cuh"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <thread>

#include "cuda_runtime.h"
#include "math_constants.h"

// csr data format
uint16 *d_ref, *d_qry;
uint16 *d_ref_data, *d_qry_data;
int *   d_ref_indptr, *d_qry_indptr, *d_ref_indices, *d_qry_indices;
uint16* d_ref_dense;

vector<float>  h_labels;
uint32         ref_height, ref_width, qry_height, qry_width;
uint32         ct_num;
vector<uint32> h_ctidx;
vector<uint32> h_ctdiff, h_ctdidx;
size_t         pitchref;
size_t         pitchqry;

uint32 *d_gene_idx, *d_cell_idx;
uint16 *d_ref_lines, *d_qry_line;
float * d_ref_rank, *d_qry_rank;
float*  d_score;

int shared_mem_per_block;

Slice slice;

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// unit is MB
uint32 getUsedMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (total - free) / 1024 / 1024;
}

uint32 getFreeMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free / 1024 / 1024;
}

bool initGPU(const int gpuid)
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    if (gpuid < devicesCount)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, gpuid);
        if (deviceProperties.major >= 2 && deviceProperties.minor >= 0)
        {
            shared_mem_per_block = deviceProperties.sharedMemPerBlock;
            cudaSetDevice(gpuid);
            return true;
        }
    }
    return false;
}

bool destroy()
{
    cudaFree(d_ref_data);
    cudaFree(d_ref_indptr);
    cudaFree(d_ref_indices);
    cudaFree(d_qry_data);
    cudaFree(d_qry_indptr);
    cudaFree(d_qry_indices);

    cudaFree(d_gene_idx);
    cudaFree(d_cell_idx);
    cudaFree(d_qry_line);
    cudaFree(d_ref_lines);
    cudaFree(d_ref_dense);
    cudaFree(d_qry_rank);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}

bool destroy_score()
{
    cudaFree(d_ref);
    cudaFree(d_qry);

    cudaFree(d_ref_data);
    cudaFree(d_ref_indptr);
    cudaFree(d_ref_indices);
    cudaFree(d_qry_data);
    cudaFree(d_qry_indptr);
    cudaFree(d_qry_indices);

    cudaFree(d_qry_rank);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}

__global__ void rankdata_pitch(uint16* dataIn, float* dataOut, const int datalen,
                               const size_t pitch, const int batch)
{  // no shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen * batch)
    {
        int     batchid      = tid / datalen;
        float   equl_cnt     = 0;
        float   less_cnt     = 1;
        uint16* dataIn_batch = ( uint16* )(( char* )dataIn + batchid * pitch);
        for (int i = 0; i < datalen; i++)
        {

            if (dataIn_batch[tid] > dataIn_batch[i])
            {
                less_cnt++;
            }
            else if (dataIn_batch[tid] == dataIn_batch[i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + (equl_cnt - 1) / 2;
    }
}

// for qry 1 line rank
__global__ void rankdata(uint16* dataIn, float* dataOut, const int datalen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen)
    {
        float equl_cnt = 0;
        float less_cnt = 1;
        for (int i = 0; i < datalen; i++)
        {
            if (dataIn[tid] > dataIn[i])
            {
                less_cnt++;
            }
            else if (dataIn[tid] == dataIn[i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + (equl_cnt - 1) / 2;
    }
}

__global__ void rankdata(float* dataIn, float* dataOut, const int datalen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen)
    {
        float equl_cnt = 0;
        float less_cnt = 1;
        for (int i = 0; i < datalen; i++)
        {
            if (dataIn[tid] > dataIn[i])
            {
                less_cnt++;
            }
            else if (dataIn[tid] == dataIn[i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + (equl_cnt - 1) / 2;
    }
}

// for ref data, including specific rows and total columns
__global__ void csr2dense(const uint16* data, const int* indptr, const int* indices,
                          const uint32* cells, const size_t cell_num, const uint32 width,
                          uint16* res)
{
    int bid     = blockIdx.x;
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int cell_line = cells[bid];
    int start     = indptr[cell_line];
    int end       = indptr[cell_line + 1];

    for (int i = start + tid; i < end; i += threads)
    {
        res[size_t(bid) * width + indices[i]] = data[i];
    }
}

// for ref data, including specific rows and total columns
__global__ void csr2dense(const uint16* data, const int* indptr, const int* indices,
                          const uint32 cell_start, const uint32 width, uint16* res)
{
    int bid     = blockIdx.x;
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int cell_line = cell_start + bid;
    int start     = indptr[cell_line];
    int end       = indptr[cell_line + 1];

    for (int i = start + tid; i < end; i += threads)
    {
        res[size_t(bid) * width + indices[i]] = data[i];
    }
}

// for qry data, including single row and specific columns
__global__ void csr2dense(const uint16* data, const int* indptr, const int* indices,
                          const uint32* genes, const size_t gene_num, const int cell_line,
                          const uint32 width, uint16* res)
{
    // int bid = blockIdx.x;
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int start = indptr[cell_line];
    int end   = indptr[cell_line + 1];

    // csr2dense
    for (int i = start + tid; i < end; i += threads)
    {
        res[width + indices[i]] = data[i];
    }
    __syncthreads();

    // fitler by columns
    for (int i = tid; i < gene_num; i += threads)
    {
        res[i] = res[width + genes[i]];
    }
}

// for total data, including all rows and columns
__global__ void csr2dense(const uint16* data, const int* indptr, const int* indices,
                          const int height, const int width, uint16* res)
{
    int bid     = blockIdx.x;
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int start = indptr[bid];
    int end   = indptr[bid + 1];

    for (int i = start + tid; i < end; i += threads)
    {
        res[size_t(bid) * width + indices[i]] = data[i];
    }
}

bool copyin_score(InputData& rawdata)
{
    ref_height = rawdata.ref_height;
    ref_width  = rawdata.ref_width;
    qry_height = rawdata.qry_height;
    qry_width  = rawdata.qry_width;
    ct_num     = rawdata.ct_num;

    size_t estimated_mem = 0;
    estimated_mem += rawdata.ref_data.size() * sizeof(uint16);
    estimated_mem += rawdata.ref_indptr.size() * sizeof(int);
    estimated_mem += rawdata.ref_indices.size() * sizeof(int);

    estimated_mem += rawdata.qry_data.size() * sizeof(uint16);
    estimated_mem += rawdata.qry_indptr.size() * sizeof(int);
    estimated_mem += rawdata.qry_indices.size() * sizeof(int);

    estimated_mem += size_t(ref_height) * (1024 * sizeof(float));
    estimated_mem /= 1024 * 1024;
    estimated_mem += 255;  // system memory
    auto free_mem = getFreeMem();
    if ((estimated_mem + 500) > free_mem)
    {
        cerr << "Need gpu memory(MB): " << estimated_mem + 500
             << " less than free memory(MB): " << free_mem << endl;
        return false;
    }

    CHECK(cudaMalloc(( void** )&d_ref_data, rawdata.ref_data.size() * sizeof(uint16)));
    CHECK(cudaMemcpy(d_ref_data, rawdata.ref_data.data(),
                     rawdata.ref_data.size() * sizeof(uint16), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_ref_indptr, rawdata.ref_indptr.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_ref_indptr, rawdata.ref_indptr.data(),
                     rawdata.ref_indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_ref_indices, rawdata.ref_indices.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_ref_indices, rawdata.ref_indices.data(),
                     rawdata.ref_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(( void** )&d_qry_data, rawdata.qry_data.size() * sizeof(uint16)));
    CHECK(cudaMemcpy(d_qry_data, rawdata.qry_data.data(),
                     rawdata.qry_data.size() * sizeof(uint16), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_qry_indptr, rawdata.qry_indptr.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_qry_indptr, rawdata.qry_indptr.data(),
                     rawdata.qry_indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_qry_indices, rawdata.qry_indices.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_qry_indices, rawdata.qry_indices.data(),
                     rawdata.qry_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    h_ctidx = rawdata.ctidx;

    CHECK(cudaMalloc(( void** )&d_score, 1024 * ref_height * sizeof(float)));

    // Big memory mode
    size_t need_mem = (size_t(ref_height) * ref_width + size_t(qry_height) * qry_width)
                      * (sizeof(uint16) + sizeof(float)) / 1024 / 1024;
    free_mem -= estimated_mem;
    cout << "free GPU device mem(MB): " << free_mem << " total need mem(MB): " << need_mem << endl;
    if (free_mem > need_mem)
    {
        cout<<"Load total ref and qry data into GPU device memory"<<endl;
        slice.on        = false;
        slice.ref_rows  = ref_height;
        slice.ref_steps = 1;
        slice.qry_rows  = qry_height;
        slice.qry_steps = 1;
        CHECK(cudaMalloc(( void** )&d_ref,
                         size_t(ref_height) * ref_width * sizeof(uint16)));
        CHECK(cudaMalloc(( void** )&d_qry,
                         size_t(qry_height) * qry_width * sizeof(uint16)));
        CHECK(cudaMalloc(( void** )&d_ref_rank,
                         size_t(ref_height) * ref_width * sizeof(float)));
        CHECK(cudaMalloc(( void** )&d_qry_rank,
                         size_t(qry_height) * qry_width * sizeof(float)));

        estimated_mem += need_mem;
    }
    else
    {
        slice.on = true;
        // Prioritize meet ref matrix
        size_t ref_need_mem = size_t(ref_height) * ref_width
                              * (sizeof(uint16) + sizeof(float)) / 1000 / 1000;
        if (free_mem > ref_need_mem)
        {

            slice.ref_rows  = ref_height;
            slice.ref_steps = 1;
            cout<<"Load total ref data into GPU device memory, size(HxW): "<<slice.ref_rows << "x"<<ref_width<<endl;
            CHECK(cudaMalloc(( void** )&d_ref,
                             size_t(slice.ref_rows) * ref_width * sizeof(uint16)));
            CHECK(cudaMalloc(( void** )&d_ref_rank,
                             size_t(slice.ref_rows) * ref_width * sizeof(float)));

            free_mem -= ref_need_mem;
            slice.qry_rows =
                free_mem * 1000 * 1000 / (sizeof(uint16) + sizeof(float)) / qry_width;
            slice.qry_steps = (qry_height - 1) / slice.qry_rows + 1;
            cout<<"Load sub qry data into GPU device memory, size(HxW): "<<slice.qry_rows << "x"<<qry_width<<endl;
            CHECK(cudaMalloc(( void** )&d_qry,
                             size_t(slice.qry_rows) * qry_width * sizeof(uint16)));
            CHECK(cudaMalloc(( void** )&d_qry_rank,
                             size_t(slice.qry_rows) * qry_width * sizeof(float)));
        }
        else
        {
            size_t total_rows =
                free_mem * 1000 * 1000 / (sizeof(uint16) + sizeof(float)) / qry_width;
            size_t ratio = float(ref_height + qry_height) / total_rows + 0.5;
            slice.ref_rows  = ref_height / ratio + 1;
            slice.ref_steps = (ref_height - 1) / slice.ref_rows + 1;
            cout<<"Load sub ref data into GPU device memory, size(HxW): "<<slice.ref_rows << "x"<<ref_width<<endl;
            CHECK(cudaMalloc(( void** )&d_ref,
                             size_t(slice.ref_rows) * ref_width * sizeof(uint16)));
            CHECK(cudaMalloc(( void** )&d_ref_rank,
                             size_t(slice.ref_rows) * ref_width * sizeof(float)));

            slice.qry_rows  = qry_height / ratio + 1;
            slice.qry_steps = (qry_height - 1) / slice.qry_rows + 1;
            cout<<"Load sub qry data into GPU device memory, size(HxW): "<<slice.qry_rows << "x"<<qry_width<<endl;
            CHECK(cudaMalloc(( void** )&d_qry,
                             size_t(slice.qry_rows) * qry_width * sizeof(uint16)));
            CHECK(cudaMalloc(( void** )&d_qry_rank,
                             size_t(slice.qry_rows) * qry_width * sizeof(float)));
        }
        estimated_mem += size_t(slice.ref_rows) * ref_width * (sizeof(uint16) + sizeof(float)) / 1024 / 1024;
        estimated_mem += size_t(slice.qry_rows) * qry_width * (sizeof(uint16) + sizeof(float)) / 1024 / 1024;
    }
    estimated_mem += 100; // for reserve
    std::cout << "score() used gpu mem(MB): " << estimated_mem << std::endl;
    return true;
}

bool copyin(InputData& rawdata)
{
    ref_height = rawdata.ref_height;
    ref_width  = rawdata.ref_width;
    qry_height = rawdata.qry_height;
    qry_width  = rawdata.qry_width;
    ct_num     = rawdata.ct_num;

    size_t estimated_mem = 0;
    estimated_mem += rawdata.ref_data.size() * sizeof(uint16);
    estimated_mem += rawdata.ref_indptr.size() * sizeof(int);
    estimated_mem += rawdata.ref_indices.size() * sizeof(int);

    estimated_mem += rawdata.qry_data.size() * sizeof(uint16);
    estimated_mem += rawdata.qry_indptr.size() * sizeof(int);
    estimated_mem += rawdata.qry_indices.size() * sizeof(int);

    estimated_mem += qry_width * (sizeof(uint16)*2 + sizeof(uint32) + sizeof(float));
    estimated_mem += ref_height * (sizeof(uint32) + sizeof(float));

    estimated_mem /= 1024 * 1024;
    estimated_mem += 255;  // system memory
    auto free_mem = getFreeMem();
    if ((estimated_mem + 500) > free_mem)
    {
        cerr << "Need gpu memory(MB): " << estimated_mem + 500
             << " less than free memory(MB): " << free_mem << endl;
        return false;
    }

    CHECK(cudaMalloc(( void** )&d_ref_data, rawdata.ref_data.size() * sizeof(uint16)));
    CHECK(cudaMemcpy(d_ref_data, rawdata.ref_data.data(),
                     rawdata.ref_data.size() * sizeof(uint16), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_ref_indptr, rawdata.ref_indptr.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_ref_indptr, rawdata.ref_indptr.data(),
                     rawdata.ref_indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_ref_indices, rawdata.ref_indices.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_ref_indices, rawdata.ref_indices.data(),
                     rawdata.ref_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(( void** )&d_qry_data, rawdata.qry_data.size() * sizeof(uint16)));
    CHECK(cudaMemcpy(d_qry_data, rawdata.qry_data.data(),
                     rawdata.qry_data.size() * sizeof(uint16), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_qry_indptr, rawdata.qry_indptr.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_qry_indptr, rawdata.qry_indptr.data(),
                     rawdata.qry_indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(( void** )&d_qry_indices, rawdata.qry_indices.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_qry_indices, rawdata.qry_indices.data(),
                     rawdata.qry_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    h_labels = rawdata.labels;

    h_ctidx = rawdata.ctidx;

    h_ctdiff = rawdata.ctdiff;
    h_ctdidx = rawdata.ctdidx;

    CHECK(cudaMalloc(( void** )&d_gene_idx, qry_width * sizeof(uint32)));
    CHECK(cudaMalloc(( void** )&d_cell_idx, ref_height * sizeof(uint32)));
    // double memory for cache
    CHECK(cudaMalloc(( void** )&d_qry_line, qry_width * sizeof(uint16) * 2));
    CHECK(cudaMalloc(( void** )&d_qry_rank, qry_width * sizeof(float)));
    CHECK(cudaMalloc(( void** )&d_score, ref_height * sizeof(float)));

    // Big memory mode
    free_mem -= estimated_mem;
    size_t need_mem = (size_t(ref_height) * ref_width)
                      * (sizeof(uint16) * 2 + sizeof(float)) / 1024 / 1024;
    cout << "free GPU device mem(MB): " << free_mem << " total need mem(MB): " << need_mem << endl;
    if (free_mem > need_mem)
    {
        cout<<"Load total ref and qry data into GPU device memory"<<endl;
        slice.on       = false;
        slice.ref_rows = ref_height;
        CHECK(cudaMalloc(( void** )&d_ref_dense,
                         size_t(ref_height) * ref_width * sizeof(uint16)));
        CHECK(cudaMalloc(( void** )&d_ref_lines,
                         size_t(ref_height) * ref_width * sizeof(uint16)));
        CHECK(cudaMalloc(( void** )&d_ref_rank,
                         size_t(ref_height) * ref_width * sizeof(float)));

        estimated_mem += need_mem;
    }
    else
    {
        slice.on = true;
        size_t total_rows =
            free_mem * 1000 * 1000 / (sizeof(uint16) * 2 + sizeof(float)) / ref_width;
        // cout << "total rows: " << total_rows << endl;

        slice.ref_rows  = total_rows + 1;
        slice.ref_steps = (ref_height - 1) / slice.ref_rows + 1;
        cout<<"Load sub ref data into GPU device memory, size(HxW): "<<slice.ref_rows << "x"<<ref_width<<endl;

        CHECK(cudaMalloc(( void** )&d_ref_dense,
                         size_t(slice.ref_rows) * ref_width * sizeof(uint16)));
        CHECK(cudaMalloc(( void** )&d_ref_lines,
                         size_t(slice.ref_rows) * ref_width * sizeof(uint16)));
        CHECK(cudaMalloc(( void** )&d_ref_rank,
                         size_t(slice.ref_rows) * ref_width * sizeof(float)));

        estimated_mem += size_t(slice.ref_rows) * ref_width * (sizeof(uint16) * 2 + sizeof(float)) / 1024 / 1024;
    }
    estimated_mem += 100; // for reserve
    std::cout << "finetune() used gpu mem(MB): " << estimated_mem << std::endl;
    return true;
}

__global__ void get_device_qry_line(uint32* gene_idx, uint16* qry, const uint32 len,
                                    const uint32 gene_len, uint16* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        res[tid] = qry[gene_idx[tid]];
    }
}

__global__ void get_device_ref_pitch(uint32* gene_idx, const uint32 gene_len,
                                     uint32* cell_idx, const uint32 cell_len, uint16* ref,
                                     const uint32 ref_width, const size_t ref_pitch,
                                     const uint32 ref_lines_width,
                                     const size_t ref_lines_pitch, uint16* ref_lines)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        uint16* row_head =
            ( uint16* )(( char* )ref + ( uint64 )(cell_idx[nx]) * ref_pitch);
        uint16* row_head_lines =
            ( uint16* )(( char* )ref_lines + ( uint64 )( nx )*ref_lines_pitch);
        row_head_lines[ny] = row_head[ref_width - gene_idx[ny] - 1];
    }
}

__global__ void get_device_ref_lines(const uint32* gene_idx, const uint32 gene_len,
                                     const uint32* cell_idx, const uint32 cell_len,
                                     uint16* ref, const uint32 ref_width,
                                     const uint64 ref_pitch, uint16* res)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (ny < gene_len)
    {
        uint16* row_head        = ( uint16* )(( char* )ref + cell_idx[nx] * ref_pitch);
        res[nx * gene_len + ny] = row_head[gene_idx[ny]];
    }
}

// only filter by columns
__global__ void get_device_ref_lines(const uint32* gene_idx, const uint32 gene_len,
                                     uint16* ref, const uint32 ref_width, uint16* res)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO: optimize for one block process one row
    if (ny < gene_len)
    {
        uint16* row_head        = ref + size_t(nx) * ref_width;
        res[nx * gene_len + ny] = row_head[gene_idx[ny]];
    }
}

__global__ void rankdata(uint16* qry, const uint32 len, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        float r = 1, s = 0;
        for (int i = 0; i < len; ++i)
        {
            if (qry[tid] == qry[i])
                s += 1;
            else if (qry[tid] > qry[i])
                r += 1;
        }
        res[tid] = r + float(s - 1) / 2;
    }
}
/*
    datain : ref
    dataout: ref_rank
    datalen: gene_len
    batch: cell_len
*/
__global__ void rankdata_batch(uint16* dataIn, float* dataOut, const int datalen,
                               const int batch)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen * batch)
    {
        int batchid  = tid / datalen;
        int equl_cnt = 0;
        int less_cnt = 1;
        for (int i = 0; i < datalen; i++)
        {

            if (dataIn[tid] > dataIn[batchid * datalen + i])
            {
                less_cnt++;
            }
            else if (dataIn[tid] == dataIn[batchid * datalen + i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + ( float )(equl_cnt - 1) / 2;
    }
}

// basic rankdata method using bincount
__global__ void rankdata_bin(uint16* qry, const uint32 cols, const uint32 rows,
                             float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows)
    {
        uint16* q = qry + tid * cols;
        float*  r = res + tid * cols;
        // travel for getting bin count
        uint16 bins[80] = { 0 };
        for (int i = 0; i < cols; ++i)
            bins[q[i]]++;

        // calculate real rank
        float ranks[80];
        float start = 0;
        for (int i = 0; i < 80; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[i] = start + (bins[i] + 1) * 0.5;
            start += bins[i];
        }

        // assign rank value
        for (int i = 0; i < cols; ++i)
            r[i] = ranks[q[i]];
    }
}

// using shared memory
__global__ void rankdata_bin2(uint16* qry, const uint32 cols, const uint32 rows,
                              float* res)
{
    __shared__ uint16 bins[80 * 64];
    __shared__ float  ranks[80 * 64];
    int               bid = threadIdx.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows)
    {
        for (int i = 0; i < 80; ++i)
        {
            bins[bid * 80 + i] = 0;
        }

        uint16* q = qry + tid * cols;
        float*  r = res + tid * cols;
        // travel for getting bin count
        for (int i = 0; i < cols; ++i)
            bins[bid * 80 + q[i]]++;

        // calculate real rank
        float start = 0;
        for (int i = 0; i < 80; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[bid * 80 + i] = start + (bins[bid * 80 + i] + 1) * 0.5;
            start += bins[bid * 80 + i];
        }

        // assign rank value
        for (int i = 0; i < cols; ++i)
            r[i] = ranks[bid * 80 + q[i]];
    }
}

// one block threads for each cell
__global__ void rankdata_bin3_pitch(uint16* qry, const size_t pitch, const uint32 cols,
                                    const uint32 rows, float* res)
{
    __shared__ int   bins[128];
    __shared__ float ranks[128];
    int              bid = blockIdx.x;
    int              tid = threadIdx.x;

    int step = cols / 64;
    if (cols % 64 != 0)
        step++;

    if (bid < rows)
    {
        bins[tid * 2]     = 0;
        bins[tid * 2 + 1] = 0;
        __syncthreads();
        uint16* q = ( uint16* )(( char* )qry + bid * pitch);
        // uint16* q = qry + bid*cols;
        float* r = res + bid * cols;
        // travel for getting bin count
        for (int i = tid * step; i < (tid + 1) * step; ++i)
        {
            if (i < cols)
                atomicAdd(&bins[q[i]], 1);
        }
        __syncthreads();

        // calculate real rank
        if (tid == 0)
        {
            int start = 0;
            for (int i = 0; i < 128; ++i)
            {
                // if (bins[i] == 0) continue;
                ranks[i] = start + (bins[i] + 1) * 0.5;
                start += bins[i];
            }
        }
        __syncthreads();

        // assign rank value
        for (int i = tid * step; i < (tid + 1) * step; ++i)
        {
            if (i < cols)
                r[i] = ranks[q[i]];
        }
    }
}

__global__ void rankdata_bin3(uint16* qry, const uint32 cols, const uint32 rows,
                              const uint64 max_uniq_gene, float* res)
{
    extern __shared__ int mem[];
    int*                  bins  = mem;
    float*                ranks = ( float* )&bins[max_uniq_gene];

    int bid     = blockIdx.x;
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int step = cols / threads;
    if (cols % threads != 0)
        step++;
    int step1 = max_uniq_gene / threads;
    if (max_uniq_gene % threads != 0)
        step1++;

    for (int i = 0; i < step1; ++i)
    {
        if ((tid * step1 + i) < max_uniq_gene)
        {
            bins[tid * step1 + i] = 0;
        }
    }
    __syncthreads();

    uint16* q = qry + bid * cols;
    float*  r = res + bid * cols;
    // travel for getting bin count
    for (int i = tid * step; i < (tid + 1) * step; ++i)
    {
        if (i < cols)
            atomicAdd(&bins[q[i]], 1);
    }
    __syncthreads();

    // calculate real rank
    if (tid == 0)
    {
        int start = 0;
        for (int i = 0; i < max_uniq_gene; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[i] = start + (bins[i] + 1) * 0.5;
            start += bins[i];
        }
    }
    __syncthreads();

    // assign rank value
    for (int i = tid * step; i < (tid + 1) * step; ++i)
    {
        if (i < cols)
            r[i] = ranks[q[i]];
    }
}

// 1xN vs MxN -> MX1
__global__ void spearman_reduce(float* qry, float* ref, const uint32 gene_num,
                                const uint32 cell_num, float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float mean = (gene_num + 1) / 2.0;
    int   step = gene_num / 128;
    if (gene_num % 128 != 0)
        step++;

    __shared__ float sumxy[128];
    __shared__ float sumxx[128];
    __shared__ float sumyy[128];

    sumxy[tid] = 0;
    sumxx[tid] = 0;
    sumyy[tid] = 0;

    for (int i = tid * step; i < (tid + 1) * step; ++i)
    {
        if (i < gene_num)
        {
            float x = qry[i] - mean;
            float y = ref[bid * gene_num + i] - mean;
            sumxy[tid] += x * y;
            sumxx[tid] += x * x;
            sumyy[tid] += y * y;
        }
    }
    __syncthreads();

    for (int stride = 1; stride < 128; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            sumxy[tid] += sumxy[tid + stride];
            sumxx[tid] += sumxx[tid + stride];
            sumyy[tid] += sumyy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float divisor = sqrt(sumxx[0] * sumyy[0]);
        if (divisor != 0)
            score[bid] = sumxy[0] / divisor;
        else

            score[bid] = CUDART_NAN_F;
    }
}

// KxN vs MxN -> MxK
__global__ void spearman(float* qry, float* ref, const uint32 gene_num,
                         const uint32 qry_cell_num, const uint32 ref_cell_num,
                         float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float mean = (gene_num + 1) / 2.0;
    int   step = gene_num / 128;
    if (gene_num % 128 != 0)
        step++;

    __shared__ float sumxy[128];
    __shared__ float sumxx[128];
    __shared__ float sumyy[128];

    for (int k = 0; k < qry_cell_num; ++k)
    {
        sumxy[tid] = 0;
        sumxx[tid] = 0;
        sumyy[tid] = 0;
        __syncthreads();

        for (int i = tid * step; i < (tid + 1) * step; ++i)
        {
            if (i < gene_num)
            {
                float x = qry[k * gene_num + i] - mean;
                float y = ref[bid * gene_num + i] - mean;
                sumxy[tid] += x * y;
                sumxx[tid] += x * x;
                sumyy[tid] += y * y;
            }
        }
        __syncthreads();

        for (int stride = 1; stride < 128; stride *= 2)
        {
            if ((tid % (2 * stride)) == 0)
            {
                sumxy[tid] += sumxy[tid + stride];
                sumxx[tid] += sumxx[tid + stride];
                sumyy[tid] += sumyy[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            float divisor = sqrt(sumxx[0] * sumyy[0]);
            if (divisor != 0)
                score[k * ref_cell_num + bid] = sumxy[0] / divisor;
            else
                score[k * ref_cell_num + bid] = CUDART_NAN_F;
        }
        __syncthreads();
    }
}

__global__ void spearman(float* qry, const uint32 start, float* ref,
                         const uint32 gene_num, const uint32 qry_cell_num,
                         const uint32 ref_cell_num, float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (start + tid >= qry_cell_num)
        return;

    float mean = (gene_num + 1) / 2.0;

    float sumxy = 0, sumxx = 0, sumyy = 0;

    for (int i = 0; i < gene_num; ++i)
    {
        float x = qry[(start + tid) * gene_num + i] - mean;
        float y = ref[bid * gene_num + i] - mean;
        sumxy += x * y;
        sumxx += x * x;
        sumyy += y * y;
    }

    float divisor = sqrt(sumxx * sumyy);
    if (divisor != 0)
        score[tid * ref_cell_num + bid] = sumxy / divisor;
    else
        score[tid * ref_cell_num + bid] = CUDART_NAN_F;
}

float percentile(vector<float> arr, int len, float p)
{
    if (len <= 1)
        return arr.front();

    float res;
    std::sort(arr.begin(), arr.begin() + len);

    vector<float> index;
    float         step = 1.0 / (len - 1);
    for (int i = 0; i < len; ++i)
        index.push_back(i * step);

    if (p <= index.front())
    {
        res = arr[0];
    }
    else if (index.back() <= p)
    {
        res = arr[len - 1];
    }
    else
    {
        auto  it        = lower_bound(index.begin(), index.end(), p);
        float prevIndex = *(it - 1);
        float prevValue = arr.at(it - index.begin() - 1);
        float nextValue = arr.at(it - index.begin());
        // linear interpolation
        res = (p - prevIndex) * (nextValue - prevValue) / step + prevValue;
    }
    return res;
}

inline bool bincount(const uint64 max_uniq_gene)
{
    // TODO: Get total amount of shared memory per block from device
    return max_uniq_gene <= min(1024, shared_mem_per_block / 8);
}

vector<int> get_label(InputData& rawdata, const uint64 max_uniq_gene, int cores)
{
    vector<float> res_scores(qry_height * ct_num, 0);
    vector<int>   res_labels(qry_height, 0);

    bool ref_only_once = (slice.ref_steps == 1);
    for (int q = 0; q < slice.qry_steps; ++q)
    {
        auto q_start = q * slice.qry_rows;
        auto q_end   = min((q + 1) * slice.qry_rows, qry_height);
        auto q_len   = q_end - q_start;
        // cout<<"qry: "<<q_start<<" "<<q_end<<" "<<q_len<<endl;

        CHECK(cudaMemset(d_qry, 0, size_t(slice.qry_rows) * qry_width * sizeof(uint16)));
        csr2dense<<<q_len, 1024>>>(d_qry_data, d_qry_indptr, d_qry_indices, q_start,
                                   qry_width, d_qry);
        CHECK(cudaGetLastError());

        if (bincount(max_uniq_gene))
        {
            // Using bincount method
            int shared_mem = max_uniq_gene * (sizeof(int) + sizeof(float));
            rankdata_bin3<<<slice.qry_rows, 64, shared_mem>>>(
                d_qry, qry_width, slice.qry_rows, max_uniq_gene, d_qry_rank);
            CHECK(cudaGetLastError());
        }
        else
        {
            rankdata_batch<<<(slice.qry_rows * qry_width - 1) / 512 + 1, 512>>>(
                d_qry, d_qry_rank, qry_width, slice.qry_rows);
            CHECK(cudaGetLastError());
        }

        vector<vector<float>> remaining(slice.qry_rows, vector<float>{});
        int                   start_ct  = 0;
        size_t                total_len = 0;
        vector<size_t>        curr_ct;

        for (int r = 0; r < slice.ref_steps; ++r)
        {
            auto r_start = r * slice.ref_rows;
            auto r_end   = min((r + 1) * slice.ref_rows, ref_height);
            auto r_len   = r_end - r_start;

            // cout<<"ref: "<<r_start<<" "<<r_end<<" "<<r_len<<endl;

            if (ref_only_once && q == 0)
            {
                CHECK(cudaMemset(d_ref, 0,
                                 size_t(slice.ref_rows) * ref_width * sizeof(uint16)));
                
                csr2dense<<<r_len, 1024>>> (d_ref_data, d_ref_indptr, d_ref_indices, r_start, 
                    ref_width, d_ref);
                CHECK(cudaGetLastError());

                if (bincount(max_uniq_gene))
                {
                    // Using bincount method
                    int shared_mem = max_uniq_gene * (sizeof(int) + sizeof(float));
                    rankdata_bin3<<<slice.ref_rows, 64, shared_mem>>>(
                        d_ref, ref_width, slice.ref_rows, max_uniq_gene, d_ref_rank);
                    CHECK(cudaGetLastError());
                }
                else
                {
                    rankdata_batch<<<(slice.ref_rows * ref_width - 1) / 512 + 1, 512>>>(
                        d_ref, d_ref_rank, ref_width, slice.ref_rows);
                    CHECK(cudaGetLastError());
                }
            }

            int j;
            for (j = start_ct; j < ct_num; ++j)
            {
                size_t start = h_ctidx[j * 2];
                size_t len   = h_ctidx[j * 2 + 1];
                total_len += len;
                if (total_len > slice.ref_rows + remaining[0].size())
                {
                    break;
                }
                curr_ct.push_back(len);
            }

            vector<float> h_score(1024 * slice.ref_rows, 0);
            for (int line = 0; line < slice.qry_rows; line += 1024)
            {
                spearman<<<slice.ref_rows, 1024>>>(d_qry_rank, line, d_ref_rank,
                                                   qry_width, slice.qry_rows,
                                                   slice.ref_rows, d_score);
                CHECK(cudaGetLastError());
                CHECK(cudaMemcpy(h_score.data(), d_score,
                                 1024 * slice.ref_rows * sizeof(float),
                                 cudaMemcpyDeviceToHost));
                for (int i = 0; i < 1024; ++i)
                {
                    if (line + i >= q_len)
                        break;

                    remaining[line + i].insert(
                        remaining[line + i].end(), h_score.begin() + i * slice.ref_rows,
                        h_score.begin() + (i + 1) * slice.ref_rows);
                }

                // Calculate real celltypes in current ref data
                for (int i = 0; i < 1024; ++i)
                {
                    if (line + i >= q_len)
                        break;

                    auto&  score         = remaining[line + i];
                    size_t tmp_total_len = 0, start = 0;
                    for (int j = 0; j < curr_ct.size(); ++j)
                    {
                        size_t len = curr_ct[j];
                        tmp_total_len += len;

                        vector<float> tmp(score.begin() + start,
                                          score.begin() + tmp_total_len);
                        float         p = percentile(tmp, len, 0.8);
                        res_scores[(q_start + line + i) * ct_num + start_ct + j] = p;
                        start += len;
                    }
                    score.assign(score.begin() + tmp_total_len, score.end());
                    score.shrink_to_fit();  // release memory in time
                }
            }

            start_ct += curr_ct.size();
            curr_ct.clear();
            total_len = 0;
        }
    }
    // calculate max score as first label
    for (int i = 0; i < res_scores.size() / ct_num; ++i)
    {
        const auto& scores = res_scores.begin() + i * ct_num;
        auto        ele    = std::minmax_element(scores, scores + ct_num);
        res_labels[i]      = (ele.second - scores);
        float thre         = *ele.second - 0.05;  // max-0.05
        for (int j = 0; j < ct_num; ++j)
        {
            if (*(scores + j) >= thre)
            {
                rawdata.labels[i * ct_num + j] = 1;
            }
        }
    }

    return res_labels;
}

vector<uint32> finetune_round(int qry_line, vector<uint32> top_labels,
                              const uint64 max_uniq_gene)
{
    set<uint32> uniq_genes;
    int         gene_thre = round(500 * pow((2 / 3.0), log2(top_labels.size())));

    for (auto& i : top_labels)
    {
        for (auto& j : top_labels)
        {
            if (i == j)
                continue;
            int pos = h_ctdidx[(i * ct_num + j) * 2];
            int len = h_ctdidx[(i * ct_num + j) * 2 + 1];
            if (len > gene_thre)
                len = gene_thre;
            uniq_genes.insert(h_ctdiff.begin() + pos, h_ctdiff.begin() + pos + len);
        }
    }
    //cout<<"uniq genes: "<<uniq_genes.size()<<endl;
    if (uniq_genes.size() < 20)
        return { top_labels.front() };

    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());

    // transfer qry data from cpu to gpu
    CHECK(cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size() * sizeof(uint32),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemset(d_qry_line, 0, qry_width * 2 * sizeof(uint16)));

    csr2dense<<<1, 1024>>>(d_qry_data, d_qry_indptr, d_qry_indices, d_gene_idx,
                           h_gene_idx.size(), qry_line, qry_width, d_qry_line);
    CHECK(cudaGetLastError());

    // get rank of qry data
    rankdata<<<(h_gene_idx.size() - 1) / 1024 + 1, 1024>>>(d_qry_line, d_qry_rank,
                                                           h_gene_idx.size());
    CHECK(cudaGetLastError());

    vector<pair<size_t, size_t>> temp;
    size_t                       total_len = 0;
    for (auto& label : top_labels)
    {
        uint32 pos = h_ctidx[label * 2];
        uint32 len = h_ctidx[label * 2 + 1];
        if (temp.empty() || (temp.back().first + temp.back().second) != pos)
        {
            temp.push_back({ pos, len });
            total_len += len;
        }
        else
        {
            temp.back().second += len;
            total_len += len;
        }
    }
    vector<uint32> h_cell_idx(total_len);
    total_len = 0;
    for (auto& tmp : temp)
    {
        size_t pos = tmp.first;
        size_t len = tmp.second;
        std::iota(h_cell_idx.begin() + total_len, h_cell_idx.begin() + total_len + len,
                  pos);
        total_len += len;
    }

    CHECK(cudaMemcpy(d_cell_idx, h_cell_idx.data(), h_cell_idx.size() * sizeof(uint32),
                     cudaMemcpyHostToDevice));

    vector<float> h_score;
    h_score.resize(total_len, 0);

    uint32 need_rows = h_cell_idx.size();
    int    steps     = (need_rows - 1) / slice.ref_rows + 1;
    // cout<<"need rows: "<<need_rows<<" "<<slice.ref_rows<<endl;
    for (int r = 0; r < steps; ++r)
    {
        auto r_start = r * slice.ref_rows;
        auto r_end   = min((r + 1) * slice.ref_rows, need_rows);
        auto r_len   = r_end - r_start;

        // cout<<"ref: "<<r_start<<" "<<r_end<<" "<<r_len<<endl;

        CHECK(cudaMemset(d_ref_dense, 0, size_t(ref_width) * r_len * sizeof(uint16)));
        csr2dense<<<r_len, 1024>>>(d_ref_data, d_ref_indptr, d_ref_indices,
                                   d_cell_idx + r_start, r_len, ref_width, d_ref_dense);
        CHECK(cudaGetLastError());

        dim3 blockDim(1, 512);
        dim3 gridDim(r_len, (h_gene_idx.size() - 1) / 512 + 1);
        get_device_ref_lines<<<gridDim, blockDim>>>(d_gene_idx, h_gene_idx.size(),
                                                    d_ref_dense, ref_width, d_ref_lines);
        CHECK(cudaGetLastError());

        if (bincount(max_uniq_gene))
        {
            int shared_mem = max_uniq_gene * (sizeof(int) + sizeof(float));
            rankdata_bin3<<<r_len, 64, shared_mem>>>(d_ref_lines, h_gene_idx.size(),
                                                     r_len, max_uniq_gene, d_ref_rank);
            CHECK(cudaGetLastError());
        }
        else
        {
            rankdata_batch<<<(r_len * h_gene_idx.size() - 1) / 512 + 1, 512>>>(
                d_ref_lines, d_ref_rank, h_gene_idx.size(), r_len);
            CHECK(cudaGetLastError());
        }

        // spearman
        spearman_reduce<<<r_len, 128>>>(d_qry_rank, d_ref_rank, h_gene_idx.size(), r_len,
                                        d_score);
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(h_score.data() + r_start, d_score, r_len * sizeof(float),
                         cudaMemcpyDeviceToHost));
    }
    uint32 start = 0;
    total_len    = 0;
    vector<float> scores;
    for (auto& label : top_labels)
    {
        uint32 len = h_ctidx[label * 2 + 1];
        total_len += len;

        vector<float> tmp(h_score.begin() + start, h_score.begin() + total_len);
        float         score = percentile(tmp, len, 0.8);
        scores.push_back(score);
        start += len;
    }

    auto  ele  = std::minmax_element(scores.begin(), scores.end());
    float thre = *ele.second - 0.05;
    if (std::isnan(*ele.second))
    {
        //cerr << "Got score 'nan'" << endl;
        return {top_labels.front()};
        //exit(-1);
    }
    vector<uint32> res;
    for (uint32 i = 0; i < scores.size(); ++i)
    {
        if (scores[i] <= *ele.first || scores[i] < thre)
            continue;
        else
            res.push_back(top_labels[i]);
    }
    if (res.empty())
        res.push_back(top_labels.front());

    return res;
}
vector<uint32> cufinetune(const uint64 max_uniq_gene)
{
    Timer timer("s");

    vector<uint32> res;
    // process each cell
    for (int i = 0; i < qry_height; ++i)
    {
        vector<uint32> top_labels;

        uint32 start = i * ct_num;
        for (int pos = 0; pos < ct_num; ++pos)
        {
            if (h_labels.at(start + pos) != 0)
                top_labels.push_back(pos);
        }

        //cout<<"cell: "<<i<<" "<<top_labels.size()<<endl;
        while (top_labels.size() > 1)
        {
            top_labels = finetune_round(i, top_labels, max_uniq_gene);
        }

        // TODO: why top labels can be empty
        if (!top_labels.empty())
            res.push_back(top_labels.front());
        else
            res.push_back(0);
        if (i != 0 && i % 1000 == 0)
        {
            cout << "processed " << i << " cells cost time(s): " << timer.toc() << endl;
        }
    }

    return res;
}
