#include "lanczosGPU.hpp"

#include <vector>
#include <exception>
#define M_PI 3.14159265358979323846

#include "cuda_runtime.h"
#include "cuda.h"

#define checkCudaError(statement) \
    {\
        cudaError_t error = statement; \
        if (error != cudaSuccess) { \
            const char *mes = cudaGetErrorString(error); \
            printf("%s", mes); \
            throw std::runtime_error(mes); \
        } \
    }
namespace
{
    using tjs_uint32 = GPU::tjs_uint32;
    
    template<int TTap>
    struct LanczosWeight
    {
        double operator()(double phase)
        {
            if (std::abs(phase) < DBL_EPSILON) return 1.0;
            if (std::abs(phase) >= (double)TTap) return 0.0;
            return std::sin(M_PI*phase)*std::sin(M_PI*phase / TTap) / (M_PI*M_PI*phase*phase / TTap);
        }
    };

    struct AxisParam
    {
        std::vector<int> start_;	// 開始インデックス
        std::vector<int> length_;	// 各要素長さ
        std::vector<double> weight_;
        std::vector<int> index_; // weight開始インデックス

        template<typename TWeightFunc>
        void calculateAxis(int srcstart, int srcend, int srclength, int dstlength, double tap, TWeightFunc& func);
    };

    // srclength = srcwidth
    // dstlength = dstwidth
    // srcstart = srcleft;
    // srcend = srcright;
    template<typename TWeightFunc>
    void AxisParam::calculateAxis(int srcstart, int srcend, int srclength, int dstlength, double tap, TWeightFunc& func)
    {
        start_.clear();
        start_.reserve(dstlength);
        length_.clear();
        length_.reserve(dstlength);
        index_.clear();
        index_.reserve(dstlength);
        int index = 0;
        if (srclength <= dstlength) { // 拡大
            double rangex = tap;
            int length = dstlength * (int)rangex * 2 + dstlength;
            weight_.reserve(length);
            for (int x = 0; x < dstlength; x++) {
                double cx = (x + 0.5)*(double)srclength / (double)dstlength + srcstart;
                int left = (int)std::floor(cx - rangex);
                int right = (int)std::floor(cx + rangex);
                if (left < srcstart) left = srcstart;
                if (right >= srcend) right = srcend;
                start_.push_back(left);
                int len = 0;
                for (int sx = left; sx < right; sx++) {
                    double dist = std::abs(sx + 0.5 - cx);
                    double weight = func(dist);
                    len++;
                    weight_.push_back(weight);
                }
                length_.push_back(len);
                index_.push_back(index);
                index += len;
            }
        }
        else { // 縮小
            double rangex = tap*(double)srclength / (double)dstlength;
            int length = srclength * (int)rangex * 2 + srclength;
            weight_.reserve(length);
            for (int x = 0; x < dstlength; x++) {
                double cx = (x + 0.5)*(double)srclength / (double)dstlength + srcstart;
                int left = (int)std::floor(cx - rangex);
                int right = (int)std::floor(cx + rangex);
                if (left < srcstart) left = srcstart;
                if (right >= srcend) right = srcend;
                start_.push_back(left);
                // 転送先座標での位置
                double delta = (double)dstlength / (double)srclength;
                double dx = (left + 0.5) * delta;
                int len = 0;
                for (int sx = left; sx < right; sx++) {
                    double dist = std::abs(dx - (x + 0.5));
                    double weight = func(dist);
                    dx += delta;
                    len++;
                    weight_.push_back(weight);
                }
                length_.push_back(len);
                index_.push_back(index);
                index += len;
            }
        }
    }

    __global__ void kernel_weightCopy(
        tjs_uint32 * const d_destbuf,
        const unsigned int destleft, const unsigned int desttop, const unsigned int destwidth, const unsigned int destheight,
        const unsigned int srcwidth, const tjs_uint32 * __restrict__  const d_srcbuf,
        const int * const d_startX, const int * const d_lengthX, const double * const d_weightX, const int * const d_indexX,
        const int * const d_startY, const int * const d_lengthY, const double * const d_weightY, const int * const d_indexY
        )
    {
        const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = threadIndex / destwidth;
        const int x = threadIndex - y * destwidth;
        if (y > destheight || x > destwidth) return;

        int wiy = d_indexY[y];
        const int top = d_startY[y];
        const int bottom = d_startY[y] + d_lengthY[y];
        const int left = d_startX[x];
        const int right = d_startX[x] + d_lengthX[x];
        double color_element[4] = { 0.0, 0.0, 0.0, 0.0 };
        double w_total = 0.0;
        for (int sy = top; sy < bottom; sy++) {
            int wix = d_indexX[x];
            for (int sx = left; sx < right; sx++) {
                const double weight = (d_weightX[wix]) * (d_weightY[wiy]);
                const tjs_uint32 color = d_srcbuf[sy * srcwidth + sx];
                color_element[0] += (color & 0xff) * weight;
                color_element[1] += ((color >> 8) & 0xff) * weight;
                color_element[2] += ((color >> 16) & 0xff) * weight;
                color_element[3] += ((color >> 24) & 0xff) * weight;
                ++wix;
                w_total += weight;
            }
            wiy++;
        }
        if (w_total != 0) {
            const double mul = 1.0 / w_total;
            color_element[0] *= mul;
            color_element[1] *= mul;
            color_element[2] *= mul;
            color_element[3] *= mul;
        }
        tjs_uint32 color = (tjs_uint32)((color_element[0] > 255) ? 255 : (color_element[0] < 0) ? 0 : color_element[0]);
        color += (tjs_uint32)((color_element[1] > 255) ? 255 : (color_element[1] < 0) ? 0 : color_element[1]) << 8;
        color += (tjs_uint32)((color_element[2] > 255) ? 255 : (color_element[2] < 0) ? 0 : color_element[2]) << 16;
        color += (tjs_uint32)((color_element[3] > 255) ? 255 : (color_element[3] < 0) ? 0 : color_element[3]) << 24;

        d_destbuf[y * destwidth + x] = color;
    }

    template<class T>
    struct DeviceBuffer
    {
        T *ptr;

        explicit DeviceBuffer(const unsigned int size) {
            checkCudaError(cudaMalloc(&ptr, sizeof(T) * size));
        }

        ~DeviceBuffer() {
            checkCudaError(cudaFree(ptr));
        }
    };
        
    struct DeviceAxisParam
    {
        int *start;
        int *length;
        double *weight;
        int *index;

        explicit DeviceAxisParam(const AxisParam&& param) {
            checkCudaError(cudaMalloc(&start, sizeof(int) * param.start_.size()));
            checkCudaError(cudaMalloc(&length, sizeof(int) * param.length_.size()));
            checkCudaError(cudaMalloc(&weight, sizeof(double) * param.weight_.size()));
            checkCudaError(cudaMalloc(&index, sizeof(int) * param.index_.size()));
            
            checkCudaError(cudaMemcpy(start, param.start_.data(), sizeof(int) * param.start_.size(), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(length, param.length_.data(), sizeof(int) * param.length_.size(), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(weight, param.weight_.data(), sizeof(double) * param.weight_.size(), cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(index, param.index_.data(), sizeof(int) * param.index_.size(), cudaMemcpyHostToDevice));
        }

        ~DeviceAxisParam() {
            checkCudaError(cudaFree(start));
            checkCudaError(cudaFree(length));
            checkCudaError(cudaFree(weight));
            checkCudaError(cudaFree(index));
        }
    };
}

template<int W>
void GPU::TVPLanczos(
    const int destpitch, tjs_uint32 * const destbuf,
    const unsigned int destleft, const unsigned int desttop, const unsigned int destwidth, const unsigned int destheight,
    const int srcpitch, const tjs_uint32 * const srcbuf,
    const unsigned int srcleft, const unsigned int srctop, const unsigned int srcwidth, const unsigned int srcheight) {

    // パラメータ導出
    LanczosWeight<W> weightfunc;
    AxisParam paramx, paramy;
    paramx.calculateAxis(srcleft, srcleft + srcwidth, srcwidth, destwidth, static_cast<double>(W), weightfunc);
    paramy.calculateAxis(srctop, srctop + srcheight, srcheight, destheight, static_cast<double>(W), weightfunc);

    // CPUからGPUへデータ転送
    DeviceAxisParam deviceParamX(std::move(paramx));
    DeviceAxisParam deviceParamY(std::move(paramy));

    DeviceBuffer<tjs_uint32> deviceDestBuf(destwidth * destheight);
    DeviceBuffer<tjs_uint32> deviceSrcBuf(srcwidth * srcheight);
    for (unsigned int y = srctop; y < srcheight; ++y) {
        checkCudaError(cudaMemcpy(deviceSrcBuf.ptr + (y - srctop) * srcwidth, srcbuf + srcleft + y * srcpitch / 4, sizeof(tjs_uint32) * srcwidth, cudaMemcpyHostToDevice));
    }

    // GPU設定検出
    int max_threads;
    if (cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("cuDeviceGetAttribute failed.");
    }
    
    // フィルタ処理実行
    const int threadNum = max_threads;
    const int blockNum = (destwidth * destheight + threadNum - 1) / threadNum;
    kernel_weightCopy<<<blockNum, threadNum>>>(
        deviceDestBuf.ptr,
        destleft, desttop, destwidth, destheight,
        srcwidth, deviceSrcBuf.ptr,
        deviceParamX.start, deviceParamX.length, deviceParamX.weight, deviceParamX.index,
        deviceParamY.start, deviceParamY.length, deviceParamY.weight, deviceParamY.index);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    // GPUからCPUへ結果を転送
    for (unsigned int y = desttop; y < destheight; ++y) {
        checkCudaError(cudaMemcpy(destbuf + destleft + y * destpitch / 4, deviceDestBuf.ptr + (y - desttop) * destwidth, sizeof(tjs_uint32) * destwidth, cudaMemcpyDeviceToHost));
    }
}

template void GPU::TVPLanczos<2>(
    const int destpitch, tjs_uint32 * const destbuf,
    const unsigned int destleft, const unsigned int desttop, const unsigned int destwidth, const unsigned int destheight,
    const int srcpitch, const tjs_uint32 * const srcbuf,
    const unsigned int srcleft, const unsigned int srctop, const unsigned int srcwidth, const unsigned int srcheight);
template void GPU::TVPLanczos<3>(
    const int destpitch, tjs_uint32 * const destbuf,
    const unsigned int destleft, const unsigned int desttop, const unsigned int destwidth, const unsigned int destheight,
    const int srcpitch, const tjs_uint32 * const srcbuf,
    const unsigned int srcleft, const unsigned int srctop, const unsigned int srcwidth, const unsigned int srcheight);
