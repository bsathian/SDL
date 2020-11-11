#ifndef Hit_h
#define Hit_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <cmath>
#include <vector>

//#include "PrintUtil.h"
#include "Module.cuh"

namespace SDL
{
    struct hits
    {
        unsigned int *nHits; //single number
        unsigned int *n2SHits;
        float *xs;
        float *ys;
        float *zs;

        unsigned int* moduleIndices;
        unsigned int* idxs;
        
        float *rts;
        float* phis;

        int *edge2SMap;
        float *highEdgeXs;
        float *highEdgeYs;
        float *lowEdgeXs;
        float *lowEdgeYs;
        
        hits();
        void freeMemory();
        ~hits();

    };

    void createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int maxHits, unsigned int max2SHits);
    void addHitToMemory(struct hits& hitsInGPU,struct modules& modulesInGPU,float x, float y, float z, unsigned int detId, unsigned int idxInNtuple);
    CUDA_HOSTDEV inline float phi(float x, float y, float z);
    CUDA_HOSTDEV inline float ATan2(float y, float x);
    CUDA_HOSTDEV float phi_mpi_pi(float phi);
    CUDA_HOSTDEV float deltaPhi(float x1, float y1, float z1, float x2, float y2, float z2);
    CUDA_HOSTDEV float deltaPhiChange(float x1, float y1, float z1, float x2, float y2, float z2);
    void getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow);

    void printHit(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex);
}
#endif

