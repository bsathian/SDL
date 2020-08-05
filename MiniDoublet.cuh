#ifndef MiniDoublet_h
#define MiniDoublet_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <array>
#include <tuple>
#include <cmath>
#include "Algo.h"
#include "Constants.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Module.cuh"
#include "Hit.cuh"

struct miniDoublets
{
    unsigned int* hitIndices;
    unsigned int* moduleIndices;
    short* pixelModuleFlag;
    unsigned int* nMDs; //counter per module
    float* dphichanges;

    float* dzs; //will store drt if the module is endcap
    float*dphis;

    float* shiftedXs;
    float* shiftedYs;
    float* shiftedZs;
    float* noShiftedDzs; //if shifted module
    float* noShiftedDphis; //if shifted module
    float* noShiftedDphiChanges; //if shifted module

    ~miniDoublets();

};

void createMDsInUnifiedMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs);
//for successful MDs
void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float dphi, float dphichange, float shfitedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx);

CUDA_HOSTDEV float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex);
CUDA_HOSTDEV inline float isTighterTiltedModules(struct modules& modulesInGPU, unsigned int moduleIndex);
CUDA_HOSTDEV inline float moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex);

CUDA_HOSTDEV bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);

CUDA_HOSTDEV void shiftStripHits(float x, float y, float z, float drdz, float slope, float* shiftedCoords);

CUDA_HOSTDEV bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);
CUDA_HOSTDEV bool runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);
CUDA_HOSTDEV bool runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange);


#endif

