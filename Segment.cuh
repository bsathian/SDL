#ifndef Segment_h
#define Segment_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#include "Constants.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"

//CUDA MATH API
#include "math.h"
#include "Constants.h"

namespace SDL
{
    struct segments
    {
        unsigned int* mdIndices;
        unsigned int* innerLowerModuleIndices;
        unsigned int* outerLowerModuleIndices;
        unsigned int* innerMiniDoubletAnchorHitIndices;
        unsigned int* outerMiniDoubletAnchorHitIndices;
        
        unsigned int* nSegments; //number of segments per inner lower module

        //not so optional dudes
        float* dPhis;
        float* dPhiMins;
        float* dPhiMaxs;
        float* dPhiChanges;
        float* dPhiChangeMins;
        float* dPhiChangeMaxs;

        //extremely not so optional dudes for the pixel segments
        float* ptIn;
        float* ptErr;
        float* px;
        float* py;
        float* pz;
        float* etaErr;


        //optional dudes
        float* zIns;
        float* zOuts;
        float* rtIns;
        float* rtOuts;
        float* dAlphaInnerMDSegments;
        float* dAlphaOuterMDSegments;
        float* dAlphaInnerMDOuterMDs;

        float* zLo;
        float* zHi;
        float* rtLo;
        float* rtHi;
        float* sdCut;
        float* dAlphaInnerMDSegmentThreshold;
        float* dAlphaOuterMDSegmentThreshold;
        float* dAlphaInnerMDOuterMDThreshold;

        segments();
	void freeMemory();
    };

    void createSegmentsInUnifiedMemory(struct segments& segmentsInGPU, unsigned int maxSegments, unsigned int nModules, unsigned int maxPixelSegments);

    CUDA_DEV void dAlphaThreshold(float* dAlphaThresholdValues, struct hits& hitsInGPU, struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex);

    CUDA_DEV void addSegmentToMemory(struct segments& segmentsInGPU, unsigned int lowerMDIndex, unsigned int upperMDIndex, unsigned int innerLowerModuleIndex, unsigned int outerLowerModuleIndex, unsigned int innerMDAnchorHitIndex, unsigned int outerMDAnchorHitIndex, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int idx);

    void addPixelSegmentToMemory(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int innerMDIndex, unsigned int outerMDIndex, unsigned int pixelModuleIndex, unsigned int innerAnchorHitIndex, unsigned int outerAnchorHitIndex, float dPhiChange, float ptIn, float ptErr, float px, float py, float pz, float etaErr, unsigned int idx);


    CUDA_DEV bool runSegmentDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex);


    CUDA_DEV bool runSegmentDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment, float&
        dAlphaInnerMDOuterMD, float& zLo, float& zHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex);

    CUDA_DEV bool runSegmentDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, struct miniDoublets& mdsInGPU, unsigned int& innerLowerModuleIndex, unsigned int& outerLowerModuleIndex, unsigned int& innerMDIndex, unsigned int& outerMDIndex, float& zIn, float& zOut, float& rtIn, float& rtOut, float& dPhi, float& dPhiMin, float& dPhiMax, float& dPhiChange, float& dPhiChangeMin, float& dPhiChangeMax, float& dAlphaInnerMDSegment, float& dAlphaOuterMDSegment,
        float& rtLo, float& rtHi, float& sdCut, float& dAlphaInnerMDSegmentThreshold, float& dAlphaOuterMDSegmentThreshold, float& dAlphaInnerMDOuterMDThreshold, float&
        dAlphaInnerMDOuterMD, unsigned int& innerMiniDoubletAnchorHitIndex, unsigned int& outerMiniDoubletAnchorHitIndex);


    void printSegment(struct segments& segmentsInGPU, struct miniDoublets& mdsInGPU, struct hits& hitsInPGU, struct modules& modulesInGPU, unsigned int segmentIndex);

}

#endif
