#include "TrackCandidate.cuh"

void SDL::createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates, unsigned int nLowerModules)
{
    cudaMallocManaged(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * nLowerModules * sizeof(short));
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * nLowerModules * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidates, nLowerModules * sizeof(unsigned int));

#pragma omp parallel for
    for(size_t i = 0; i<nLowerModules;i++)
    {
        trackCandidatesInGPU.nTrackCandidates[i] = 0;
    }
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTwoTracklets(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 0;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTrackletToTriplet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTripletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 1;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTripletIndex, trackCandidateType))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTripletToTracklet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTripletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 2;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTripletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}

__device__ bool SDL::hasCommonSegment(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerObjectIndex, unsigned int outerObjectIndex, short trackCandidateType)
{
    unsigned int innerObjectOuterSegmentIndex, outerObjectInnerSegmentIndex;

    if(trackCandidateType == 0)
    {
        //2 tracklets
        innerObjectOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 1)
    {
        //T4T3
        innerObjectOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 2)
    {
        //T3T4
        innerObjectOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }

    return (innerObjectOuterSegmentIndex == outerObjectInnerSegmentIndex);
}
