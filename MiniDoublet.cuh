#ifndef MiniDoublet_h
#define MiniDoublet_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <array>
#include <tuple>
#include <math.h>
#include "Constants.h"
#include "Algo.h"
#include "PrintUtil.h"
#include "MathUtil.cuh"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Hit.cuh"
#include "Module.cuh"
namespace SDL
{
    class Module;
    class Hit;
    class Segment;
}

namespace SDL
{
    class MiniDoublet
    {
        private:

            // TODO: Rename lower and upper as inner and outer

            // Lower is always the one closer to the beam position
            Hit* lowerHitPtr_;

            // Upper is always the one further away from the beam position
            Hit* upperHitPtr_;

            // Anchor hit is either Pixel hit (if available) or lower hit
            Hit* anchorHitPtr_;

            // Bits to flag whether this mini-doublet passes some algorithm
            int passAlgo_;

            // Some mini-doublet related reconstructon variables
            Hit lowerShiftedHit_;
            Hit upperShiftedHit_;
            float dz_;
            float shiftedDz_;
            float dphi_;
            float dphi_noshift_;
            float dphichange_;
            float dphichange_noshift_;

            float drdz_; //drdz of lower module
            float slopeForHitShifting_;

            // Pointers of segments containing this Mini-doublet as inner mini doublet
            std::vector<Segment*> outwardSegmentPtrs;

            // Pointers of segments containing this Mini-doublet as outer mini doublet
            std::vector<Segment*> inwardSegmentPtrs;

            float miniCut_;

        public:
            MiniDoublet();
            MiniDoublet(const MiniDoublet&);
            MiniDoublet(Hit* lowerHit, Hit* upperHit);
            ~MiniDoublet();

            const std::vector<Segment*>& getListOfOutwardSegmentPtrs();
            const std::vector<Segment*>& getListOfInwardSegmentPtrs();

            void addOutwardSegmentPtr(Segment* sg);
            void addInwardSegmentPtr(Segment* sg);

            Hit* lowerHitPtr() const;
            Hit* upperHitPtr() const;
            Hit* anchorHitPtr() const;
            const int& getPassAlgo() const;
            CUDA_HOSTDEV const Hit& getLowerShiftedHit() const;
            CUDA_HOSTDEV const Hit& getUpperShiftedHit() const;
            __device__ __host__ const float& getDz() const;
            CUDA_HOSTDEV const float& getShiftedDz() const;
            CUDA_HOSTDEV const float& getDeltaPhi() const;
            CUDA_HOSTDEV const float& getDeltaPhiChange() const;
            CUDA_HOSTDEV const float& getDeltaPhiNoShift() const;
            CUDA_HOSTDEV const float& getDeltaPhiChangeNoShift() const;
            CUDA_HOSTDEV const float& getMiniCut() const;


            void setAnchorHit();
            __device__ __host__ void setLowerShiftedHit(float, float, float, int=-1);
            __device__ __host__ void setUpperShiftedHit(float, float, float, int=-1);
            __device__ __host__ void setDz(float);
            __device__ __host__ void setShiftedDz(float);
            __device__ __host__ void setDeltaPhi(float);
            __device__ __host__ void setDeltaPhiChange(float);
            __device__ __host__ void setDeltaPhiNoShift(float);
            __device__ __host__ void setDeltaPhiChangeNoShift(float);
            __device__ __host__ void setMiniCut(float);

            // return whether it passed the algorithm
            bool passesMiniDoubletAlgo(MDAlgo algo) const;

            // The function to run mini-doublet algorithm on a mini-doublet candidate
            CUDA_HOSTDEV void runMiniDoubletAlgo(MDAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accepts the mini-doublet
            CUDA_HOSTDEV void runMiniDoubletAllCombAlgo();

            // The default algorithms;
            CUDA_HOSTDEV void runMiniDoubletDefaultAlgo(SDL::LogLevel logLevel);
            CUDA_HOSTDEV void runMiniDoubletDefaultAlgoBarrel(SDL::LogLevel logLevel);
            CUDA_HOSTDEV void runMiniDoubletDefaultAlgoEndcap(SDL::LogLevel logLevel);

            bool isIdxMatched(const MiniDoublet&) const;
            bool isAnchorHitIdxMatched(const MiniDoublet&) const;

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const MiniDoublet& md);
            friend std::ostream& operator<<(std::ostream& out, const MiniDoublet* md);

            // The math for the threshold cut value to apply between hits for mini-doublet
            // The main idea is to be consistent with 1 GeV minimum pt
            // Some residual effects such as tilt, multiple scattering, beam spots are considered
            //static float dPhiThreshold(const Hit&, const Module&);
            __device__ __host__ float dPhiThreshold(const Hit& lowerHit, const Module& module, const float dPhi = 0, const float dz = 1);

            // The math for shifting the pixel hit up or down along the PS module orientation (deprecated)
            static float fabsdPhiPixelShift(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The math for shifting the strip hit up or down along the PS module orientation (deprecated)
            static float fabsdPhiStripShift(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The math for shifting the strip hit up or dow__device__ __host__ n along the PS module orientation, returns new x, y and z position
            CUDA_HOSTDEV void shiftStripHits(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, float* shiftedCoords, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The function to actually determine whether a pair of hits is a reco-ed mini doublet or not
            static bool isHitPairAMiniDoublet(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, MDAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // Condition that a module falls into "barrel"-logic of the mini-doublet algorithm
            static bool useBarrelLogic(const Module& lowerModule);

            // The function to determine transition region for inner most tilted layer
            static bool isNormalTiltedModules(const Module& lowerModule);

            // The function to determine transition region for inner most tilted layer (same as isNormalTiltedModules)
            __device__ __host__ static bool isTighterTiltedModules(const Module& lowerModule);

            // The function to determine gap
            __device__ __host__ static float moduleGapSize(const Module& lowerModule);

            //Function to set drdz so that we don't transport the tilted module map every time into the GPU, also
            //GPUs don't have STL yet, so we can't transport the map even if we wanted
            __device__ __host__ void setDrDz(float);

            __device__ __host__ void setLowerModuleSlope(float);


            inline void* operator new [](std::size_t len)
            {
                void *ptr;
                cudaMallocManaged(&ptr,len);
                cudaDeviceSynchronize();
                return ptr;
            }

            
    };
}

#endif
