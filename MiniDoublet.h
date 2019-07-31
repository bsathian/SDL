#ifndef MiniDoublet_h
#define MiniDoublet_h

#include <array>
#include <tuple>

#include "Algo.h"
#include "Hit.h"
#include "Module.h"
#include "PrintUtil.h"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"

namespace SDL
{
    class Module;
    class Hit;
}

namespace SDL
{
    class MiniDoublet
    {
        private:

            // Lower is always the one closer to the beam position
            Hit* lowerHitPtr_;

            // Upper is always the one further away from the beam position
            Hit* upperHitPtr_;

            // Bits to flag whether this mini-doublet passes some algorithm
            int passAlgo_;

            // Some mini-doublet related variables
            float dz_;
            float shiftedDz_;

        public:
            MiniDoublet();
            MiniDoublet(const MiniDoublet&);
            MiniDoublet(Hit* lowerHit, Hit* upperHit);
            ~MiniDoublet();

            Hit* lowerHitPtr() const;
            Hit* upperHitPtr() const;
            const int& getPassAlgo() const;
            const float& getDz() const;
            const float& getShiftedDz() const;

            void setDz(float dz);
            void setShiftedDz(float dz);

            // return whether it passed the algorithm
            bool passesMiniDoubletAlgo(MDAlgo algo) const;

            // The function to run mini-doublet algorithm on a mini-doublet candidate
            void runMiniDoubletAlgo(MDAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accepts the mini-doublet
            void runMiniDoubletAllCombAlgo();

            // The default algorithms;
            void runMiniDoubletDefaultAlgo(SDL::LogLevel logLevel);
            void runMiniDoubletDefaultAlgoBarrel(SDL::LogLevel logLevel);
            void runMiniDoubletDefaultAlgoEndcap(SDL::LogLevel logLevel);

            bool isIdxMatched(const MiniDoublet&) const;

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const MiniDoublet& md);
            friend std::ostream& operator<<(std::ostream& out, const MiniDoublet* md);

            // The math for the threshold cut value to apply between hits for mini-doublet
            // The main idea is to be consistent with 1 GeV minimum pt
            // Some residual effects such as tilt, multiple scattering, beam spots are considered
            static float dPhiThreshold(const Hit&, const Module&);

            // The math for shifting the pixel hit up or down along the PS module orientation (deprecated)
            static float fabsdPhiPixelShift(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The math for shifting the strip hit up or down along the PS module orientation (deprecated)
            static float fabsdPhiStripShift(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The math for shifting the strip hit up or down along the PS module orientation, returns new x, y and z position
            static std::tuple<float, float, float> shiftStripHits(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The function to actually determine whether a pair of hits is a reco-ed mini doublet or not
            static bool isHitPairAMiniDoublet(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, MDAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // Condition that a module falls into "barrel"-logic of the mini-doublet algorithm
            static bool useBarrelLogic(const Module& lowerModule);

            // The function to determine transition region for inner most tilted layer
            static bool isNormalTiltedModules(const Module& lowerModule);

    };
}

#endif
