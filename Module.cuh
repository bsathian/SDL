#ifndef Module_h
#define Module_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


#include <vector>
#include <iostream>

#include "PrintUtil.h"
#include "MiniDoublet.cuh"
#include "ModulePrimitive.cuh"
#include "Hit.cuh"

namespace SDL{
    class MiniDoublet;
    class Segment;
    class Triplet;
    class Tracklet;
    class Hit;
}

namespace SDL
{

    class Module
    {


        private:
    
            SDL::ModulePrimitive *moduleInfo;

            // vector of hit pointers
            std::vector<Hit*> hits_;

            // vector of mini doublet pointers
            std::vector<MiniDoublet*> miniDoublets_;

            // vector of segment pointers
            std::vector<Segment*> segments_;

            // vector of triplet pointers
            std::vector<Triplet*> triplets_;

            // vector of tracklet pointers
            std::vector<Tracklet*> tracklets_;


        public:
            enum SubDet
            {
                Barrel = 5,
                Endcap = 4
            };

            enum Side
            {
                NegZ = 1,
                PosZ = 2,
                Center = 3
            };

            enum ModuleType
            {
                PS,
                TwoS
            };
            enum ModuleLayerType
            {
                Pixel,
                Strip
            };

            // constructor/destructor
            Module();
            Module(unsigned int detId);
            Module(const Module&);
            Module(ModulePrimitive*);
            ~Module();

            // accessor functions
            CUDA_HOSTDEV const unsigned int& detId() const;
            CUDA_HOSTDEV const unsigned int& partnerDetId() const;
            CUDA_HOSTDEV const unsigned short& subdet() const;
            CUDA_HOSTDEV const unsigned short& side() const;
            CUDA_HOSTDEV const unsigned short& layer() const;
            CUDA_HOSTDEV const unsigned short& rod() const;
            CUDA_HOSTDEV const unsigned short& ring() const;
            CUDA_HOSTDEV const unsigned short& module() const;
            CUDA_HOSTDEV const unsigned short& isLower() const;
            CUDA_HOSTDEV const bool& isInverted() const;
            CUDA_HOSTDEV const ModuleType& moduleType() const;
            CUDA_HOSTDEV const ModuleLayerType& moduleLayerType() const;
            const std::vector<Hit*>& getHitPtrs() const;
            const std::vector<MiniDoublet*>& getMiniDoubletPtrs() const;
            const std::vector<Segment*>& getSegmentPtrs() const;
            const std::vector<Triplet*>& getTripletPtrs() const;
            const std::vector<Tracklet*>& getTrackletPtrs() const;
            const SDL::ModulePrimitive* modulePrimitive() const;

            // modifying the class content
            void setDetId(unsigned int);
            void addHit(Hit* hit);
            void addMiniDoublet(MiniDoublet* md);
            void addSegment(Segment* sg);
            void addTriplet(Triplet* tp);
            void addTracklet(Tracklet* tp);

            // static functions to parse detId
            static unsigned short parseSubdet(unsigned int);
            static unsigned short parseSide(unsigned int);
            static unsigned short parseLayer(unsigned int);
            static unsigned short parseRod(unsigned int);
            static unsigned short parseRing(unsigned int);
            static unsigned short parseModule(unsigned int);
            static unsigned short parseIsLower(unsigned int);
            static bool parseIsInverted(unsigned int);
            static unsigned int parsePartnerDetId(unsigned int);
            static ModuleType parseModuleType(unsigned int);
            static ModuleLayerType parseModuleLayerType(unsigned int);

            // printing
            friend std::ostream& operator<<(std::ostream& os, const Module& module);
            friend std::ostream& operator<<(std::ostream& os, const Module* module);

    };

}

#endif
