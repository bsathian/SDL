#ifndef Event_h
#define Event_h

#include <vector>
#include <list>
#include <map>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

#include "Module.cuh"
#include "Hit.cuh"
#include "MiniDoublet.cuh"
#include "cuda_profiler_api.h"

namespace SDL
{
    class Event
    {
    private:
        std::array<unsigned int, 6> n_hits_by_layer_barrel_;
        std::array<unsigned int, 5> n_hits_by_layer_endcap_;
        std::array<unsigned int, 6> n_minidoublets_by_layer_barrel_;
        std::array<unsigned int, 5> n_minidoublets_by_layer_endcap_;
        void incrementNumberOfMiniDoublets();

        //CUDA stuff
        struct hits hitsInGPU;
        struct miniDoublets mdsInGPU;
        int nHits;
        int nMiniDoublets;


    public:
        Event();
        ~Event();

        void addHitToEvent(float x, float y, float z, unsigned int detId); //call the appropriate hit function, then increment the counter here
        void addMiniDoubletsToEvent();
        void resetObjectsInModule();
        void createMiniDoublets();

        unsigned int getNumberOfHits();
        unsigned int getNumberOfHitsByLayer(unsigned int layer);
        unsigned int getNumberOfHitsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfHitsByLayerEndcap(unsigned int layer);

        unsigned int getNumberOfMiniDoublets();
        unsigned int getNumberOfMiniDoubletsByLayer(unsigned int layer);
        unsigned int getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer);
        unsigned int getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer);
    };

    //global stuff

    extern struct modules modulesInGPU;
    extern unsigned int nModules;
    void initModules(); //read from file and init

}

__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU);
extern const unsigned int N_MAX_MD_PER_MODULES;
extern const unsigned int N_MAX_HITS_PER_MODULE;



#endif
