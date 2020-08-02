#ifndef Module_h
#define Module_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <iostream>
#include <unordered_map>
#include "MiniDoublet.cuh"
#include "Hit.cuh"
#include "TiltedGeometry.h"
#include "EndcapGeometry.h"
#include "ModuleConnectionMap.h"

struct modules
{
    unsigned int* detIds;
    unsigned int* moduleMap;
    unsigned int* nConnectedModules;
    float* drdzs;
    float* slopes;
    unsigned int *nModules; //single number
    
    short* layers;
    short* rings;
    short* modules;
    short* rods;
    short* subdets;
    short* sides;
    
    CUDA_HOSTDEV bool isInverted(unsigned int index);
    CUDA_HOSTDEV bool isLower(unsigned int index);
    CUDA_HOSTDEV unsigned int partnerDetIdIndex(unsigned int index);
    CUDA_HOSTDEV ModuleType moduleType(unsigned int index);
    CUDA_HOSTDEV ModuleLayerType moduleLayerType(unsigned int index);

    int* hitRanges;
    int* mdRanges;
    //others will be added later

};
std::unordered_map <unsigned int,unsigned int> detIdToIndex;

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

//functions
void createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules);
void fillConnectedModuleArray(struct modules& modulesInGPU);
void loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules);
void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned int& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side);
void resetObjectRanges(struct modules& modulesInGPU, int nModules);
#endif

