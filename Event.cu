# include "Event.cuh"


N_MAX_HITS_PER_MODULE = 100;
N_MAX_MD_PER_MODULES = 100;
SDL::Event::Event()
{
    //reset the arrays
    for(int i = 0; i<6; i++)
    {
        n_hits_by_layer_barrel_[i] = 0;
        n_minidoublets_by_layer_barrel_[i] = 0;
        if(i<5)
        {
            n_hits_by_layer_endcap_[i] = 0;
            n_minidoublets_by_layer_endcap_[i] = 0;
        }
    }
}

SDL::Event::~Event()
{
}

void SDL::initModules()
{
    if(modulesInGPU.detIds == nullptr) //check for nullptr and create memory
    {
        loadModulesFromFile(modulesInGPU,nModules); //nModules gets filled here
    }
    resetObjectRanges(modulesInGPU,nModules);
}

void SDL::Event::resetObjectsInModule()
{
    resetObjectRanges(modulesInGPU,nModules);
}

void SDL::Event::addHitToEvent(float x, float y, float z, unsigned int detId)
{
    const int HIT_MAX = 1000000;
    const int HIT_2S_MAX = 100000;

    if(hitsInGPU.xs == nullptr)
    {
        createHitsInUnifiedMemory(hitsInGPU,HIT_MAX,HIT_2S_MAX);
    }
    //calls the addHitToMemory function
    addHitToMemory(hitsInGPU, modulesInGPU, x, y, z, detId);

    unsigned int moduleLayer = modulesInGPU.layers[(*detIdToIndex)[detId]];
    unsigned int subdet = modulesInGPU.subdets[(*detIdToIndex)[detId]];

    if(subdet == Barrel)
    {
        n_hits_by_layer_barrel_[moduleLayer]++;
    }
    else
    {
        n_hits_by_layer_endcap_[moduleLayer]++;
    }

}

void SDL::Event::addMiniDoubletsToEvent()
{
    for(unsigned int i = 0; i<nModules; i++)
    {
        modulesInGPU.mdRanges[i * 2] = i * N_MAX_MD_PER_MODULES;
        modulesInGPU.mdRanges[i * 2 + 1] = (i * N_MAX_MD_PER_MODULES) + mdsInGPU.nMDs[i];
     
        if(modulesInGPU.subdets[i] == Barrel)
        {
            n_minidoublets_by_layer_barrel_[modulesInGPU.layers[i] -1] ++;
        }
        else
        {
            n_minidoublets_by_layer_endcap_[modulesInGPU.layers[i] - 1] ++;
        }
    }
}

void SDL::Event::createMiniDoublets()
{
    if(mdsInGPU.hitIndices == nullptr)
    {
        createMDsInUnifiedMemory(mdsInGPU, N_MAX_MD_PER_MODULES, nModules);
    }
    unsigned int nLowerModules = *modulesInGPU.nLowerModules;
    dim3 nThreads(1,16,16);
    dim3 nBlocks((nLowerModules % nThreads.x == 0 ? nModules/nThreads.x : nModules/nThreads.x + 1),(N_MAX_HITS_PER_MODULE % nThreads.y == 0 ? N_MAX_HITS_PER_MODULE/nThreads.y : N_MAX_HITS_PER_MODULE/nThreads.y + 1), (N_MAX_HITS_PER_MODULE % nThreads.z == 0 ? N_MAX_HITS_PER_MODULE/nThreads.z : N_MAX_HITS_PER_MODULE/nThreads.z + 1));
    std::cout<<nBlocks.x<<" "<<nBlocks.y<<" "<<nBlocks.z<<" "<<std::endl;
    
    createMiniDoubletsInGPU<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU);

}


__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(lowerModuleArrayIdx > (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIdx = modulesInGPU.lowerModuleIndices[lowerModuleArrayIdx];
    int upperModuleIdx = modulesInGPU.partnerModuleIndex(lowerModuleIdx);
    int lowerHitIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int upperHitIdx = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIdx * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIdx * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIdx * 2 + 1] - modulesInGPU.hitRanges[upperModuleIdx * 2] + 1;
    //consider assigining a dummy computation function for these
    if(lowerHitIdx > nLowerHits) return;
    if(upperHitIdx > nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIdx * 2] + lowerHitIdx;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIdx * 2] + upperHitIdx;

    float dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIdx, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);
    if(success)
    {
        unsigned int mdModuleIdx = atomicAdd(&mdsInGPU.nMDs[lowerModuleIdx],1);
        unsigned int mdIdx = lowerModuleIdx * N_MAX_MD_PER_MODULES + mdModuleIdx;

        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIdx, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIdx);
    }
}

unsigned int SDL::Event::getNumberOfHits()
{
    unsigned int hits = 0;
    for(auto &it:n_hits_by_layer_barrel_)
    {
        hits += it;
    }
    for(auto& it:n_hits_by_layer_endcap_)
    {
        hits += it;
    }

    return hits;
}

unsigned int SDL::Event::getNumberOfHitsByLayer(unsigned int layer)
{
    if(layer == 6)
        return n_hits_by_layer_barrel_[layer];
    else
        return n_hits_by_layer_barrel_[layer] + n_hits_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfHitsByLayerBarrel(unsigned int layer)
{
    return n_hits_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfHitsByLayerEndcap(unsigned int layer)
{
    return n_hits_by_layer_endcap_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoublets()
{
     unsigned int miniDoublets = 0;
    for(auto &it:n_minidoublets_by_layer_barrel_)
    {
        miniDoublets += it;
    }
    for(auto &it:n_minidoublets_by_layer_endcap_)
    {
        miniDoublets += it;
    }

    return miniDoublets;
   
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayer(unsigned int layer)
{
     if(layer == 6)
        return n_minidoublets_by_layer_barrel_[layer];
    else
        return n_minidoublets_by_layer_barrel_[layer] + n_minidoublets_by_layer_endcap_[layer];   
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer)
{
    return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer)
{
    return n_minidoublets_by_layer_endcap_[layer];
}
