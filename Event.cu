#include "Event.cuh"

//CUDA Kernel for Minidoublet creation
/*__global__ void createMiniDoubletsInGPU(SDL::MiniDoublet* mdCands, int n, SDL::MDAlgo algo)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = tid; i<n; i+= stride)
    {
        mdCands[i].runMiniDoubletAlgo(algo);
    }

}*/

__global__ void runMiniDoubletGPUAlgo(int moduleId,SDL::Hit** lowerHits, SDL::Hit** upperHits,int nLowerHits,int nUpperHits,SDL::MiniDoublet* mdsInGPU,int* mdMemoryCounter,SDL::MDAlgo algo)
{
    int MAX_MD_MODULE = 100 * 100; //max number of MD cands per module
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i<nLowerHits * nUpperHits;i+=stride)
    {
//        mdCands[moduleId * maxMDModule + tid] 
        SDL::MiniDoublet mdCand(lowerHits[i/nUpperHits],upperHits[i%nUpperHits]);
        mdCand.runMiniDoubletAlgo(algo);
        if(mdCand.passesMiniDoubletAlgo(algo))
        {
            //atomic here -possible point of failure!!!!!
            int idx = atomicAdd(mdMemoryCounter,1);
            mdsInGPU[idx] = mdCand; //this is gonna be expensive
        }
//        mdCands[moduleId * maxMDModule + tid].runMiniDoubletAlgo(algo);
        //write the atomic add here
    }
}

/*__global__ void createMiniDoubletsInGPU(int nModules, SDL::MiniDoublet* mdsInGPU,SDL::Module** lowerModulesInGPU,int* mdMemoryCounter,SDL::MDAlgo algo)
{
    //create the MD candidates - 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i<nModules; i+= stride)
    {
        //fetch the hit arrays from the modules and launch the other kernel
        SDL::Module* lowerModule = lowerModulesInGPU[i];
        SDL::Module* upperModule = (lowerModulesInGPU[i]->partnerModule());
        if(upperModule == nullptr) continue;
        int numberOfLowerHits = lowerModule->getNumberOfHits();
        int numberOfUpperHits = upperModule->getNumberOfHits();
        SDL::Hit** lowerHits = lowerModule->getHitPtrs();
        SDL::Hit** upperHits = upperModule->getHitPtrs();
        if(numberOfLowerHits == 0 ||  numberOfUpperHits == 0) continue; 
        int nThreads = min(32,(numberOfLowerHits * numberOfUpperHits));
        //int nThreads = 32;
        int nBlocks = (numberOfLowerHits * numberOfUpperHits)%nThreads == 0 ? (numberOfLowerHits * numberOfUpperHits)/nThreads : (numberOfLowerHits * numberOfUpperHits)/nThreads + 1;
        printf("nBlocks = %d lower hits = %d upper hits = %d\n",nBlocks,numberOfLowerHits,numberOfUpperHits);
        runMiniDoubletGPUAlgo<<<nBlocks,nThreads>>>(i,lowerHits,upperHits,numberOfLowerHits,numberOfUpperHits,mdsInGPU,mdMemoryCounter,algo);
    }
}*/


__global__ void createMiniDoubletsInGPU(int nModules,SDL::MiniDoublet* mdsInGPU,SDL::Module** lowerModulesInGPU,int* mdMemoryCounter,SDL::MDAlgo algo)
{
    int moduleIter = blockIdx.x * blockDim.x + threadIdx.x;
    int moduleStride = blockDim.x * gridDim.x;
    int lowerHitIter = blockIdx.y * blockDim.y + threadIdx.y;
    int lowerHitStride = blockDim.y * gridDim.y;
    int upperHitIter = blockIdx.z * blockDim.z + threadIdx.z;
    int upperHitStride = blockDim.z * gridDim.z;

    for(int i = moduleIter; i<nModules;i+=moduleStride)
    {
        SDL::Module* lowerModule = lowerModulesInGPU[i];
        SDL::Module* upperModule = (lowerModulesInGPU[i]->partnerModule());
        if(upperModule == nullptr) continue;
        int numberOfLowerHits = lowerModule->getNumberOfHits();
        int numberOfUpperHits = upperModule->getNumberOfHits();

        if(numberOfLowerHits == 0 || numberOfUpperHits == 0) continue;

        SDL::Hit** lowerHits = lowerModule->getHitPtrs();
        SDL::Hit** upperHits = upperModule->getHitPtrs();

        for(int j = lowerHitIter; j<numberOfLowerHits;j+=lowerHitStride)
        {
            for(int k = upperHitIter;k<numberOfUpperHits;k+=upperHitStride)
            {
                if(lowerHits[j] == nullptr || upperHits[k] == nullptr)
                    printf("nullptr hit encountered!\n");
                SDL::MiniDoublet mdCand(lowerHits[j],upperHits[k]);
                mdCand.runMiniDoubletAlgo(algo);
                if(mdCand.passesMiniDoubletAlgo(algo))
                {
                    int idx = atomicAdd(mdMemoryCounter,1);
//                    printf("counter = %d\n",idx);
                    mdsInGPU[idx] = mdCand;
                }
            }
        }

    }

}

SDL::Event::Event() : logLevel_(SDL::Log_Nothing)
{
    //createLayers();
    n_hits_by_layer_barrel_.fill(0);
    n_hits_by_layer_endcap_.fill(0);
    n_hits_by_layer_barrel_upper_.fill(0);
    n_hits_by_layer_endcap_upper_.fill(0);
    n_miniDoublet_candidates_by_layer_barrel_.fill(0);
    n_segment_candidates_by_layer_barrel_.fill(0);
    n_tracklet_candidates_by_layer_barrel_.fill(0);
    n_triplet_candidates_by_layer_barrel_.fill(0);
    n_trackcandidate_candidates_by_layer_barrel_.fill(0);
    n_miniDoublet_by_layer_barrel_.fill(0);
    n_segment_by_layer_barrel_.fill(0);
    n_tracklet_by_layer_barrel_.fill(0);
    n_triplet_by_layer_barrel_.fill(0);
    n_trackcandidate_by_layer_barrel_.fill(0);
    n_miniDoublet_candidates_by_layer_endcap_.fill(0);
    n_segment_candidates_by_layer_endcap_.fill(0);
    n_tracklet_candidates_by_layer_endcap_.fill(0);
    n_triplet_candidates_by_layer_endcap_.fill(0);
    n_trackcandidate_candidates_by_layer_endcap_.fill(0);
    n_miniDoublet_by_layer_endcap_.fill(0);
    n_segment_by_layer_endcap_.fill(0);
    n_tracklet_by_layer_endcap_.fill(0);
    n_triplet_by_layer_endcap_.fill(0);
    n_trackcandidate_by_layer_endcap_.fill(0);
    moduleMemoryCounter = 0;
    lowerModuleMemoryCounter = 0;
    hitMemoryCounter = 0;
    hit2SEdgeMemoryCounter = 0;
    //mdMemoryCounter = 0;

}

SDL::Event::~Event()
{
    cudaFree(hitsInGPU);
    cudaFree(modulesInGPU);
    cudaFree(lowerModulesInGPU);
    cudaFree(mdsInGPU);
    //cudaFree(mdCandsGPU);
}


bool SDL::Event::hasModule(unsigned int detId)
{
    if (modulesMapByDetId_.find(detId) == modulesMapByDetId_.end())
    {
        return false;
    }
    else
    {
        return true;
    }
}

void SDL::Event::setLogLevel(SDL::LogLevel logLevel)
{
    logLevel_ = logLevel;
}

void SDL::Event::initModulesInGPU()
{
    const int MODULE_MAX=50000;
    cudaProfilerStart();
    cudaMallocManaged(&modulesInGPU,MODULE_MAX * sizeof(SDL::Module));
    cudaMallocManaged(&lowerModulesInGPU, MODULE_MAX * sizeof(SDL::Module*));
    cudaProfilerStop();
}

SDL::Module* SDL::Event::getModule(unsigned int detId)
{
    // using std::map::emplace
    if(moduleMemoryCounter == 0)
    {
        initModulesInGPU();
    }
    std::pair<std::map<unsigned int, Module*>::iterator, bool> emplace_result = modulesMapByDetId_.emplace(detId,nullptr);
    // Retreive the module
    auto& inserted_or_existing = (*(emplace_result.first)).second;

    // If new was inserted, then insert to modulePtrs_ pointer list
    if (emplace_result.second) // if true, new was inserted
    {
        //cudaMallocManaged(&((*(emplace_result.first)).second),sizeof(SDL::Module));
         (*(emplace_result.first)).second = &modulesInGPU[moduleMemoryCounter];

        //*inserted_or_existing =SDL:: Module(detId);
        modulesInGPU[moduleMemoryCounter] = SDL::Module(detId);
        Module* module_ptr = inserted_or_existing;
        module_ptr->setDrDz(tiltedGeometry.getDrDz(detId));
        if(module_ptr->subdet() == SDL::Module::Endcap)
        {
            module_ptr->setSlope(SDL::endcapGeometry.getSlopeLower(detId));
        }
        else
        {
            module_ptr->setSlope(SDL::tiltedGeometry.getSlope(detId));
        }

        
        // Add the module pointer to the list of modules
        modulePtrs_.push_back(module_ptr);
        // If the module is lower module then add to list of lower modules
        if (module_ptr->isLower())
        {
            lowerModulesInGPU[lowerModuleMemoryCounter] = module_ptr;
            lowerModuleMemoryCounter++;
        }
       
       moduleMemoryCounter++;

    }

    return inserted_or_existing;
}

const std::vector<SDL::Module*> SDL::Event::getModulePtrs() const
{
    return modulePtrs_;
}

const std::vector<SDL::Module*> SDL::Event::getLowerModulePtrs() const
{
    return lowerModulePtrs_;
}

/*
void SDL::Event::createLayers()
{
    // Create barrel layers
    for (int ilayer = SDL::Layer::BarrelLayer0; ilayer < SDL::Layer::nBarrelLayer; ++ilayer)
    {
        barrelLayers_[ilayer] = SDL::Layer(ilayer, SDL::Layer::Barrel);
        layerPtrs_.push_back(&(barrelLayers_[ilayer]));
    }

    // Create endcap layers
    for (int ilayer = SDL::Layer::EndcapLayer0; ilayer < SDL::Layer::nEndcapLayer; ++ilayer)
    {
        endcapLayers_[ilayer] = SDL::Layer(ilayer, SDL::Layer::Endcap);
        layerPtrs_.push_back(&(endcapLayers_[ilayer]));
    }
}

SDL::Layer& SDL::Event::getLayer(int ilayer, SDL::Layer::SubDet subdet)
{
    if (subdet == SDL::Layer::Barrel)
        return barrelLayers_[ilayer];
    else // if (subdet == SDL::Layer::Endcap)
        return endcapLayers_[ilayer];
}

const std::vector<SDL::Layer*> SDL::Event::getLayerPtrs() const
{
    return layerPtrs_;
}*/

void SDL::Event::initHitsInGPU()
{
    const int HIT_MAX = 1000000;
    cudaMallocManaged(&hitsInGPU,HIT_MAX * sizeof(SDL::Hit));
    const int HIT_2S_MAX = 100000;
    cudaMallocManaged(&hits2sEdgeInGPU,HIT_2S_MAX * sizeof(SDL::Hit));
    //cudaDeviceSynchronize();
}

void SDL::Event::addHitToModule(SDL::Hit hit, unsigned int detId)
{
    // Add to global list of hits, where it will hold the object's instance
    // And get the module (if not exists, then create), and add the address to Module.hits_
    //construct a cudaMallocManaged object and send that in, so that we won't have issues in the GPU
    if(hitMemoryCounter == 0)
    {
        initHitsInGPU();
    }
    hitsInGPU[hitMemoryCounter] = hit;
    hitsInGPU[hitMemoryCounter].setModule(getModule(detId));
    getModule(detId)->addHit(&hitsInGPU[hitMemoryCounter]);
    hits_.push_back(hitsInGPU[hitMemoryCounter]);


    // Count number of hits in the event
    incrementNumberOfHits(*getModule(detId));

    // If the hit is 2S in the endcap then the hit boundary needs to be set
    if (getModule(detId)->subdet() == SDL::Module::Endcap and getModule(detId)->moduleType() == SDL::Module::TwoS)
    {
         
        hits2sEdgeInGPU[hit2SEdgeMemoryCounter] = SDL::GeometryUtil::stripHighEdgeHit(hitsInGPU[hitMemoryCounter]);
        hits2sEdgeInGPU[hit2SEdgeMemoryCounter+1] = SDL::GeometryUtil::stripLowEdgeHit(hitsInGPU[hitMemoryCounter]);
//        hits_2s_edges_.push_back(GeometryUtil::stripHighEdgeHit(&hits_.back()));
//        hits_.back().setHitHighEdgePtr(&(hits_2s_edges_.back()));
//        hits_2s_edges_.push_back(GeometryUtil::stripLowEdgeHit(*hitForGPU));
//        hits_.back().setHitLowEdgePtr(&(hits_2s_edges_.back()));
        hits_2s_edges_.push_back(hits2sEdgeInGPU[hit2SEdgeMemoryCounter]);
        hitsInGPU[hitMemoryCounter].setHitHighEdgePtr(&hits2sEdgeInGPU[hit2SEdgeMemoryCounter]);

        hits_2s_edges_.push_back(hits2sEdgeInGPU[hit2SEdgeMemoryCounter+1]);
        hitsInGPU[hitMemoryCounter].setHitLowEdgePtr(&hits2sEdgeInGPU[hit2SEdgeMemoryCounter+1]);

        hit2SEdgeMemoryCounter+= 2;
    }

    hitMemoryCounter++;
}

void SDL::Event::initMDsInGPU()
{
    const int MD_MAX = 60000;
    cudaMallocManaged(&mdsInGPU,MD_MAX * sizeof(SDL::MiniDoublet));
    cudaMallocManaged(&mdMemoryCounter,sizeof(int));
    *mdMemoryCounter = 0;
    //cudaDeviceSynchronize();
}

void SDL::Event::addMiniDoubletToEvent(SDL::MiniDoublet md, unsigned int detId)//, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of mini doublets, where it will hold the object's instance

    // And get the module (if not exists, then create), and add the address to Module.hits_
    //construct a cudaMallocManaged object and send that in, so that we won't have issues in the GPU
    getModule(detId)->addMiniDoublet(&md);
    miniDoublets_.push_back(md);

    incrementNumberOfMiniDoublets(*getModule(detId));
    // And get the layer
//    getLayer(layerIdx, subdet).addMiniDoublet(&mdsInGPU[mdMemoryCounter]);
}

[[deprecated("SDL:: addMiniDoubletToLowerModule() is deprecated. Use addMiniDoubletToEvent")]]
void SDL::Event::addMiniDoubletToLowerModule(SDL::MiniDoublet md, unsigned int detId)
{
    // Add to global list of mini doublets, where it will hold the object's instance
    miniDoublets_.push_back(md);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addMiniDoublet(&(miniDoublets_.back()));
}
/*
void SDL::Event::addSegmentToEvent(SDL::Segment sg, unsigned int detId, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addSegment(&(segments_.back()));

    // And get the layer andd the segment to it
    getLayer(layerIdx, subdet).addSegment(&(segments_.back()));

    // Link segments to mini-doublets
    segments_.back().addSelfPtrToMiniDoublets();

}

void SDL::Event::addTrackletToEvent(SDL::Tracklet tl, unsigned int detId, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    tracklets_.push_back(tl);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addTracklet(&(tracklets_.back()));

    // And get the layer andd the segment to it
    getLayer(layerIdx, subdet).addTracklet(&(tracklets_.back()));

    // Link segments to mini-doublets
    tracklets_.back().addSelfPtrToSegments();

}

void SDL::Event::addTripletToEvent(SDL::Triplet tp, unsigned int detId, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    triplets_.push_back(tp);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addTriplet(&(triplets_.back()));

    // And get the layer andd the triplet to it
    getLayer(layerIdx, subdet).addTriplet(&(triplets_.back()));
}

[[deprecated("SDL:: addSegmentToLowerModule() is deprecated. Use addSegmentToEvent")]]
void SDL::Event::addSegmentToLowerModule(SDL::Segment sg, unsigned int detId)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the module (if not exists, then create), and add the address to Module.hits_
    getModule(detId)->addSegment(&(segments_.back()));
}

[[deprecated("SDL:: addSegmentToLowerLayer() is deprecated. Use addSegmentToEvent")]]
void SDL::Event::addSegmentToLowerLayer(SDL::Segment sg, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of segments, where it will hold the object's instance
    segments_.push_back(sg);

    // And get the layer
    getLayer(layerIdx, subdet).addSegment(&(segments_.back()));
}

void SDL::Event::addTrackletToLowerLayer(SDL::Tracklet tl, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of tracklets, where it will hold the object's instance
    tracklets_.push_back(tl);

    // And get the layer
    getLayer(layerIdx, subdet).addTracklet(&(tracklets_.back()));
}

void SDL::Event::addTrackCandidateToLowerLayer(SDL::TrackCandidate tc, int layerIdx, SDL::Layer::SubDet subdet)
{
    // Add to global list of trackcandidates, where it will hold the object's instance
    trackcandidates_.push_back(tc);

    // And get the layer
    getLayer(layerIdx, subdet).addTrackCandidate(&(trackcandidates_.back()));
}

*/

//This dude needs to get into the GPU
/*
void SDL::Event::createMiniDoublets(MDAlgo algo)
{
    for(int i = 0; i < moduleMemoryCounter; i++)
    {
        moduleMemoryCounter[i].setPartnerModule(*getModule(moduleMemoryCounter.partnerDetId()));
    }

    //set partner modules first for the modules
    // Loop over lower modules
    const int MAX_MD_CAND = 500000;
    initMDsInGPU();

//    cudaMallocManaged(&mdCandsGPU,(int)(1.5 * MAX_MD_CAND)*sizeof(SDL::MiniDoublet));
     
//    cudaMemPrefetchAsync(modulesInGPU,50000 * sizeof(Hit),0);
//    cudaMemPrefetchAsync(hitsInGPU,1000000 * sizeof(Hit),0);
    mdGPUCounter = 0;

    //gpu code - put this inside the kernel



    for (auto& lowerModulePtr : getLowerModulePtrs())
    {
        // Create mini doublets
        createMiniDoubletsFromLowerModule(lowerModulePtr->detId(), MAX_MD_CAND,algo);
    }
    if(mdGPUCounter < MAX_MD_CAND and mdGPUCounter > 0) //incomplete dudes from the final iteration
    {
        miniDoubletGPUWrapper(algo);
    }
}*/


void SDL::Event::createMiniDoublets(MDAlgo algo)
{
    for(int i = 0; i < moduleMemoryCounter; i++)
    {
        modulesInGPU[i].setPartnerModule(getModule(modulesInGPU[i].partnerDetId()));
    }

    //const int MAX_MD_CAND = 500000;
    initMDsInGPU();

    int nModules = lowerModuleMemoryCounter;
    int MAX_HITS = 100;
//    int nBlocks = (nModules % nThreads == 0) ? nModules/nThreads : nModules/nThreads + 1;
    dim3 nThreads(16,8,8);
    dim3 nBlocks((nModules % nThreads.x == 0 ? nModules/nThreads.x : nModules/nThreads.x + 1),(MAX_HITS % nThreads.y == 0 ? MAX_HITS/nThreads.y : MAX_HITS/nThreads.y + 1), (MAX_HITS % nThreads.z == 0 ? MAX_HITS/nThreads.z : MAX_HITS/nThreads.z + 1));
      std::cout<<nBlocks.x<<" " <<nBlocks.y<<" "<<nBlocks.z<<" "<<std::endl;
//    int nBlocks = (mdGPUCounter % nThreads == 0) ? mdGPUCounter/nThreads : mdGPUCounter/nThreads + 1;
    cudaProfilerStart();
    createMiniDoubletsInGPU<<<nBlocks,nThreads>>>(nModules,mdsInGPU,lowerModulesInGPU,mdMemoryCounter,algo);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    cudaProfilerStop();
    if (cudaerr != cudaSuccess)
    {          
        std::cout<<"kernel launch failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;    
    }
        std::cout<<"Number of mini-doublets:"<<*mdMemoryCounter<<std::endl;
    //add mini-doublets to the module arrays for other stuff outside
    for(int i=0; i<*mdMemoryCounter;i++)
    {
//        SDL::cout<<mdsInGPU[i]<<std::endl;
        if(mdsInGPU[i].lowerHitPtr() == nullptr)
            std::cout<<"lower hit nullptr"<<std::endl;
        SDL::Module& lowerModule = (Module&)(mdsInGPU[i].lowerHitPtr()->getModule());
        if(lowerModule.subdet() == SDL::Module::Barrel)
        {
            addMiniDoubletToEvent(mdsInGPU[i],lowerModule.detId());//,lowerModule.layer(),SDL::Layer::Barrel);    
        }
        else
        {
            addMiniDoubletToEvent(mdsInGPU[i],lowerModule.detId());//,lowerModule.layer(),SDL::Layer::Barrel);
        }
    }
}


/*void SDL::Event::miniDoubletGPUWrapper(SDL::MDAlgo algo)
{
    int nThreads = 256;
    int nBlocks = (mdGPUCounter % nThreads == 0) ? mdGPUCounter/nThreads : mdGPUCounter/nThreads + 1;
    createMiniDoubletsInGPU <<<nBlocks, nThreads>>> (mdCandsGPU,mdGPUCounter,algo);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {          
        std::cout<<"kernel launch failed with error : "<<cudaGetErrorString(cudaerr)<<std::endl;    
    }

    for(int i = 0; i < mdGPUCounter; i++)
    {
        auto mdCand = mdCandsGPU[i];
        if(mdCand.passesMiniDoubletAlgo(algo))
        {
            // Count the number of md formed
            SDL::Module& lowerModule = (Module&)((mdCand.lowerHitPtr())->getModule()); 
            incrementNumberOfMiniDoublets(lowerModule);

            if (lowerModule.subdet() == SDL::Module::Barrel)
            {
                addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::Layer::Barrel);
            }
            else
            {
                addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::Layer::Endcap);
            }
        }
    }
    mdGPUCounter = 0; 
   
}*/

/*
void SDL::Event::createMiniDoubletsFromLowerModule(unsigned int detId, int maxMDCands,SDL::MDAlgo algo)
{
    // Get reference to the lower Module
    Module& lowerModule = *getModule(detId);
    //std::cout<<"\nLower module = "<<detId<<" position in array = "<<(getModule(detId) - modulesInGPU)<<std::endl;
    // Get reference to the upper Module
    Module& upperModule = *getModule(lowerModule.partnerDetId());
    //std::cout<<"Upper module = "<<lowerModule.partnerDetId()<<" position in array = "<<getModule(lowerModule.partnerDetId()) - modulesInGPU<<std::endl;
    // Double nested loops
    // Loop over lower module hits
    for(size_t i =0; i<lowerModule.getHitPtrs().size();i++)
//    for (auto& lowerHitPtr : lowerModule.getHitPtrs())
    {
        // Get reference to lower Hit
//        SDL::Hit& lowerHit = *lowerHitPtr;

        // Loop over upper module hits
        for(size_t j = 0; j<upperModule.getHitPtrs().size();j++)
//        for (auto& upperHitPtr : upperModule.getHitPtrs())
        {
            auto& lowerHitPtr = lowerModule.getHitPtrs().at(i);
	    auto& upperHitPtr = upperModule.getHitPtrs().at(j);
            // Get reference to upper Hit
//            SDL::Hit& upperHit = *upperHitPtr;

            // Create a mini-doublet candidate
            SDL::MiniDoublet mdCand(lowerHitPtr, upperHitPtr);
            if(lowerModule.moduleType() == SDL::Module::PS and upperModule.moduleLayerType() == SDL::Module::Strip)
            {
                mdCand.setDrDz(tiltedGeometry.getDrDz(upperModule.detId())); 
            }
            else
            {
                mdCand.setDrDz(tiltedGeometry.getDrDz(lowerModule.detId()));

            }
            if(lowerModule.subdet() == SDL::Module::Endcap)
            {
                if(lowerModule.moduleType() == SDL::Module::PS and upperModule.moduleLayerType() == SDL::Module::Strip)
                {
                    mdCand.setLowerModuleSlope(SDL::endcapGeometry.getSlopeLower(upperModule.detId()));
                }
                else
                {
                    mdCand.setLowerModuleSlope(SDL::endcapGeometry.getSlopeLower(lowerModule.detId()));
                }
            }
            else
            {
                //FIXME: Might need some jugaad for nonexistent det Ids
                if(lowerModule.moduleType() == SDL::Module::PS and upperModule.moduleLayerType() == SDL::Module::Strip)
                {
                    mdCand.setLowerModuleSlope(SDL::tiltedGeometry.getSlope(upperModule.detId()));
                }
                else
                {
                    mdCand.setLowerModuleSlope(SDL::tiltedGeometry.getSlope(lowerModule.detId()));
                }
            }
//	        memcpy(&mdCandsGPU[mdGPUCounter],&mdCand,sizeof(SDL::MiniDoublet));
            mdCandsGPU[mdGPUCounter + i * upperModule.getHitPtrs().size() + j] = mdCand;
//            mdGPUCounter++;

             // Count the number of mdCand considered
    //        incrementNumberOfMiniDoubletCandidates(lowerModule);
        }
    }
    //Checking here
    
    incrementNumberOfMiniDoubletCandidates(lowerModule,upperModule.getHitPtrs().size() * lowerModule.getHitPtrs().size());
    mdGPUCounter += lowerModule.getHitPtrs().size() * upperModule.getHitPtrs().size();
    if(mdGPUCounter >= maxMDCands)
    {
        miniDoubletGPUWrapper(algo);
    }       
    // Run mini-doublet algorithm on mdCand (mini-doublet candidate)
           //after running MD algo'
}*/


/*
void SDL::Event::createPseudoMiniDoubletsFromAnchorModule(SDL::MDAlgo algo)
{

    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        unsigned int detId = lowerModulePtr->detId();

        // Get reference to the lower Module
        Module& lowerModule = *getModule(detId);

        // Assign anchor hit pointers based on their hit type
        bool loopLower = true;
        if (lowerModule.moduleType() == SDL::Module::PS)
        {
            if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
            {
                loopLower = true;
            }
            else
            {
                loopLower = false;
            }
        }
        else
        {
            loopLower = true;
        }

        // Get reference to the upper Module
        Module& upperModule = *getModule(lowerModule.partnerDetId());

        if (loopLower)
        {
            // Loop over lower module hits
            for (auto& lowerHitPtr : lowerModule.getHitPtrs())
            {
                // Get reference to lower Hit
                SDL::Hit& lowerHit = *lowerHitPtr;

                // Loop over upper module hits
                for (auto& upperHitPtr : upperModule.getHitPtrs())
                {

                    // Get reference to upper Hit
                    SDL::Hit& upperHit = *upperHitPtr;

                    // Create a mini-doublet candidate
                    SDL::MiniDoublet mdCand(lowerHitPtr, upperHitPtr);

                    // Count the number of mdCand considered
                    incrementNumberOfMiniDoubletCandidates(lowerModule);

                    // Run mini-doublet algorithm on mdCand (mini-doublet candidate)
                    mdCand.runMiniDoubletAlgo(SDL::AllComb_MDAlgo, logLevel_);

                    if (mdCand.passesMiniDoubletAlgo(SDL::AllComb_MDAlgo))
                    {

                        // Count the number of md formed
                        incrementNumberOfMiniDoublets(lowerModule);

                        if (lowerModule.subdet() == SDL::Module::Barrel)
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::Layer::Barrel);
                        else
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::Layer::Endcap);

                        // Break to exit on first pseudo mini-doublet
                        break;
                    }

                }

            }

        }
        else
        {
            // Loop over lower module hits
            for (auto& upperHitPtr : upperModule.getHitPtrs())
            {
                // Get reference to upper Hit
                SDL::Hit& upperHit = *upperHitPtr;

                // Loop over upper module hits
                for (auto& lowerHitPtr : lowerModule.getHitPtrs())
                {

                    // Get reference to lower Hit
                    SDL::Hit& lowerHit = *lowerHitPtr;

                    // Create a mini-doublet candidate
                    SDL::MiniDoublet mdCand(lowerHitPtr, upperHitPtr);

                    // Count the number of mdCand considered
                    incrementNumberOfMiniDoubletCandidates(lowerModule);

                    // Run mini-doublet algorithm on mdCand (mini-doublet candidate)
                    mdCand.runMiniDoubletAlgo(SDL::AllComb_MDAlgo, logLevel_);

                    if (mdCand.passesMiniDoubletAlgo(SDL::AllComb_MDAlgo))
                    {

                        // Count the number of md formed
                        incrementNumberOfMiniDoublets(lowerModule);

                        if (lowerModule.subdet() == SDL::Module::Barrel)
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::Layer::Barrel);
                        else
                            addMiniDoubletToEvent(mdCand, lowerModule.detId(), lowerModule.layer(), SDL::Layer::Endcap);

                        // Break to exit on first pseudo mini-doublet
                        break;
                    }

                }

            }

        }
    }

}*/

/*
void SDL::Event::createSegments(SGAlgo algo)
{

    for (auto& segment_compatible_layer_pair : SDL::Layer::getListOfSegmentCompatibleLayerPairs())
    {
        int innerLayerIdx = segment_compatible_layer_pair.first.first;
        SDL::Layer::SubDet innerLayerSubDet = segment_compatible_layer_pair.first.second;
        int outerLayerIdx = segment_compatible_layer_pair.second.first;
        SDL::Layer::SubDet outerLayerSubDet = segment_compatible_layer_pair.second.second;
        createSegmentsFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    }
}

void SDL::Event::createSegmentsWithModuleMap(SGAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // Create mini doublets
        createSegmentsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createSegmentsFromInnerLowerModule(unsigned int detId, SDL::SGAlgo algo)
{

    // x's and y's are mini doublets
    // -------x--------
    // --------x------- <--- outer lower module
    //
    // --------y-------
    // -------y-------- <--- inner lower module

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module mini-doublets
    for (auto& innerMiniDoubletPtr : innerLowerModule.getMiniDoubletPtrs())
    {

        // Get reference to mini-doublet in inner lower module
        SDL::MiniDoublet& innerMiniDoublet = *innerMiniDoubletPtr;

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(detId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerMiniDoubletPtr : outerLowerModule.getMiniDoubletPtrs())
            {

                // Get reference to mini-doublet in outer lower module
                SDL::MiniDoublet& outerMiniDoublet = *outerMiniDoubletPtr;

                // Create a segment candidate
                SDL::Segment sgCand(innerMiniDoubletPtr, outerMiniDoubletPtr);

                // Run segment algorithm on sgCand (segment candidate)
                sgCand.runSegmentAlgo(algo, logLevel_);

                // Count the # of sgCands considered by layer
                incrementNumberOfSegmentCandidates(innerLowerModule);

                if (sgCand.passesSegmentAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfSegments(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }
        }
    }
}

void SDL::Event::createTriplets(TPAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (lowerModulePtr->layer() != 4 and lowerModulePtr->layer() != 3)
        //     continue;

        // Create mini doublets
        createTripletsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createTripletsFromInnerLowerModule(unsigned int detId, SDL::TPAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Segment& innerSegment = *innerSegmentPtr;

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(detId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                // Get reference to mini-doublet in outer lower module
                SDL::Segment& outerSegment = *outerSegmentPtr;

                // Create a segment candidate
                SDL::Triplet tpCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tpCand (segment candidate)
                tpCand.runTripletAlgo(algo, logLevel_);

                // Count the # of tpCands considered by layer
                incrementNumberOfTripletCandidates(innerLowerModule);

                if (tpCand.passesTripletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTriplets(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addTripletToEvent(tpCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addTripletToEvent(tpCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }
        }
    }
}

void SDL::Event::createTracklets(TLAlgo algo)
{
    for (auto& tracklet_compatible_layer_pair : SDL::Layer::getListOfTrackletCompatibleLayerPairs())
    {
        int innerLayerIdx = tracklet_compatible_layer_pair.first.first;
        SDL::Layer::SubDet innerLayerSubDet = tracklet_compatible_layer_pair.first.second;
        int outerLayerIdx = tracklet_compatible_layer_pair.second.first;
        SDL::Layer::SubDet outerLayerSubDet = tracklet_compatible_layer_pair.second.second;
        createTrackletsFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    }
}

void SDL::Event::createTrackletsWithModuleMap(TLAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (lowerModulePtr->layer() != 1)
        //     continue;

        // Create mini doublets
        createTrackletsFromInnerLowerModule(lowerModulePtr->detId(), algo);

    }
}

// Create tracklets from inner modules
void SDL::Event::createTrackletsFromInnerLowerModule(unsigned int detId, SDL::TLAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Segment& innerSegment = *innerSegmentPtr;

        // Get the outer mini-doublet module detId
        const SDL::Module& innerSegmentOuterModule = innerSegment.outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerSegmentOuterModuleDetId = innerSegmentOuterModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerSegmentOuterModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerSegmentPtr : outerLowerModule.getSegmentPtrs())
            {

                // Count the # of tlCands considered by layer
                incrementNumberOfTrackletCandidates(innerLowerModule);

                // Get reference to mini-doublet in outer lower module
                SDL::Segment& outerSegment = *outerSegmentPtr;

                // Create a tracklet candidate
                SDL::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                // Run segment algorithm on tlCand (tracklet candidate)
                tlCand.runTrackletAlgo(algo, logLevel_);

                if (tlCand.passesTrackletAlgo(algo))
                {

                    // Count the # of sg formed by layer
                    incrementNumberOfTracklets(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }

        }

    }

}

// Create tracklets via navigation
void SDL::Event::createTrackletsViaNavigation(SDL::TLAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // Get reference to the inner lower Module
        Module& innerLowerModule = *getModule(lowerModulePtr->detId());

        // Triple nested loops
        // Loop over inner lower module for segments
        for (auto& innerSegmentPtr : innerLowerModule.getSegmentPtrs())
        {

            // Get reference to segment in inner lower module
            SDL::Segment& innerSegment = *innerSegmentPtr;

            // Get the connecting segment ptrs
            for (auto& connectingSegmentPtr : innerSegmentPtr->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
            {

                for (auto& outerSegmentPtr : connectingSegmentPtr->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                {

                    // Count the # of tlCands considered by layer
                    incrementNumberOfTrackletCandidates(innerLowerModule);

                    // Get reference to mini-doublet in outer lower module
                    SDL::Segment& outerSegment = *outerSegmentPtr;

                    // Create a tracklet candidate
                    SDL::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

                    // Run segment algorithm on tlCand (tracklet candidate)
                    tlCand.runTrackletAlgo(algo, logLevel_);

                    if (tlCand.passesTrackletAlgo(algo))
                    {

                        // Count the # of sg formed by layer
                        incrementNumberOfTracklets(innerLowerModule);

                        if (innerLowerModule.subdet() == SDL::Module::Barrel)
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                        else
                            addTrackletToEvent(tlCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
                    }

                }

            }

        }
    }

}


// Create tracklets from two layers (inefficient way)
void SDL::Event::createTrackletsFromTwoLayers(int innerLayerIdx, SDL::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::Layer::SubDet outerLayerSubDet, TLAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerSegmentPtr : innerLayer.getSegmentPtrs())
    {
        SDL::Segment& innerSegment = *innerSegmentPtr;
        for (auto& outerSegmentPtr : outerLayer.getSegmentPtrs())
        {
            // SDL::Segment& outerSegment = *outerSegmentPtr;

            // if (SDL::Tracklet::isSegmentPairATracklet(innerSegment, outerSegment, algo, logLevel_))
            //     addTrackletToLowerLayer(SDL::Tracklet(innerSegmentPtr, outerSegmentPtr), innerLayerIdx, innerLayerSubDet);

            SDL::Segment& outerSegment = *outerSegmentPtr;

            SDL::Tracklet tlCand(innerSegmentPtr, outerSegmentPtr);

            tlCand.runTrackletAlgo(algo, logLevel_);

            // Count the # of tracklet candidate considered by layer
            incrementNumberOfTrackletCandidates(innerLayer);

            if (tlCand.passesTrackletAlgo(algo))
            {

                // Count the # of tracklet formed by layer
                incrementNumberOfTracklets(innerLayer);

                addTrackletToLowerLayer(tlCand, innerLayerIdx, innerLayerSubDet);
            }

        }
    }
}

// Create segments from two layers (inefficient way)
void SDL::Event::createSegmentsFromTwoLayers(int innerLayerIdx, SDL::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::Layer::SubDet outerLayerSubDet, SGAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerMiniDoubletPtr : innerLayer.getMiniDoubletPtrs())
    {
        SDL::MiniDoublet& innerMiniDoublet = *innerMiniDoubletPtr;

        for (auto& outerMiniDoubletPtr : outerLayer.getMiniDoubletPtrs())
        {
            SDL::MiniDoublet& outerMiniDoublet = *outerMiniDoubletPtr;

            SDL::Segment sgCand(innerMiniDoubletPtr, outerMiniDoubletPtr);

            sgCand.runSegmentAlgo(algo, logLevel_);

            if (sgCand.passesSegmentAlgo(algo))
            {
                const SDL::Module& innerLowerModule = innerMiniDoubletPtr->lowerHitPtr()->getModule();
                if (innerLowerModule.subdet() == SDL::Module::Barrel)
                    addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Barrel);
                else
                    addSegmentToEvent(sgCand, innerLowerModule.detId(), innerLowerModule.layer(), SDL::Layer::Endcap);
            }

        }
    }
}

void SDL::Event::createTrackCandidates(TCAlgo algo)
{
    // TODO Implement some structure for Track Candidates
    // for (auto& trackCandidate_compatible_layer_pair : SDL::Layer::getListOfTrackCandidateCompatibleLayerPairs())
    // {
    //     int innerLayerIdx = trackCandidate_compatible_layer_pair.first.first;
    //     SDL::Layer::SubDet innerLayerSubDet = trackCandidate_compatible_layer_pair.first.second;
    //     int outerLayerIdx = trackCandidate_compatible_layer_pair.second.first;
    //     SDL::Layer::SubDet outerLayerSubDet = trackCandidate_compatible_layer_pair.second.second;
    //     createTrackCandidatesFromTwoLayers(innerLayerIdx, innerLayerSubDet, outerLayerIdx, outerLayerSubDet, algo);
    // }

    createTrackCandidatesFromTwoLayers(1, SDL::Layer::Barrel, 3, SDL::Layer::Barrel, algo);

}

// Create trackCandidates from two layers (inefficient way)
void SDL::Event::createTrackCandidatesFromTwoLayers(int innerLayerIdx, SDL::Layer::SubDet innerLayerSubDet, int outerLayerIdx, SDL::Layer::SubDet outerLayerSubDet, TCAlgo algo)
{
    Layer& innerLayer = getLayer(innerLayerIdx, innerLayerSubDet);
    Layer& outerLayer = getLayer(outerLayerIdx, outerLayerSubDet);

    for (auto& innerTrackletPtr : innerLayer.getTrackletPtrs())
    {
        SDL::Tracklet& innerTracklet = *innerTrackletPtr;

        for (auto& outerTrackletPtr : outerLayer.getTrackletPtrs())
        {

            SDL::Tracklet& outerTracklet = *outerTrackletPtr;

            SDL::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

            tcCand.runTrackCandidateAlgo(algo, logLevel_);

            // Count the # of track candidates considered
            incrementNumberOfTrackCandidateCandidates(innerLayer);

            if (tcCand.passesTrackCandidateAlgo(algo))
            {

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidates(innerLayer);

                addTrackCandidateToLowerLayer(tcCand, innerLayerIdx, innerLayerSubDet);
            }

        }
    }
}

void SDL::Event::createTrackCandidatesFromTriplets(TCAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (not (lowerModulePtr->layer() == 1))
        //     continue;

        // Create mini doublets
        createTrackCandidatesFromInnerModulesFromTriplets(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createTrackCandidatesFromInnerModulesFromTriplets(unsigned int detId, SDL::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTripletPtr : innerLowerModule.getTripletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Triplet& innerTriplet = *innerTripletPtr;

        // Get the outer mini-doublet module detId
        const SDL::Module& innerTripletOutermostModule = innerTriplet.outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTripletOutermostModuleDetId = innerTripletOutermostModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTripletOutermostModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTripletPtr : outerLowerModule.getTripletPtrs())
            {

                // Count the # of tlCands considered by layer
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                // Segment between innerSgOuterMD - outerSgInnerMD
                SDL::Segment sgCand(innerTripletPtr->outerSegmentPtr()->outerMiniDoubletPtr(),outerTripletPtr->innerSegmentPtr()->innerMiniDoubletPtr());

                // Run the segment algo (supposedly is fast)
                sgCand.runSegmentAlgo(SDL::Default_SGAlgo, logLevel_);

                if (not (sgCand.passesSegmentAlgo(SDL::Default_SGAlgo)))
                {
                    continue;
                }

                // SDL::Tracklet tlCand(innerTripletPtr->innerSegmentPtr(), &sgCand);

                // // Run the segment algo (supposedly is fast)
                // tlCand.runTrackletAlgo(SDL::Default_TLAlgo, logLevel_);

                // if (not (tlCand.passesTrackletAlgo(SDL::Default_TLAlgo)))
                // {
                //     continue;
                // }

                SDL::Tracklet tlCandOuter(&sgCand, outerTripletPtr->outerSegmentPtr());

                // Run the segment algo (supposedly is fast)
                tlCandOuter.runTrackletAlgo(SDL::Default_TLAlgo, logLevel_);

                if (not (tlCandOuter.passesTrackletAlgo(SDL::Default_TLAlgo)))
                {
                    continue;
                }

                SDL::TrackCandidate tcCand(innerTripletPtr, outerTripletPtr);

                // if (tcCand.passesTrackCandidateAlgo(algo))
                // {

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidates(innerLowerModule);

                addTrackCandidateToLowerLayer(tcCand, 1, SDL::Layer::Barrel);
                // if (innerLowerModule.subdet() == SDL::Module::Barrel)
                //     addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Barrel);
                // else
                //     addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Endcap);

                // }

            }

        }

    }


}

void SDL::Event::createTrackCandidatesFromTracklets(TCAlgo algo)
{
    // Loop over lower modules
    for (auto& lowerModulePtr : getLowerModulePtrs())
    {

        // if (not (lowerModulePtr->layer() == 1))
        //     continue;

        // Create mini doublets
        createTrackCandidatesFromInnerModulesFromTracklets(lowerModulePtr->detId(), algo);

    }
}

void SDL::Event::createTrackCandidatesFromInnerModulesFromTracklets(unsigned int detId, SDL::TCAlgo algo)
{

    // Get reference to the inner lower Module
    Module& innerLowerModule = *getModule(detId);

    // Triple nested loops
    // Loop over inner lower module for segments
    for (auto& innerTrackletPtr : innerLowerModule.getTrackletPtrs())
    {

        // Get reference to segment in inner lower module
        SDL::Tracklet& innerTracklet = *innerTrackletPtr;

        // Get the outer mini-doublet module detId
        const SDL::Module& innerTrackletSecondModule = innerTracklet.innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule();

        unsigned int innerTrackletSecondModuleDetId = innerTrackletSecondModule.detId();

        // Get connected outer lower module detids
        const std::vector<unsigned int>& connectedModuleDetIds = moduleConnectionMap.getConnectedModuleDetIds(innerTrackletSecondModuleDetId);

        // Loop over connected outer lower modules
        for (auto& outerLowerModuleDetId : connectedModuleDetIds)
        {

            if (not hasModule(outerLowerModuleDetId))
                continue;

            // Get reference to the outer lower module
            Module& outerLowerModule = *getModule(outerLowerModuleDetId);

            // Loop over outer lower module mini-doublets
            for (auto& outerTrackletPtr : outerLowerModule.getTrackletPtrs())
            {

                SDL::Tracklet& outerTracklet = *outerTrackletPtr;

                SDL::TrackCandidate tcCand(innerTrackletPtr, outerTrackletPtr);

                tcCand.runTrackCandidateAlgo(algo, logLevel_);

                // Count the # of track candidates considered
                incrementNumberOfTrackCandidateCandidates(innerLowerModule);

                if (tcCand.passesTrackCandidateAlgo(algo))
                {

                    // Count the # of track candidates considered
                    incrementNumberOfTrackCandidates(innerLowerModule);

                    if (innerLowerModule.subdet() == SDL::Module::Barrel)
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Barrel);
                    else
                        addTrackCandidateToLowerLayer(tcCand, innerLowerModule.layer(), SDL::Layer::Endcap);
                }

            }

        }

    }

}*/

// Multiplicity of mini-doublets
unsigned int SDL::Event::getNumberOfHits() { return hits_.size(); }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerBarrel(unsigned int ilayer) { return n_hits_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerEndcap(unsigned int ilayer) { return n_hits_by_layer_endcap_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerBarrelUpperModule(unsigned int ilayer) { return n_hits_by_layer_barrel_upper_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfHitsByLayerEndcapUpperModule(unsigned int ilayer) { return n_hits_by_layer_endcap_upper_[ilayer]; }

// Multiplicity of mini-doublets
unsigned int SDL::Event::getNumberOfMiniDoublets() { return miniDoublets_.size(); }

// Multiplicity of segments
//unsigned int SDL::Event::getNumberOfSegments() { return segments_.size(); }

// Multiplicity of tracklets
//unsigned int SDL::Event::getNumberOfTracklets() { return tracklets_.size(); }

// Multiplicity of triplets
//unsigned int SDL::Event::getNumberOfTriplets() { return triplets_.size(); }

// Multiplicity of track candidates
//unsigned int SDL::Event::getNumberOfTrackCandidates() { return trackcandidates_.size(); }

/*
// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfMiniDoubletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_miniDoublet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_miniDoublet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::Event::getNumberOfSegmentCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_segment_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_segment_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_tracklet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_tracklet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::Event::getNumberOfTripletCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_triplet_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_triplet_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackCandidateCandidates() { unsigned int n = 0; for (unsigned int i = 0; i < 6; ++i) {n += n_trackcandidate_candidates_by_layer_barrel_[i];} for (unsigned int i = 0; i < 5; ++i) {n += n_trackcandidate_candidates_by_layer_endcap_[i];} return n; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfMiniDoubletCandidatesByLayerBarrel(unsigned int ilayer) { return n_miniDoublet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::Event::getNumberOfSegmentCandidatesByLayerBarrel(unsigned int ilayer) { return n_segment_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackletCandidatesByLayerBarrel(unsigned int ilayer) { return n_tracklet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::Event::getNumberOfTripletCandidatesByLayerBarrel(unsigned int ilayer) { return n_triplet_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackCandidateCandidatesByLayerBarrel(unsigned int ilayer) { return n_trackcandidate_candidates_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet candidates considered in this event
unsigned int SDL::Event::getNumberOfMiniDoubletCandidatesByLayerEndcap(unsigned int ilayer) { return n_miniDoublet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of segment candidates considered in this event
unsigned int SDL::Event::getNumberOfSegmentCandidatesByLayerEndcap(unsigned int ilayer) { return n_segment_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of tracklet candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackletCandidatesByLayerEndcap(unsigned int ilayer) { return n_tracklet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of triplet candidates considered in this event
unsigned int SDL::Event::getNumberOfTripletCandidatesByLayerEndcap(unsigned int ilayer) { return n_triplet_candidates_by_layer_endcap_[ilayer]; }

// Multiplicity of track candidate candidates considered in this event
unsigned int SDL::Event::getNumberOfTrackCandidateCandidatesByLayerEndcap(unsigned int ilayer) { return n_trackcandidate_candidates_by_layer_endcap_[ilayer]; }
*/
// Multiplicity of mini-doublet formed in this event
unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int ilayer) { return n_miniDoublet_by_layer_barrel_[ilayer]; }
/*
// Multiplicity of segment formed in this event
unsigned int SDL::Event::getNumberOfSegmentsByLayerBarrel(unsigned int ilayer) { return n_segment_by_layer_barrel_[ilayer]; }

// Multiplicity of tracklet formed in this event
unsigned int SDL::Event::getNumberOfTrackletsByLayerBarrel(unsigned int ilayer) { return n_tracklet_by_layer_barrel_[ilayer]; }

// Multiplicity of triplet formed in this event
unsigned int SDL::Event::getNumberOfTripletsByLayerBarrel(unsigned int ilayer) { return n_triplet_by_layer_barrel_[ilayer]; }

// Multiplicity of track candidate formed in this event
unsigned int SDL::Event::getNumberOfTrackCandidatesByLayerBarrel(unsigned int ilayer) { return n_trackcandidate_by_layer_barrel_[ilayer]; }

// Multiplicity of mini-doublet formed in this event
unsigned int SDL::Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int ilayer) { return n_miniDoublet_by_layer_endcap_[ilayer]; }

// Multiplicity of segment formed in this event
unsigned int SDL::Event::getNumberOfSegmentsByLayerEndcap(unsigned int ilayer) { return n_segment_by_layer_endcap_[ilayer]; }

// Multiplicity of tracklet formed in this event
unsigned int SDL::Event::getNumberOfTrackletsByLayerEndcap(unsigned int ilayer) { return n_tracklet_by_layer_endcap_[ilayer]; }

// Multiplicity of triplet formed in this event
unsigned int SDL::Event::getNumberOfTripletsByLayerEndcap(unsigned int ilayer) { return n_triplet_by_layer_endcap_[ilayer]; }

// Multiplicity of track candidate formed in this event
unsigned int SDL::Event::getNumberOfTrackCandidatesByLayerEndcap(unsigned int ilayer) { return n_trackcandidate_by_layer_endcap_[ilayer]; }*/

// Multiplicity of hits in this event
void SDL::Event::incrementNumberOfHits(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);

    // Only count hits in lower module
    if (not module.isLower())
    {
        if (isbarrel)
            n_hits_by_layer_barrel_upper_[layer-1]++;
        else
            n_hits_by_layer_endcap_upper_[layer-1]++;
    }
    else
    {
        if (isbarrel)
            n_hits_by_layer_barrel_[layer-1]++;
        else
            n_hits_by_layer_endcap_[layer-1]++;
    }
}

// Multiplicity of mini-doublet candidates considered in this event
void SDL::Event::incrementNumberOfMiniDoubletCandidates(SDL::Module& module,int number)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_miniDoublet_candidates_by_layer_barrel_[layer-1]+=number;
    else
        n_miniDoublet_candidates_by_layer_endcap_[layer-1]+=number;
}

/*
// Multiplicity of segment candidates considered in this event
void SDL::Event::incrementNumberOfSegmentCandidates(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_segment_candidates_by_layer_barrel_[layer-1]++;
    else
        n_segment_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of triplet candidates considered in this event
void SDL::Event::incrementNumberOfTripletCandidates(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_triplet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_triplet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet candidates considered in this event
void SDL::Event::incrementNumberOfTrackletCandidates(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_tracklet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet candidates considered in this event
void SDL::Event::incrementNumberOfTrackletCandidates(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_tracklet_candidates_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate candidates considered in this event
void SDL::Event::incrementNumberOfTrackCandidateCandidates(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_trackcandidate_candidates_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_candidates_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate candidates considered in this event
void SDL::Event::incrementNumberOfTrackCandidateCandidates(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_trackcandidate_candidates_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_candidates_by_layer_endcap_[layer-1]++;
}*/

// Multiplicity of mini-doublet formed in this event
void SDL::Event::incrementNumberOfMiniDoublets(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_miniDoublet_by_layer_barrel_[layer-1]++;
    else
        n_miniDoublet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of segment formed in this event

/*void SDL::Event::incrementNumberOfSegments(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_segment_by_layer_barrel_[layer-1]++;
    else
        n_segment_by_layer_endcap_[layer-1]++;
}


// Multiplicity of triplet formed in this event
void SDL::Event::incrementNumberOfTriplets(SDL::Module& module)
{
    int layer = module.layer();
    int isbarrel = (module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_triplet_by_layer_barrel_[layer-1]++;
    else
        n_triplet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet formed in this event
void SDL::Event::incrementNumberOfTracklets(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_tracklet_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of tracklet formed in this event
void SDL::Event::incrementNumberOfTracklets(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_tracklet_by_layer_barrel_[layer-1]++;
    else
        n_tracklet_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate formed in this event
void SDL::Event::incrementNumberOfTrackCandidates(SDL::Layer& _layer)
{
    int layer = _layer.layerIdx();
    int isbarrel = (_layer.subdet() == SDL::Layer::Barrel);
    if (isbarrel)
        n_trackcandidate_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_by_layer_endcap_[layer-1]++;
}

// Multiplicity of track candidate formed in this event
void SDL::Event::incrementNumberOfTrackCandidates(SDL::Module& _module)
{
    int layer = _module.layer();
    int isbarrel = (_module.subdet() == SDL::Module::Barrel);
    if (isbarrel)
        n_trackcandidate_by_layer_barrel_[layer-1]++;
    else
        n_trackcandidate_by_layer_endcap_[layer-1]++;
}*/

namespace SDL
{
    std::ostream& operator<<(std::ostream& out, const Event& event)
    {

        out << "" << std::endl;
        out << "==============" << std::endl;
        out << "Printing Event" << std::endl;
        out << "==============" << std::endl;
        out << "" << std::endl;

        for (auto& modulePtr : event.modulePtrs_)
        {
            out << modulePtr;
        }

/*        for (auto& layerPtr : event.layerPtrs_)
        {
            out << layerPtr;
        }*/

        return out;
    }

    std::ostream& operator<<(std::ostream& out, const Event* event)
    {
        out << *event;
        return out;
    }

}


