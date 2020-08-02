# include "Module.cuh"

void createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules)
{
    /* modules stucture object will be created in Event.cu*/

    cudaMallocManaged(&modulesInGPU.detIds,nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.moduleMap,nModules * 40 * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.nConnectedModules,nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.drdzs,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.slopes,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.nModules,sizeof(unsigned int));

    cudaMallocManaged(&modulesInGPU.layers,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rings,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.modules,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rods,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.subdets,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.sides,nModules * sizeof(short));

    cudaMallocManaged(&modulesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.mdRanges,nModules * 2 * sizeof(int));

    *modulesInGPU.nModules = nModules;

}


void loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules)
{
    /*modules structure object will be created in Event.cu*/
    /* Load the whole text file into the unordered_map first*/

    std::ifstream ifile;
    ifile.open("data/centroid.txt");
    std::string line;
    unsigned int counter = 0;

    while(std::getline(ifile,line))
    {
        std::stringstream ss(line);
        std::string token;
        bool flag = 0;
        
        while(std::getline(ss,token,','))
        {
            if(flag == 1) break;
            detIdToIndex[counter] = atoi(ss);
            flag = 1;
            counter++;
        }
    }
    nModules = counter+1;

    createModulesInUnifiedMemory(modulesInGPU,nModules);

    for(auto& it = detIdToIndex.begin(); it != detIdToIndex.end(); it++)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second; 
        modulesInGPU.detIds[index] = detId;
        unsigned short layer,ring,rod,module,subdet,side;
        setDerivedQuantities(detId,layer,ring,rod,module,subdet,side); 
        modulesInGPU.layers[index] = layer;
        modulesInGPU.rings[index] = ring;
        modulesInGPU.rods[index] = rod;
        modulesInGPU.modules[index] = module;
        modulesInGPU.subdets[index] = subdet;
        modulesInGPU.sides[index] = side; 

        modulesInGPU.drdzs[index] = tiltedGeometry.getDrDz(detId);
        modulesInGPU.slopes[index] = (subdet == Endcap) ? endcapGeometry.getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
    }
    fillConnectedModuleArray(modulesInGPU,nModules);
    resetObjectRanges(modulesInGPU,nModules);
} 

void fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules)
{
    for(auto& it = detIdToIndex.begin(); it != detIdToIndex.end(); it++)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        std::vector<unsigned int>& connectedModules = moduleConnectionMap.getConnectedModuleDetails(detId);
        nConnectedModules[index] = connectedModules.size();
        for(unsigned int i = 0; i< nConnectedModules[index];i++)
        {
            moduleMap[index][i] = detIdToIndex[connectedModules[i]];
        }
    }
}

void setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side)
{
    subdet = (detId & (7 << 25)) >> 25;
    side = (subdet == Endcap) ? (detId & (3 << 23)) >> 23 : (detId & (3 << 18)) >> 18;
    layer = (subdet == Endcap) ? (detId & (7 << 18)) >> 18 : (detId & (7 << 20)) >> 20;
    ring = (subdet == Endcap) ? (detId & (15 << 12)) >> 12 : 0;
    module = (detId & (127 << 2)) >> 2;
    rod = (subdet == Endcap) ? 0 : (detId & (127 << 10)) >> 10;
}

//auxilliary functions - will be called as needed
bool modules::isInverted(unsigned int index)
{
    if (subdets[index] == Endcap)
    {
        if (sides[index] == NegZ)
        {
            return modules[index] % 2 == 1;
        }
        else if (sides[index] == PosZ)
        {
            return modules[index] % 2 == 0;
        }
        else
        {
            return 0;
        }
    }
    else if (subdets[index] == Barrel)
    {
        if (sides[index] == Center)
        {
            if (layers[index] <= 3)
            {
                return modules[index] % 2 == 1;
            }
            else if (layers[index] >= 4)
            {
                return modules[index] % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else if (sides[index] == NegZ or sides[index] == PosZ)
        {
            if (layers[index] <= 2)
            {
                return modules[index] % 2 == 1;
            }
            else if (layers[index] == 3)
            {
                return modules[index] % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

bool modules::isLower(unsigned int index)
{
    return (isInverted(detId)) ? !(detId & 1) : (detId & 1);
}

unsigned int modules::partnerDetIdIndex(unsigned int index)
{
    /*We need to ensure modules with successive det Ids are right next to each other
    or we're dead*/
    if(isLower(index))
    {
        return (isInverted(index) ? index - 1: index + 1);
    }
    else
    {
        return (isInverted(index) ? index + 1 : index - 1);
    }
}

ModuleType modules::moduleType(unsigned int index)
{
    if(subdets[index] == Barrel)
    {
        if(layers[index] <= 3)
        {
            return PS;
        }
        else
        {
            return TwoS;
        }
    }
    else
    {
        if(layers[index] <= 2)
        {
            if(rings[index] <= 10)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
        else
        {
            if(rings[index] <= 7)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
    }
}

ModuleLayerType modules::moduleLayerType(unsigned int index)
{
    if(moduleType(index) == TwoS)
    {
        return Strip;
    }
    if(isInverted(index))
    {
        if(isLower(index))
        {
            return Strip;
        }
        else
        {
            return Pixel;
        }
    }
    else
    {
        if(isLower(index))
        {
            return Pixel;
        }
        else
        {
            return Strip;
        }
    }
}

void resetObjectRanges(struct modules& modulesInGPU, int nModules)
{

#pragma omp parallel for default(shared)
    for(size_t i = 0; i<nModules *2; i++)
    {
        modulesInGPU.hitRanges[i] = -1;
        modulesInGPU.mdRanges[i] = -1;
    }

}

