# include "Hit.cuh"

void createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int nMaxHits,unsigned int nMax2SHits)
{
    //nMaxHits and nMax2SHits are the maximum possible numbers
    cudaMallocManaged(&hitsInGPU.x, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.y, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.z, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.moduleIndices, nMaxHits * sizeof(unsigned int));

    cudaMallocManaged(&hitsInGPU.rts, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.phis, nMaxHits * sizeof(float));

    cudaMallocManaged(&hitsInGPU.edge2SMap, nMaxHits * sizeof(int)); //hits to edge hits map. Signed int
    cudaMallocManaged(&hitsInGPU.highEdgeXs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.highEdgeYs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeXs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeYs, nMax2SHits * sizeof(float));

    //counters
    cudaMallocManaged(&hitsInGPU.nHits, sizeof(unsigned int));
    *hitsInGPU.nHits = 0;
    cudaMallocManaged(&hitsInGPU.n2SHits, sizeof(unsigned int));
    *hitsInGPU.n2SHits = 0;
}

void addHitToMemory(struct hits& hitsInGPU, struct modules& modulesInGPU, float x, float y, float z, unsigned int detId)
{
    unsigned int idx = *hitsInGPU.nHits;
    unsigned int idxEdge2S = *hitsInGPU.n2SHits;

    hitsInGPU.x[idx] = x;
    hitsInGPU.y[idx] = y;
    hitsInGPU.z[idx] = z;
    hitsInGPU.rts[idx] = sqrt(x*x + y*y);
    hitsInGPU.phi[idx] = phi(x,y,z);
    unsigned int moduleIndex = detIdToIndex[detId]
    hitsInGPU.moduleIndices[idx] = moduleIndex;
    if(modulesInGPU.subdets[moduleIndex] == Endcap and modulesInGPU.moduleLayerType(moduleIndex) == TwoS)
    {
        float xhigh, yhigh, xlow, ylow;
        getEdgeHits(detId,x,y,xhigh,yhigh,xlow,ylow);
        hitsInGPU.edge2SMap[index] = idxEdge2S;
        hitsInGPU.highEdgeXs[idxEdge2S] = xhigh;
        hitsInGPU.highEdgeYs[idxEdge2S] = yhigh;
        hitsInGPU.lowEdgeXs[idxEdge2S] = xlow;
        hitsInGPU.lowEdgeYs[idxEdge2S] = ylow;

        (*hitsInGPU.n2SHits)++;
    }
    else
    {
        hitsInGPU.edge2SMap[index] = -1;
    }

    //set the hit ranges appropriately in the modules struct

    //start the index rolling if the module is encountered for the first time
    if(modulesInGPU.hitRanges[moduleIndex * 2] == -1)
    {
        modulesInGPU.hitRanges[moduleIndex * 2] = idx;
    }
    //always update the end index
    modulesInGPU.hitRanges[moduleIndex * 2 + 1] = idx;

    (*hitsInGPU.nHits)++;
    
}

inline float phi(float x, float y, float z)
{
    phi_mpi_pi(M_PI + SDL::MathUtil::ATan2(-y_, -x_));
}
inline float phi_mpi_pi(float phi)
{
    if (isnan(x))
    {
       printf("phi_mpi_pi() function called with NaN\n");
        return x;
    }

    while (x >= M_PI)
        x -= 2. * M_PI;

    while (x < -M_PI)
        x += 2. * M_PI;

    return x;
}

float deltaPhi(float x1, float y1, float z1, float x2, float y2, float z2)
{
    phi1 = phi(x1,y1,z1);
    phi2 = phi(x2,y2,z2);
    return phi_mpi_pi(std::abs(phi2 - phi1));
}

float deltaPhiChange(float x1, float y1, float z1, float x2, float y2, float z2)
{
    return deltaPhi(x1,y1,z1,x2-x1, y2-y1, z2-z1);
}

void getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow)
{
    float phi = endcapGeometry.getCentroidPhi(detId);
    xhigh = x + 2.5 * cos(phi);
    yhigh = x + 2.5 * sin(phi);
    xlow = x - 2.5 * cos(phi);
    ylow = x - 2.5 * sin(phi);
}