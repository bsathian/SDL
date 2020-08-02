# include "MiniDoublet.cuh"

void createMDsInUnifiedMemory(struct miniDoublets& mdsInGPU, unsigned int maxMDs, unsigned int nModules)
{
    cudaMallocManaged(&mdsInGPU.hitIndices, maxMDs * 2 * sizeof(unsigned int));
    cudaMallocManaged(&mdsInGPU.moduleIndices, maxMDs * sizeof(unsigned int));
    cudaMallocManaged(&mdsInGPU.pixelModuleFlag, maxMDs * sizeof(short));
    cudaMallocManaged(&mdsInGPU.dphichanges, maxMDs * sizeof(float));

    cudaMallocManaged(&mdsInGPU.nMDs, nModules * sizeof(int));

    cudaMallocManaged(&mdsInGPU.dzs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.dphis, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.shiftedXs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.shiftedYs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.shiftedZs, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.noShiftedDz, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.noShiftedDphis, maxMDs * sizeof(float));
    cudaMallocManaged(&mdsInGPU.noShiftedDphiChanges, maxMDs * sizeof(float));
}

void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, unsigned int lowerModuleIdx, float dz, float dphi, float dphichange, float shfitedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx)
{
    //the index into which this MD needs to be written will be computed in the kernel
    //nMDs variable will be incremented in the kernel, no need to worry about that here
    
    mdsInGPU.hitIndices[idx * 2] = lowerHitIdx;
    mdsInGPU.hitIndices[idx * 2 + 1] = upperHitIdx;
    mdsInGPU.moduleIndices[idx] = lowerModuleIdx;
    if(modulesInGPU.moduleType(lowerModuleIdx) == PS
    {
        if(modulesInGPU.moduleLayerType(lowerModuleIdx) == Pixel)
        {
            pixelModuleFlag = 0;
        }
        else
        {
            pixelModuleFlag = 1;
        }
    }
    else
    {
        pixelModuleFlag = -1;
    }

    mdsInGPU.dphichanges[idx] = dphichange;

    mdsInGPU.dphis[idx] = dphi;
    mdsInGPU.dzs[idx] = dz;
    mdsInGPU.shiftedXs[idx] = shiftedX;
    mdsInGPU.shiftedYs[idx] = shiftedY;
    mdsInGPU.shiftedZs[idx] = shiftedZ;

    mdsInGPU.noShiftedDzs[idx] = noshiftedDz;
    mdsInGPU.noShiftedDphis[idx] = noShiftedDphi;
    mdsInGPU.noShfitedDphiChanges[idx] = noShiftedDphiChange;
}

bool runMiniDoubletDefaultAlgoBarrel(struct modules& modulesInGPU, struct hits& hitsInGPU, float& dz, float& dPhi, float& dPhiChange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noshiftedDz, float& noShiftedDphi, float& noShiftedDphichange)
{
}

bool runMiniDoubletDefaultAlgoEndcap(struct modules& modulesInGPU, struct hits& hitsInGPU, float& drt, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange)
{

}

bool runMiniDoubletDefaultAlgo(struct modules& modulesInGPU, struct hits& hitsInGPU, float& dz, float& dphi, float& dphichange, float& shiftedX, float& shiftedY, float& shiftedZ, float& noShiftedDz, float& noShiftedDphi, float& noShiftedDphiChange)
{

}

float dPhiThreshold(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex, unsigned int moduleIndex)
{

}

inline float isTighterTiltedModules(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    // The "tighter" tilted modules are the subset of tilted modules that have smaller spacing
    // This is the same as what was previously considered as"isNormalTiltedModules"
    // See Figure 9.1 of https://cds.cern.ch/record/2272264/files/CMS-TDR-014.pdf
    short subdet = modulesInGPU.subdets[moduleIndex];
    short layer = modulesInGPU.layers[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];
    short rod = modulesInGPU.rods[moduleIndex];

    if (
           (subdet == Barrel and side != Center and layer== 3)
           or (subdet == Barrel and side == NegZ and layer == 2 and rod > 5)
           or (subdet == Barrel and side == PosZ and layer == 2 and rod < 8)
           or (subdet == Barrel and side == NegZ and layer == 1 and rod > 9)
           or (subdet == Barrel and side == PosZ and layer == 1 and rod < 4)
       )
        return true;
    else
        return false;

}

inline float moduleGapSize(struct modules& modulesInGPU, unsigned int moduleIndex)
{
    float miniDeltaTilted[] = {0.26, 0.26, 0.26};
    float miniDeltaLooseTilted[] =  {0.4,0.4,0.4};
    float miniDeltaFlat[] =  {0.26, 0.16, 0.16, 0.18, 0.18, 0.18};
    float miniDeltaEndcap[5][15];

    for (size_t i = 0; i < 5; i++)
    {
        for (size_t j = 0; j < 15; j++)
        {
            if (i == 0 || i == 1)
            {
                if (j < 10)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
            else if (i == 2 || i == 3)
            {
                if (j < 8)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j]  = 0.18;
                }
            }
            else
            {
                if (j < 9)
                {
                    miniDeltaEndcap[i][j] = 0.4;
                }
                else
                {
                    miniDeltaEndcap[i][j] = 0.18;
                }
            }
        }
    }

    unsigned int iL = modulesInGPU.layers[moduleIndex]-1;
    unsigned int iR = modulesInGPU.rings[moduleIndex] - 1;
    short subdet = modulesInGPU.subdets[moduleIndex];
    short side = modulesInGPU.sides[moduleIndex];

    float moduleSeparation = 0;

    if (subdet == Barrel and side == Center)
    {
        moduleSeparation = miniDeltaFlat[iL];
    }
    else if (isTighterTiltedModules(modulesInGPU, moduleIndex))
    {
        moduleSeparation = miniDeltaTilted[iL];
    }
    else if (subdet == Endcap)
    {
        moduleSeparation = miniDeltaEndcap[iL][iR];
    }
    else //Loose tilted modules
    {
        moduleSeparation = miniDeltaLooseTilted[iL];
    }
}

void shiftStripHits(struct modules& modulesInGPU, struct hits& hitsInGPU, unsigned int lowerModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords)
{

    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry
    // SDL::tiltedGeometry

    // Some variables relevant to the function
    float xp; // pixel x (pixel hit x)
    float yp; // pixel y (pixel hit y)
    float xa; // "anchor" x (the anchor position on the strip module plane from pixel hit)
    float ya; // "anchor" y (the anchor position on the strip module plane from pixel hit)
    float xo; // old x (before the strip hit is moved up or down)
    float yo; // old y (before the strip hit is moved up or down)
    float xn; // new x (after the strip hit is moved up or down)
    float yn; // new y (after the strip hit is moved up or down)
    float abszn; // new z in absolute value
    float zn; // new z with the sign (+/-) accounted
    float angleA; // in r-z plane the theta of the pixel hit in polar coordinate is the angleA
    float angleB; // this is the angle of tilted module in r-z plane ("drdz"), for endcap this is 90 degrees
    bool isEndcap; // If endcap, drdz = infinity
    unsigned int pixelHitIndex; // Pointer to the pixel hit
    unsigned int stripHitIndex; // Pointer to the strip hit
    float moduleSeparation;
    float drprime; // The radial shift size in x-y plane projection
    float drprime_x; // x-component of drprime
    float drprime_y; // y-component of drprime
    float slope; // The slope of the possible strip hits for a given module in x-y plane
    float absArctanSlope;
    float angleM; // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
    float absdzprime; // The distance between the two points after shifting
    float drdz_;

    // Assign hit pointers based on their hit type
    if (modulesInGPU.moduleType(lowerModuleIndex) == PS)
    {
        if (modulesInGPU.moduleLayerType(lowerModuleIndex)== Pixel)
        {
            pixelHitIndex = lowerHitIndex;
            stripHitIndex = upperHitIndex;
        }
        else
        {
            pixelHitIndex = upperHitIndex;
            stripHitIndex = lowerHitIndex;
        }
    }
    else // if (lowerModule.moduleType() == SDL::Module::TwoS) // If it is a TwoS module (if this is called likely an endcap module) then anchor the inner hit and shift the outer hit
    {
        pixelHitIndex = lowerHitIndex; // Even though in this case the "pixelHitPtr" is really just a strip hit, we pretend it is the anchoring pixel hit
        stripHitIndex = upperHitIndex;
    }

    // If it is endcap some of the math gets simplified (and also computers don't like infinities)
    isEndcap = modulesInGPU.subdets[lowerModuleIndex]== SDL::Module::Endcap;

    // NOTE: TODO: Keep in mind that the sin(atan) function can be simplifed to something like x / sqrt(1 + x^2) and similar for cos
    // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time

    // The pixel hit is used to compute the angleA which is the theta in polar coordinate
    // angleA = std::atan(pixelHitPtr->rt() / pixelHitPtr->z() + (pixelHitPtr->z() < 0 ? M_PI : 0)); // Shift by pi if the z is negative so that the value of the angleA stays between 0 to pi and not -pi/2 to pi/2

    angleA = fabs(std::atan(hitsInGPU.rts[pixelHitIndex] / hitsInGPU.zs[pixelHitIndex]));
    // angleB = isEndcap ? M_PI / 2. : -std::atan(tiltedGeometry.getDrDz(detid) * (lowerModule.side() == SDL::Module::PosZ ? -1 : 1)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa
    drdz_ = modulesInGPU.drdzs[lowerModuleIndex];
    angleB = ((isEndcap) ? M_PI / 2. : atan(drdz_)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa


    moduleSeparation = moduleGapSize(modulesInGPU, moduleIndex);

    // Sign flips if the pixel is later layer
    if (modulesInGPU.moduleType(lowerModuleIndex) == PS and modulesInGPU.moduleLayerType(lowerModuleIndex) != Pixel)
    {
        moduleSeparation *= -1;
    }

    drprime = (moduleSeparation / std::sin(angleA + angleB)) * std::sin(angleA);
    slope = modulesInGPU.slopes[moduleIndex];

    // Compute arctan of the slope and take care of the slope = infinity case
    absArctanSlope = ((slope != SDL_INF) ? fabs(std::atan(slope)) : M_PI / 2); // Since C++ can't represent infinity, SDL_INF = 123456789 was used to represent infinity in the data table

    // The pixel hit position
    xp = hitsInGPU.xs[pixelHitIndex];
    yp = hitsInGPU.ys[pixelHitIndex];

    // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
    if (xp > 0 and yp > 0)
    {
        angleM = absArctanSlope;
    }
    else if (xp > 0 and yp < 0)
    {
        angleM = M_PI - absArctanSlope;
    }
    else if (xp < 0 and yp < 0)
    {
        angleM = M_PI + absArctanSlope;
    }
    else // if (xp < 0 and yp > 0)
    {
        angleM = 2 * M_PI - absArctanSlope;
    }

    // Then since the angleM sign is taken care of properly
    drprime_x = drprime * std::sin(angleM);
    drprime_y = drprime * std::cos(angleM);

    // The new anchor position is
    xa = xp + drprime_x;
    ya = yp + drprime_y;

    // The original strip hit position
    xo = hitsInGPU.xs[stripHitIndex];
    yo = hitsInGPU.ys[stripHitIndex];

    // Compute the new strip hit position (if the slope vaule is in special condition take care of the exceptions)
    if (slope == SDL_INF) // Special value designated for tilted module when the slope is exactly infinity (module lying along y-axis)
    {
        xn = xa; // New x point is simply where the anchor is
        yn = yo; // No shift in y
    }
    else if (slope == 0)
    {
        xn = xo; // New x point is simply where the anchor is
        yn = ya; // No shift in y
    }
    else
    {
        xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
        yn = (xn - xa) * slope + ya; // new yn
    }

    // Computing new Z position
    absdzprime = fabs(moduleSeparation / std::sin(angleA + angleB) * std::cos(angleA)); // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

    // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
    if (lowerModule.moduleLayerType() == SDL::Module::Pixel)
    {
        abszn = std::abs(hitsInGPU.zs[pixelHitIndex]) + absdzprime;
    }
    else
    {
        abszn = std::abs(hitsInGPU.zs[pixelHitIndex]) - absdzprime;
    }

    zn = abszn * ((hitsInGPU.zs[pixelHitIndex] > 0) ? 1 : -1); // Apply the sign of the zn

/*    if (logLevel == SDL::Log_Debug3)
    {
        SDL::cout << upperHit << std::endl;
        SDL::cout << lowerHit << std::endl;
        SDL::cout <<  " lowerModule.moduleType()==SDL::Module::PS: " << (lowerModule.moduleType()==SDL::Module::PS) <<  std::endl;
        SDL::cout <<  " lowerModule.moduleLayerType()==SDL::Module::Pixel: " << (lowerModule.moduleLayerType()==SDL::Module::Pixel) <<  std::endl;
        SDL::cout <<  " pixelHitPtr: " << pixelHitPtr <<  std::endl;
        SDL::cout <<  " stripHitPtr: " << stripHitPtr <<  std::endl;
        SDL::cout <<  " detid: " << detid <<  std::endl;
        SDL::cout <<  " isEndcap: " << isEndcap <<  std::endl;
        SDL::cout <<  " pixelHitPtr->rt(): " << pixelHitPtr->rt() <<  std::endl;
        SDL::cout <<  " pixelHitPtr->z(): " << pixelHitPtr->z() <<  std::endl;
        SDL::cout <<  " angleA: " << angleA <<  std::endl;
        SDL::cout <<  " angleB: " << angleB <<  std::endl;
        SDL::cout <<  " moduleSeparation: " << moduleSeparation <<  std::endl;
        SDL::cout <<  " drprime: " << drprime <<  std::endl;
        SDL::cout <<  " slope: " << slope <<  std::endl;
        SDL::cout <<  " absArctanSlope: " << absArctanSlope <<  std::endl;
        SDL::cout <<  " angleM: " << angleM <<  std::endl;
        SDL::cout <<  " drprime_x: " << drprime_x <<  std::endl;
        SDL::cout <<  " drprime_y: " << drprime_y <<  std::endl;
        SDL::cout <<  " xa: " << xa <<  std::endl;
        SDL::cout <<  " ya: " << ya <<  std::endl;
        SDL::cout <<  " xo: " << xo <<  std::endl;
        SDL::cout <<  " yo: " << yo <<  std::endl;
        SDL::cout <<  " xn: " << xn <<  std::endl;
        SDL::cout <<  " yn: " << yn <<  std::endl;
        SDL::cout <<  " absdzprime: " << absdzprime <<  std::endl;
        SDL::cout <<  " zn: " << zn <<  std::endl;
    }*/

    shiftedCoords[0] = xn;
    shiftedCoords[1] = yn;
    shiftedCoords[2] = zn;

}


