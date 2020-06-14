# include "ModulePrimitive.cuh"


SDL::ModulePrimitive::ModulePrimitive()
{
    setDetId(0);
}

SDL::ModulePrimitive::ModulePrimitive(unsigned int detId)
{
    setDetId(detId);
}

SDL::ModulePrimitive::ModulePrimitive(const ModulePrimitive& module)
{
    setDetId(module.detId());
}

SDL::ModulePrimitive::~ModulePrimitive()
{
}

const unsigned short& SDL::ModulePrimitive::subdet() const
{
    return subdet_;
}

const unsigned short& SDL::ModulePrimitive::side() const
{
    return side_;
}

const unsigned short& SDL::ModulePrimitive::layer() const
{
    return layer_;
}

const unsigned short& SDL::ModulePrimitive::rod() const
{
    return rod_;
}

const unsigned short& SDL::ModulePrimitive::ring() const
{
    return ring_;
}

const unsigned short& SDL::ModulePrimitive::module() const
{
    return module_;
}

const unsigned short& SDL::ModulePrimitive::isLower() const
{
    return isLower_;
}

const unsigned int& SDL::ModulePrimitive::detId() const
{
    return detId_;
}

const unsigned int& SDL::ModulePrimitive::partnerDetId() const
{
    return partnerDetId_;
}

const bool& SDL::ModulePrimitive::isInverted() const
{
    return isInverted_;
}

const SDL::ModulePrimitive::ModuleType& SDL::ModulePrimitive::moduleType() const
{
    return moduleType_;
}

const SDL::ModulePrimitive::ModuleLayerType& SDL::ModulePrimitive::moduleLayerType() const
{
    return moduleLayerType_;
}

void SDL::ModulePrimitive::setDetId(unsigned int detId)
{
    detId_ = detId;
    setDerivedQuantities();
}

void SDL::ModulePrimitive::setDerivedQuantities()
{
    subdet_ = parseSubdet(detId_);
    side_ = parseSide(detId_);
    layer_ = parseLayer(detId_);
    rod_ = parseRod(detId_);
    ring_ = parseRing(detId_);
    module_ = parseModule(detId_);
    isLower_ = parseIsLower(detId_);
    isInverted_ = parseIsInverted(detId_);
    partnerDetId_ = parsePartnerDetId(detId_);
    moduleType_ = parseModuleType(detId_);
    moduleLayerType_ = parseModuleLayerType(detId_);
}

unsigned short SDL::ModulePrimitive::parseSubdet(unsigned int detId)
{
    return (detId & (7 << 25)) >> 25;
}

unsigned short SDL::ModulePrimitive::parseSide(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::ModulePrimitive::Endcap)
    {
        return (detId & (3 << 23)) >> 23;
    }
    else if (parseSubdet(detId) == SDL::ModulePrimitive::Barrel)
    {
        return (detId & (3 << 18)) >> 18;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::ModulePrimitive::parseLayer(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::ModulePrimitive::Endcap)
    {
        return (detId & (7 << 18)) >> 18;
    }
    else if (parseSubdet(detId) == SDL::ModulePrimitive::Barrel)
    {
        return (detId & (7 << 20)) >> 20;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::ModulePrimitive::parseRod(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::ModulePrimitive::Endcap)
    {
        return 0;
    }
    else if (parseSubdet(detId) == SDL::ModulePrimitive::Barrel)
    {
        return (detId & (127 << 10)) >> 10;
    }
    else
    {
        return 0;
    }
}

unsigned short SDL::ModulePrimitive::parseRing(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::ModulePrimitive::Endcap)
    {
        return (detId & (15 << 12)) >> 12;
    }
    else if (parseSubdet(detId) == SDL::ModulePrimitive::Barrel)
    {
        return 0;
    }
    else
    {
        return 0;
    }

}

unsigned short SDL::ModulePrimitive::parseModule(unsigned int detId)
{
    return (detId & (127 << 2)) >> 2;
}

unsigned short SDL::ModulePrimitive::parseIsLower(unsigned int detId)
{
    return ((parseIsInverted(detId)) ? !(detId & 1) : (detId & 1));
}

bool SDL::ModulePrimitive::parseIsInverted(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::ModulePrimitive::Endcap)
    {
        if (parseSide(detId) == SDL::ModulePrimitive::NegZ)
        {
            return parseModule(detId) % 2 == 1;
        }
        else if (parseSide(detId) == SDL::ModulePrimitive::PosZ)
        {
            return parseModule(detId) % 2 == 0;
        }
        else
        {
            SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
            return 0;
        }
    }
    else if (parseSubdet(detId) == SDL::ModulePrimitive::Barrel)
    {
        if (parseSide(detId) == SDL::ModulePrimitive::Center)
        {
            if (parseLayer(detId) <= 3)
            {
                return parseModule(detId) % 2 == 1;
            }
            else if (parseLayer(detId) >= 4)
            {
                return parseModule(detId) % 2 == 0;
            }
            else
            {
                SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
                return 0;
            }
        }
        else if (parseSide(detId) == SDL::ModulePrimitive::NegZ or parseSide(detId) == SDL::ModulePrimitive::PosZ)
        {
            if (parseLayer(detId) <= 2)
            {
                return parseModule(detId) % 2 == 1;
            }
            else if (parseLayer(detId) == 3)
            {
                return parseModule(detId) % 2 == 0;
            }
            else
            {
                SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
                return 0;
            }
        }
        else
        {
            SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
            return 0;
        }
    }
    else
    {
        SDL::cout << "Warning: parseIsInverted() categorization failed" << std::endl;
        return 0;
    }
}

unsigned int SDL::ModulePrimitive::parsePartnerDetId(unsigned int detId)
{
    if (parseIsLower(detId))
        return ((parseIsInverted(detId)) ? detId - 1 : detId + 1);
    else
        return ((parseIsInverted(detId)) ? detId + 1 : detId - 1);
}

SDL::ModulePrimitive::ModuleType SDL::ModulePrimitive::parseModuleType(unsigned int detId)
{
    if (parseSubdet(detId) == SDL::ModulePrimitive::Barrel)
    { 
        if (parseLayer(detId) <= 3)
            return SDL::ModulePrimitive::PS;
        else
            return SDL::ModulePrimitive::TwoS;
    }
    else
    {
        if (parseLayer(detId) <= 2)
        {
            if (parseRing(detId) <= 10)
                return SDL::ModulePrimitive::PS;
            else
                return SDL::ModulePrimitive::TwoS;
        }
        else
        {
            if (parseRing(detId) <= 7)
                return SDL::ModulePrimitive::PS;
            else
                return SDL::ModulePrimitive::TwoS;
        }
    }
}

SDL::ModulePrimitive::ModuleLayerType SDL::ModulePrimitive::parseModuleLayerType(unsigned int detId)
{
    if (parseModuleType(detId) == SDL::ModulePrimitive::TwoS)
        return SDL::ModulePrimitive::Strip;
    if (parseIsInverted(detId))
    {
        if (parseIsLower(detId))
            return SDL::ModulePrimitive::Strip;
        else
            return SDL::ModulePrimitive::Pixel;
    }
    else
    {
        if (parseIsLower(detId))
            return SDL::ModulePrimitive::Pixel;
        else
            return SDL::ModulePrimitive::Strip;
    }
}
