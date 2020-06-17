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


