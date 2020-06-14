#include "Module.cuh"

SDL::Module::Module()
{
    moduleInfo = new ModulePrimitive(0);
}

SDL::Module::Module(unsigned int detId)
{
    moduleInfo = new ModulePrimitive(detId);
}

SDL::Module::Module(SDL::ModulePrimitive* modulePrimitive)
{
    moduleInfo = modulePrimitive;
}

SDL::Module::Module(const Module& module)
{
    moduleInfo = new ModulePrimitive(module.detId());
}

SDL::Module::~Module()
{
}

const SDL::ModulePrimitive* SDL::Module::modulePrimitive() const
{
    return moduleInfo;
}

//Ensuring backwards compatibility
const unsigned short& SDL::Module::subdet() const
{
    return moduleInfo->subdet();
}

const unsigned short& SDL::Module::side() const
{
    return moduleInfo->side();
}

const unsigned short& SDL::Module::layer() const
{
    return moduleInfo->layer();
}

const unsigned short& SDL::Module::rod() const
{
    return moduleInfo->rod();
}

const unsigned short& SDL::Module::ring() const
{
    return moduleInfo->ring();
}

const unsigned short& SDL::Module::module() const
{
    return moduleInfo->module();
}

const unsigned short& SDL::Module::isLower() const
{
    return moduleInfo->isLower();
}

const unsigned int& SDL::Module::detId() const
{
    return moduleInfo->detId();
}

const unsigned int& SDL::Module::partnerDetId() const
{
    return moduleInfo->partnerDetId();
}

const bool& SDL::Module::isInverted() const
{
    return moduleInfo->isInverted();
}

const SDL::Module::ModuleType& SDL::Module::moduleType() const
{
    return (SDL::Module::ModuleType&)moduleInfo->moduleType();
}

const SDL::Module::ModuleLayerType& SDL::Module::moduleLayerType() const
{
    return (SDL::Module::ModuleLayerType&)moduleInfo->moduleLayerType();
}

const std::vector<SDL::Hit*>& SDL::Module::getHitPtrs() const
{
    return hits_;
}

const std::vector<SDL::MiniDoublet*>& SDL::Module::getMiniDoubletPtrs() const
{
    return miniDoublets_;
}

const std::vector<SDL::Segment*>& SDL::Module::getSegmentPtrs() const
{
    return segments_;
}

const std::vector<SDL::Triplet*>& SDL::Module::getTripletPtrs() const
{
    return triplets_;
}

const std::vector<SDL::Tracklet*>& SDL::Module::getTrackletPtrs() const
{
    return tracklets_;
}

//backwards compatibility
void SDL::Module::setDetId(unsigned int detId)
{
    moduleInfo->setDetId(detId);
}

void SDL::Module::addHit(SDL::Hit* hit)
{
    // Set the information on the module for where this hit resides
    // So we can swim backwards to find which module this hit resides
    // for any meta-object that contains this hit
//    hit->setModule(this);
    
    // Then add to the module
    hits_.push_back(hit);
}

void SDL::Module::addMiniDoublet(SDL::MiniDoublet* md)
{
    miniDoublets_.push_back(md);
}

void SDL::Module::addSegment(SDL::Segment* sg)
{
    segments_.push_back(sg);
}

void SDL::Module::addTriplet(SDL::Triplet* tp)
{
    triplets_.push_back(tp);
}

void SDL::Module::addTracklet(SDL::Tracklet* tp)
{
    tracklets_.push_back(tp);
}

namespace SDL
{
    std::ostream& operator<<(std::ostream& out, const Module& module)
    {
        out << "Module(detId=" << module.detId();
        out << ", subdet=" << (module.subdet() == SDL::Module::Barrel ? "Barrel" : "Endcap");
        out << ", side=" << (module.side() == SDL::Module::Center ? "Center" : "Side");
        out << ", layer=" << module.layer();
        out << ", rod=" << module.rod();
        out << ", ring=" << module.ring();
        out << ", module=" << module.module();
        out << ", moduleType=" << (module.moduleType() == SDL::Module::PS ? "PS" : "2S");
        out << ", moduleLayerType=" << (module.moduleLayerType() == SDL::Module::Pixel ? "Pixel" : "Strip");
        out << ", isLower=" << module.isLower();
        out << ", isInverted=" << module.isInverted();
        out << ", isNormalTitled=" << SDL::MiniDoublet::isTighterTiltedModules(*module.modulePrimitive());
        out << ")" << std::endl;
        // for (auto& hitPtr : module.hits_)
        //     out << hitPtr << std::endl;
        // for (auto& mdPtr : module.miniDoublets_)
        //     out << mdPtr << std::endl;
        // for (auto& sgPtr : module.segments_)
        //     out << sgPtr << std::endl;
        out << "" << std::endl;

        return out;
    }

    std::ostream& operator<<(std::ostream& out, const Module* module)
    {
        out << *module;
        return out;
    }
}
