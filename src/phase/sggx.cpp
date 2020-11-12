/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * This code is largely based on Wen Zhang's Mitsuba plugin from 
 * https://github.com/zhangwengame/SGGX-Plugin-for-Mitsuba
 */

#include <mitsuba/core/frame.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>


#include "sggx.h"

MTS_NAMESPACE_BEGIN

#if defined(SGGX_STATISTICS)
static StatsCounter avgSampleIterations("SGGX model",
        "Average rejection sampling iterations", EAverage);
#endif

/*!\plugin{sggx}{SGGX microflake distribution}
 * \order{5}
 * 
 * This plugin implements ``The SGGX Microflake Distribution'' \cite{heitz2015sggx}.
 * It can be used as spatially varying phase function for heterogeneous media.
 * 
 * This model receives its parameters from a medium with SGGX type, see \pluginref{heterogeneous}.
 */
class SGGXPhaseFunction : public PhaseFunction {
public:
    SGGXPhaseFunction(const Properties &props) : PhaseFunction(props) {
        
    }

    SGGXPhaseFunction(Stream *stream, InstanceManager *manager)
        : PhaseFunction(stream, manager) {
        configure();
    }

    virtual ~SGGXPhaseFunction() {
        fclose(stdout);
    }

    void configure() {
        PhaseFunction::configure();
        m_type = EAnisotropic | ENonSymmetric;
        // freopen("SGGX-phase-sample-test-cases.txt","w", stdout);
        freopen("SGGX-phase-eval-test-cases.txt","w", stdout);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        PhaseFunction::serialize(stream, manager);
    }

    Float eval(const PhaseFunctionSamplingRecord &pRec) const {
        // printf("wi = Vector3f(%ff, %ff, %ff);\n", pRec.wi.x, pRec.wi.y, pRec.wi.z);
        // printf("wo = Vector3f(%ff, %ff, %ff);\n", pRec.wo.x, pRec.wo.y, pRec.wo.z);
        Float Sxx = pRec.mRec.sggxS[0];
        Float Syy = pRec.mRec.sggxS[1];
        Float Szz = pRec.mRec.sggxS[2];
        Float Sxy = pRec.mRec.sggxS[3];
        Float Sxz = pRec.mRec.sggxS[4];
        Float Syz = pRec.mRec.sggxS[5];
        if (Sxx*Sxx + Syy*Syy + Szz*Szz + Sxy*Sxy + Sxz*Sxz + Syz*Syz < 1e-3)
            return 0.0;
        Vector wi(pRec.wi);
        Vector wo(pRec.wo);
        wi = normalize(wi);
        wo = normalize(wo);
        
        float value = eval_specular(wi, wo, Sxx, Syy, Szz, Sxy, Sxz, Syz);
        if (value != value) {
            value = 0.0;
        }

        printf("%.9f %.9f %.9f %.9f %.9f %.9f\n", Sxx, Syy, Szz, Sxy, Sxz, Syz);
        printf("%.9f %.9f %.9f\n", pRec.wi.x, pRec.wi.y, pRec.wi.z);
        printf("%.9f %.9f %.9f\n", pRec.wo.x, pRec.wo.y, pRec.wo.z);
        printf("%.9f\n", value);
        
        return value;
    }

    inline Float sample(PhaseFunctionSamplingRecord &pRec, Sampler *sampler) const {
        Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
        Sxx = pRec.mRec.sggxS[0];
        Syy = pRec.mRec.sggxS[1];
        Szz = pRec.mRec.sggxS[2];
        Sxy = pRec.mRec.sggxS[3];
        Sxz = pRec.mRec.sggxS[4];
        Syz = pRec.mRec.sggxS[5];
        if (Sxx*Sxx + Syy*Syy + Szz*Szz + Sxy*Sxy + Sxz*Sxz + Syz*Syz < 1e-3)
            return 0.0;
        
        Vector wi(pRec.wi);

        wi = normalize(wi);
        
        Point2 point = sampler->next2D();
        //printf("sampler = Point2f(%ff, %ff);\n", point.x, point.y);
        Vector wo = sample_specular(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz, point.x, point.y);
        wo = normalize(wo);
        if (wo.x != wo.x || wo.y != wo.y || wo.z != wo.z) {
            return 0.0;
        }
        // printf("%.9f %.9f %.9f %.9f %.9f %.9f\n", Sxx, Syy, Szz, Sxy, Sxz, Syz);
        // printf("%.9f %.9f\n", point.x, point.y);
        // printf("%.9f %.9f %.9f\n", pRec.wi.x, pRec.wi.y, pRec.wi.z);
        // printf("%.9f %.9f %.9f\n", wo.x, wo.y, wo.z);
        pRec.wo = Vector(wo.x,wo.y,wo.z);

        return 1.0f;
    }

    Float sample(PhaseFunctionSamplingRecord &pRec,
            Float &pdf, Sampler *sampler) const {
        if (fabs(sample(pRec, sampler)) <1e-6) {
            pdf = 0; return 0.0f;
        }
        pdf = eval(pRec);
        return 1.0f;
    }

    bool needsDirectionallyVaryingCoefficients() const { return true; }

    Float sigmaDirSGGX(Float *S, Vector v) const {
        Vector wi = Vector(v);
        // printf("v = Vector3f(%ff, %ff, %ff);\n", v.x, v.y, v.z);
        wi = normalize(wi);
        float ret = sigma(wi,S[0],S[1],S[2],S[3],S[4],S[5]);
        return ret;
    }

    Float sigmaDirMax() const {
        return 1.0;
    }
    
    std::string toString() const {
        std::ostringstream oss;
        oss << "SggxPhaseFunction[" << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    
};

MTS_IMPLEMENT_CLASS_S(SGGXPhaseFunction, false, PhaseFunction)
MTS_EXPORT_PLUGIN(SGGXPhaseFunction, "SGGX phase function");
MTS_NAMESPACE_END
