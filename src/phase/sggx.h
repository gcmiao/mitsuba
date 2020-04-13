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
 * This code is largely based on code samples from the supplementary
 * material of the paper ``The SGGX Microflake Distribution''.
 */

#if !defined(__SGGX_FIBER_DIST_H)
#define __SGGX_FIBER_DIST_H

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

typedef Vector Vector;

// This function implements Frisvad's algorithm 
void buildOrthonormalBasis(Vector& omega_1, Vector& omega_2, const Vector& omega_3)
{
    if (omega_3.z < -0.9999999f)
    {
        omega_1 = Vector(0.0f, -1.0f, 0.0f);
        omega_2 = Vector(-1.0f, 0.0f, 0.0f);
    }
    else {
        const float a = 1.0f / (1.0f + omega_3.z);
        const float b = -omega_3.x*omega_3.y*a;
        omega_1 = Vector(1.0f - omega_3.x*omega_3.x*a, b, -omega_3.x);
        omega_2 = Vector(b, 1.0f - omega_3.y*omega_3.y*a, -omega_3.y);
    }
}

float D(Vector wm,
    float S_xx, float S_yy, float S_zz,
    float S_xy, float S_xz, float S_yz)
{
    const float detS =
        S_xx*S_yy*S_zz - S_xx*S_yz*S_yz - S_yy*S_xz*S_xz - S_zz*S_xy*S_xy + 2.0f*S_xy*S_xz*S_yz;
    const float den = wm.x*wm.x*(S_yy*S_zz - S_yz*S_yz) + wm.y*wm.y*(S_xx*S_zz - S_xz*S_xz) + wm.z*wm.z*(S_xx*S_yy - S_xy*S_xy)
        + 2.0f*(wm.x*wm.y*(S_xz*S_yz - S_zz*S_xy) + wm.x*wm.z*(S_xy*S_yz - S_yy*S_xz) + wm.y*wm.z*(S_xy*S_xz - S_xx*S_yz));
    const float D = powf(fabsf(detS), 1.50f) / (M_PI*den*den);
    return D;
}

Vector sample_VNDF(const Vector wi,
    const float S_xx, const float S_yy, const float S_zz,
    const float S_xy, const float S_xz, const float S_yz,
    const float U1, const float U2)
{
    // generate sample (u, v, w)
    const float r = sqrtf(U1);
    const float phi = 2.0f*M_PI*U2;
    const float u = r*cosf(phi);
    const float v = r*sinf(phi);
    const float w = sqrtf(1.0f - u*u - v*v);
    // build orthonormal basis
    Vector wk, wj;
    buildOrthonormalBasis(wk, wj, wi);
    // project S in this basis
    const float S_kk = wk.x*wk.x*S_xx + wk.y*wk.y*S_yy + wk.z*wk.z*S_zz
        + 2.0f * (wk.x*wk.y*S_xy + wk.x*wk.z*S_xz + wk.y*wk.z*S_yz);
    const float S_jj = wj.x*wj.x*S_xx + wj.y*wj.y*S_yy + wj.z*wj.z*S_zz
        + 2.0f * (wj.x*wj.y*S_xy + wj.x*wj.z*S_xz + wj.y*wj.z*S_yz);
    const float S_ii = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
        + 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
    const float S_kj = wk.x*wj.x*S_xx + wk.y*wj.y*S_yy + wk.z*wj.z*S_zz
        + (wk.x*wj.y + wk.y*wj.x)*S_xy
        + (wk.x*wj.z + wk.z*wj.x)*S_xz
        + (wk.y*wj.z + wk.z*wj.y)*S_yz;
    const float S_ki = wk.x*wi.x*S_xx + wk.y*wi.y*S_yy + wk.z*wi.z*S_zz
        + (wk.x*wi.y + wk.y*wi.x)*S_xy + (wk.x*wi.z + wk.z*wi.x)*S_xz + (wk.y*wi.z + wk.z*wi.y)*S_yz;
    const float S_ji = wj.x*wi.x*S_xx + wj.y*wi.y*S_yy + wj.z*wi.z*S_zz
        + (wj.x*wi.y + wj.y*wi.x)*S_xy
        + (wj.x*wi.z + wj.z*wi.x)*S_xz
        + (wj.y*wi.z + wj.z*wi.y)*S_yz;
    // compute normal
    float sqrtDetSkji = sqrtf(fabsf(S_kk*S_jj*S_ii - S_kj*S_kj*S_ii - S_ki*S_ki*S_jj - S_ji*S_ji*S_kk + 2.0f*S_kj*S_ki*S_ji));
    float inv_sqrtS_ii = 1.0f / sqrtf(S_ii);
    float tmp = sqrtf(S_jj*S_ii - S_ji*S_ji);
    Vector Mk(sqrtDetSkji / tmp, 0.0f, 0.0f);
    Vector Mj(-inv_sqrtS_ii*(S_ki*S_ji - S_kj*S_ii) / tmp, inv_sqrtS_ii*tmp, 0);
    Vector Mi(inv_sqrtS_ii*S_ki, inv_sqrtS_ii*S_ji, inv_sqrtS_ii*S_ii);
    Vector wm_kji = normalize(u*Mk + v*Mj + w*Mi);
    // rotate back to world basis
    return wm_kji.x * wk + wm_kji.y * wj + wm_kji.z * wi;
}

// projected area
float sigma(Vector wi,
    float S_xx, float S_yy, float S_zz,
    float S_xy, float S_xz, float S_yz)
{
    const float sigma_squared = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
        + 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
    return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f; // conditional to avoid numerical errors
}

float eval_diffuse(Vector wi, Vector wo,
    const float S_xx, const float S_yy, const float S_zz,
    const float S_xy, const float S_xz, const float S_yz,
    const float U1, const float U2)
{
    // sample VNDF
    const Vector wm = sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
    // eval diffuse
    return 1.0f / M_PI * std::max(0.0f, dot(wo, wm));
}

float eval_specular(Vector wi, Vector wo,
    const float S_xx, const float S_yy, const float S_zz,
    const float S_xy, const float S_xz, const float S_yz)
{
    Vector wh = normalize(wi + wo);
    return 0.25f * D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) / sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz);
}

Vector sample_specular(const Vector wi,
    const float S_xx, const float S_yy, const float S_zz,
    const float S_xy, const float S_xz, const float S_yz,
    const float U1, const float U2)
{
    // sample VNDF
    const Vector wm = sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);
    // specular reflection
    const Vector wo = -wi + wm * 2.0f * dot(wm, wi);
    return wo;
}

MTS_NAMESPACE_END

#endif /* __SGGX_FIBER_DIST_H */
