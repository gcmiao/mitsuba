/*
    Copyright (c) 2019-2020 by Dennis den Brok.

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

#include <algorithm>
#include <iostream>
#include <vector>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/shape.h>

#include <hdf5/serial/H5Cpp.h>

#include <eigen3/Eigen/Core>

#include <CGAL/Barycentric_coordinates_2/Triangle_coordinates_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>

MTS_NAMESPACE_BEGIN

/*!\plugin[btftensor]{btf\_tensor}{Uncompressed Bidirectional Texture Function}
 * \order{20}
 * \parameters{
 *     \parameter{filename}{\String}{Path to tensor file}
 *     \parameter{x, y}{\Integer}{
 *       Region of interest offset that is applied to UV lookups
 *     }
 *     \parameter{w, h}{\Integer}{
 *       Region of interest size
 *     }
 *     \parameter{uscale, vscale}{\Float}{
 *       Multiplicative factors that should be applied to UV lookups
 *     }
 * }
 *
 * This plugin implements rendering uncompressed BTF tensors, it is used
 * in the paper ``Per-Image Super-Resolution for Material BTFs''
 * by Dennis den Brok, Sebastian Merzbach, Michael Weinmann, and 
 * Reinhard Klein \cite{denbrok2020iccp}.
 */
class BTFTensor : public BSDF {
public:
    BTFTensor(const Properties& props) : BSDF(props) {
        m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename", ""));
        if (!fs::exists(m_filename))
            Log(EError, "BTF tensor file \"%s\" could not be found!",
                m_filename.string().c_str());
        m_uscale = props.hasProperty("uscale") ? props.getFloat("uscale") : 1.0;
        m_vscale = props.hasProperty("vscale") ? props.getFloat("vscale") : 1.0;
        m_roi.x = props.hasProperty("x") ? props.getInteger("x") : 0;
        m_roi.y = props.hasProperty("y") ? props.getInteger("y") : 0;
        m_roi.w = props.hasProperty("w") ? props.getInteger("w") : 0;
        m_roi.h = props.hasProperty("h") ? props.getInteger("h") : 0;
    }
    
    BTFTensor(Stream* stream, InstanceManager* manager) : BSDF(stream, manager) {
        this->configure();
    }
    
    void
    configure() {
        Log(EInfo, "Reading tensor from \"%s\" ..", m_filename.c_str());
        H5::H5File mat_file(m_filename.c_str(), H5F_ACC_RDONLY);
        
        // Read data tensor.
        try {
            H5::DataSet ds_tensor = mat_file.openDataSet("/tensor");
            H5::DataSpace data_tensor = ds_tensor.getSpace();
            hsize_t dims[5];
            data_tensor.getSimpleExtentDims(dims, NULL);
            m_numChans = (unsigned int)dims[0];
            m_numLights = (unsigned int)dims[1];
            m_numViews = (unsigned int)dims[2];
            m_width = (unsigned int)dims[3];
            m_height = (unsigned int)dims[4];
            m_tensor.resize(m_numChans * (m_numLights + m_numBottomRing) * (m_numViews + m_numBottomRing) * m_width * m_height);
            hsize_t dims_out[5];
            hsize_t offs_out[5];
            hsize_t count_out[5];
            dims_out[0] = m_numChans;
            dims_out[1] = m_numLights + m_numBottomRing;
            dims_out[2] = m_numViews + m_numBottomRing;
            dims_out[3] = m_width;
            dims_out[4] = m_height;
            offs_out[0] = offs_out[1] = offs_out[2] = offs_out[3] = offs_out[4] = 0;
            count_out[0] = m_numChans;
            count_out[1] = m_numLights;
            count_out[2] = m_numViews;
            count_out[3] = m_width;
            count_out[4] = m_height;
            Log(EInfo, "n_ch = %d", m_numChans);
            H5::DataSpace mem_space(5, dims_out);
            mem_space.selectHyperslab(H5S_SELECT_SET, count_out, offs_out);
            ds_tensor.read(m_tensor.data(), H5::PredType::NATIVE_FLOAT, mem_space);
            
            // Read sample coordinates and precompute Delaunay triangulations.
            std::vector<double> L_(3 * m_numLights), V_(3 * m_numViews);
            H5::DataSet ds_L = mat_file.openDataSet("/L");
            H5::DataSet ds_V = mat_file.openDataSet("/V");
            ds_L.read(L_.data(), H5::PredType::NATIVE_DOUBLE);
            ds_V.read(V_.data(), H5::PredType::NATIVE_DOUBLE);
    
            // This hack appears necessary to allow CGAL to handle samples very close to theta = pi/2.
            double factor = 1 / (-1e-2 + sqrt(1 + (-1e-2) * (-1e-2)));
            std::vector<std::pair<K::Point_2, int> > points;
            for (unsigned int i = 0; i < m_numLights; ++i) {
                points.push_back(std::make_pair(DelaunayTriangulation::Point(0.5 * L_[m_numLights * 0 + i] / (L_[m_numLights * 2 + i] + 1) + 0.5,
                                                0.5 * L_[m_numLights * 1 + i] / (L_[m_numLights * 2 + i] + 1) + 0.5),
                                                i));
            }
            for (unsigned int i = 0; i < m_numBottomRing; ++i) {
                points.push_back(std::make_pair(DelaunayTriangulation::Point(0.5 * cos((double)i / (double)m_numBottomRing * 2 * M_PI) * factor + 0.5,
                                                0.5 * sin((double)i / (double)m_numBottomRing * 2 * M_PI) * factor + 0.5),
                                                i + m_numLights));
            }
            m_triangLight.insert(points.begin(), points.end());
            points.clear();
            for (unsigned int i = 0; i < m_numViews; ++i) {
                points.push_back(std::make_pair(DelaunayTriangulation::Point(0.5 * V_[m_numViews * 0 + i] / (V_[m_numViews * 2 + i] + 1) + 0.5,
                                                0.5 * V_[m_numViews * 1 + i] / (V_[m_numViews * 2 + i] + 1) + 0.5),
                                                i));
            }
            for (unsigned int i = 0; i < m_numBottomRing; ++i) {
                points.push_back(std::make_pair(DelaunayTriangulation::Point(0.5 * cos((double)i / (double)m_numBottomRing * 2 * M_PI) * factor + 0.5,
                                                0.5 * sin((double)i / (double)m_numBottomRing * 2 * M_PI) * factor + 0.5),
                                                i + m_numViews));
            }
            m_triangView.insert(points.begin(), points.end());
            
            // If no ROI is specified, set to full texture size.
            if (m_roi.w == 0) {
                m_roi.w = m_width;
            }
            if (m_roi.h == 0) {
                m_roi.h = m_height;
            }
            
            m_components.clear();
            m_components.push_back(EDiffuseReflection | EFrontSide | ESpatiallyVarying | EAnisotropic);
            
            BSDF::configure();
            
            m_components.clear();
            m_components.push_back(EDiffuseReflection | EFrontSide | ESpatiallyVarying | EAnisotropic);
        } catch (std::runtime_error ex) {
            Log(EError, ex.what());
        }
    }
    
    Spectrum
    eval(const BSDFSamplingRecord& bRec, EMeasure measure) const {
        if (!(bRec.typeMask & EDiffuseReflection) ||
                measure != ESolidAngle ||
                Frame::cosTheta(bRec.wi) <= 0 ||
                Frame::cosTheta(bRec.wo) <= 0) {
            return Spectrum(0.0f);
        }
        
        unsigned int x = (unsigned int)round(bRec.its.uv.x * m_uscale * (m_roi.w - 1)) % m_roi.w + m_roi.x;
        unsigned int y = (unsigned int)round(bRec.its.uv.y * m_vscale * (m_roi.h - 1)) % m_roi.h + m_roi.y;
        
        DelaunayTriangulation::Point p_v(0.5 * (double)bRec.wi[0] / ((double)bRec.wi[2] + 1) + 0.5,
                                         0.5 * (double)bRec.wi[1] / ((double)bRec.wi[2] + 1) + 0.5);
        DelaunayTriangulation::Point p_l(0.5 * (double)bRec.wo[0] / ((double)bRec.wo[2] + 1) + 0.5,
                                         0.5 * (double)bRec.wo[1] / ((double)bRec.wo[2] + 1) + 0.5);
        DelaunayTriangulation::Face_handle f_v = m_triangView.locate(p_v);
        if (f_v == NULL || m_triangView.is_infinite(f_v)) {
            Log(EInfo, "wi: %f %f %f", bRec.wi[0], bRec.wi[1], bRec.wi[2]);
            return Spectrum(0.0f);
        }
        DelaunayTriangulation::Face_handle f_l = m_triangLight.locate(p_l);
        if (f_l == NULL || m_triangLight.is_infinite(f_l)) {
            Log(EInfo, "wo: %f %f %f", bRec.wo[0], bRec.wo[1], bRec.wo[2]);
            return Spectrum(0.0f);
        }
        
        std::vector<K::FT> bc_l;
        std::vector<int> I_l(3);
        CGAL::Barycentric_coordinates::Triangle_coordinates_2<K>(f_l->vertex(0)->point(),
                f_l->vertex(1)->point(),
                f_l->vertex(2)->point())(p_l,
                                         std::back_inserter(bc_l));
        for (unsigned int i = 0; i < 3; ++i) {
            I_l[i] = f_l->vertex(i)->info();
        }
        
        std::vector<K::FT> bc_v;
        std::vector<int> I_v(3);
        CGAL::Barycentric_coordinates::Triangle_coordinates_2<K>(f_v->vertex(0)->point(),
                f_v->vertex(1)->point(),
                f_v->vertex(2)->point())(p_v,
                                         std::back_inserter(bc_v));
                                         
        double sum = 1;
        for (int i = 0; i < 3; ++i) {
            I_v[i] = f_v->vertex(i)->info();
            if (I_v[i] >= (int) m_numViews) {
                sum -= bc_v[i];
                bc_v[i] = 0;
            }
        }
        for (int i = 0; i < 3; ++i) {
            bc_v[i] /= sum;
        }
        
        Spectrum val(0.0f);
        for (unsigned int v = 0; v < 3; ++v) {
            for (unsigned int l = 0; l < 3; ++l) {
                for (unsigned int ch = 0; ch < m_numChans; ++ch) {
                    val[ch] += bc_l[l] * bc_v[v] *
                               m_tensor[y + m_height * (x + m_width * (I_v[v] + (m_numViews + m_numBottomRing) * (I_l[l] + (m_numLights + m_numBottomRing) * ch)))];
                }
            }
        }
        for (unsigned int ch = 0; ch < m_numChans; ++ch) {
            if (val[ch] < 0) {
                val[ch] = 0;
            }
        }
        return val;
    }
    
    Float
    pdf(const BSDFSamplingRecord& bRec, EMeasure measure) const {
        if (!(bRec.typeMask & EDiffuseReflection) ||
                measure != ESolidAngle ||
                Frame::cosTheta(bRec.wi) <= 0 ||
                Frame::cosTheta(bRec.wo) <= 0) {
            return 0.0f;
        }
        
        return warp::squareToCosineHemispherePdf(bRec.wo);
    }
    
    Spectrum
    sample(BSDFSamplingRecord& bRec, const Point2& sample) const {
        if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0) {
            return Spectrum(0.0f);
        }
        
        bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EDiffuseReflection;
        
        return eval(bRec, ESolidAngle) * M_PI / Frame::cosTheta(bRec.wo);
    }
    
    Spectrum
    sample(BSDFSamplingRecord& bRec, Float& pdf, const Point2& sample) const {
        if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0) {
            return Spectrum(0.0f);
        }
        
        bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EDiffuseReflection;
        pdf = warp::squareToCosineHemispherePdf(bRec.wo);
        
        return eval(bRec, ESolidAngle) * M_PI / Frame::cosTheta(bRec.wo);
    }
    
    void
    serialize(Stream* stream, InstanceManager* manager) const {
        BSDF::serialize(stream, manager);
    }
    
    Float
    getRoughness(const Intersection& its, int component) const {
        return std::numeric_limits<Float>::infinity();
    }
    
    std::string
    toString() const {
        std::ostringstream oss;
        oss << "BTFTensor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  path = \"" << m_filename << "\"," << endl
            << "  numLights = \"" << m_numLights << "\"," << endl
            << "  numViews = \"" << m_numViews << "\"," << endl
            << "  channels = \"" << m_numChans << "\"," << endl
            << "  width = \"" << m_width << "\"," << endl
            << "  height = \"" << m_height << "\"," << endl
            << "  uscale = \"" << m_uscale << "\"," << endl
            << "  vscale = \"" << m_vscale << "\"," << endl
            << "]";
        return oss.str();
    }
    
    MTS_DECLARE_CLASS()
    
private:
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb;
    typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
    typedef CGAL::Delaunay_triangulation_2<K, Tds> DelaunayTriangulation;
    
    struct {
        int x, y;
        int w, h;
    } m_roi;
    
    static const unsigned int m_numBottomRing = 32;
    boost::filesystem::path m_filename;
    std::vector<float> m_tensor;
    DelaunayTriangulation m_triangLight;
    DelaunayTriangulation m_triangView;
    unsigned int m_numChans, m_numLights, m_numViews, m_width, m_height;
    double m_uscale, m_vscale;
};

MTS_IMPLEMENT_CLASS_S(BTFTensor, false, BSDF)
MTS_EXPORT_PLUGIN(BTFTensor, "BTF tensor")
MTS_NAMESPACE_END

