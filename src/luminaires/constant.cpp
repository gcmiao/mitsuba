#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

/**
 * Constant background light source
 */
class ConstantLuminaire : public Luminaire {
public:
	ConstantLuminaire(const Properties &props) : Luminaire(props) {
		m_intensity = props.getSpectrum("intensity", 1.0f);
		m_type = EDiffuseDirection | EOnSurface;
	}

	ConstantLuminaire(Stream *stream, InstanceManager *manager) 
		: Luminaire(stream, manager) {
		m_intensity = Spectrum(stream);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		Luminaire::serialize(stream, manager);

		m_intensity.serialize(stream);
	}

	void preprocess(const Scene *scene) {
		/* Get the scene's bounding sphere and slightly enlarge it */
		m_bsphere = scene->getBSphere();
		m_bsphere.radius *= 1.01f;
		m_surfaceArea = m_bsphere.radius * m_bsphere.radius * M_PI;
	}

	Spectrum getPower() const {
		return m_intensity * m_surfaceArea * M_PI;
	}

	Spectrum Le(const Ray &ray) const {
		return m_intensity;
	}

	Spectrum Le(const LuminaireSamplingRecord &lRec) const {
		return m_intensity;
	}

	inline void sample(const Point &p, LuminaireSamplingRecord &lRec,
		const Point2 &sample) const {
		lRec.d = squareToSphere(sample);
		lRec.sRec.p = p - lRec.d * (2 * m_bsphere.radius);
		lRec.pdf = 1.0f / (4*M_PI);
		lRec.Le = m_intensity;
	}

	inline Float pdf(const Point &p, const LuminaireSamplingRecord &lRec) const {
		return 1.0f / (4*M_PI);
	}

	/* Sampling routine for surfaces - just do BSDF sampling */
	void sample(const Intersection &its, LuminaireSamplingRecord &lRec,
		const Point2 &sample) const {
		const BSDF *bsdf = its.shape->getBSDF();
		BSDFQueryRecord bRec(its, sample);
		Spectrum val = bsdf->sample(bRec);
		if (!val.isBlack()) {
			lRec.pdf = bsdf->pdf(bRec);
			lRec.Le = m_intensity;
			lRec.d = -its.toWorld(bRec.wo);
			lRec.sRec.p = its.p - lRec.d * (2 * m_bsphere.radius);
		} else {
			lRec.pdf = 0;
		}
	}

	inline Float pdf(const Intersection &its, const LuminaireSamplingRecord &lRec) const {
		const BSDF *bsdf = its.shape->getBSDF();
		BSDFQueryRecord bRec(its, its.toLocal(-lRec.d));
		return bsdf->pdf(bRec);
	}

	/**
	 * This is the tricky bit - we want to sample a ray that
	 * has uniform density over the set of all rays passing
	 * through the scene.
	 * For more detail, see "Using low-discrepancy sequences and 
	 * the Crofton formula to compute surface areas of geometric models"
	 * by Li, X. and Wang, W. and Martin, R.R. and Bowyer, A. 
	 * (Computer-Aided Design vol 35, #9, pp. 771--782)
	 */
	void sampleEmission(EmissionRecord &eRec, 
		const Point2 &sample1, const Point2 &sample2) const {
		/* Chord model - generate the ray passing through two uniformly
		   distributed points on a sphere containing the scene */
		Vector d = squareToSphere(sample1);
		eRec.sRec.p = m_bsphere.center + d * m_bsphere.radius;
		eRec.sRec.n = Normal(-d);
		Point p2 = m_bsphere.center + squareToSphere(sample2) * m_bsphere.radius;
		eRec.d = p2 - eRec.sRec.p;
		Float length = eRec.d.length();

		if (length == 0) {
			eRec.P = Spectrum(0.0f);
			eRec.pdfArea = eRec.pdfDir = 1.0f;
			return;
		}

		eRec.d /= length;
		eRec.pdfArea = 1.0f / (4 * M_PI * m_bsphere.radius * m_bsphere.radius);
		eRec.pdfDir = INV_PI * dot(eRec.sRec.n, eRec.d);
		eRec.P = m_intensity;
	}

	void sampleEmissionArea(EmissionRecord &eRec, const Point2 &sample) const {
		Vector d = squareToSphere(sample);
		eRec.sRec.p = m_bsphere.center + d * m_bsphere.radius;
		eRec.sRec.n = Normal(-d);
		eRec.pdfArea = 1.0f / (4 * M_PI * m_bsphere.radius * m_bsphere.radius);
		eRec.P = m_intensity * M_PI;
	}

	Spectrum sampleEmissionDirection(EmissionRecord &eRec, const Point2 &sample) const {
		Point p2 = m_bsphere.center + squareToSphere(sample) * m_bsphere.radius;
		eRec.d = p2 - eRec.sRec.p;
		Float length = eRec.d.length();

		if (length == 0.0f) {
			eRec.pdfDir = 1.0f;
			return Spectrum(0.0f);
		}
		
		eRec.d /= length;
		eRec.pdfDir = INV_PI * dot(eRec.sRec.n, eRec.d);
		return Spectrum(INV_PI);
	}

	void pdfEmission(EmissionRecord &eRec) const {
		Float dp = dot(eRec.sRec.n, eRec.d);
		if (dp > 0)
			eRec.pdfDir = INV_PI * dp;
		else
			eRec.pdfDir = 0;
		eRec.pdfArea = 1.0f / (4 * M_PI * m_bsphere.radius * m_bsphere.radius);
	}

	Spectrum f(const EmissionRecord &eRec) const {
		return Spectrum(INV_PI);
	}
	
	Spectrum fArea(const EmissionRecord &eRec) const {
		return m_intensity * M_PI;
	}

	bool isBackgroundLuminaire() const {
		return true;
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "ConstantLuminaire[" << std::endl
			<< "  intensity = " << m_intensity.toString() << "," << std::endl
			<< "  power = " << getPower().toString() << std::endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	Spectrum m_intensity;
	BSphere m_bsphere;
};

MTS_IMPLEMENT_CLASS_S(ConstantLuminaire, false, Luminaire)
MTS_EXPORT_PLUGIN(ConstantLuminaire, "Constant background luminaire");
MTS_NAMESPACE_END