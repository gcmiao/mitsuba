#include <mitsuba/render/film.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include "banner.h"

MTS_NAMESPACE_BEGIN

/**
 * Simple film implementation, which stores the captured image
 * as an RGBA-based high dynamic-range EXR file.
 * No gamma correction is applied and spectral radiance values
 * are converted to linear RGB using the CIE 1931 XYZ color matching 
 * functions and ITU-R Rec. BT.709
 */
class EXRFilm : public Film {
protected:
	/* Pixel data structure */
	struct Pixel {
		Spectrum spec;
		Float alpha;
		Float weight;

		Pixel() : spec(0.0f), alpha(0.0f), weight(0.0f) {
		}
	};

	Pixel *m_pixels;
	bool m_hasBanner;
	bool m_hasAlpha;
public:
	EXRFilm(const Properties &props) : Film(props) {
		m_pixels = new Pixel[m_cropSize.x * m_cropSize.y];
		/* Should an alpha channel be added to the output image? */
		m_hasAlpha = props.getBoolean("alpha", true);
		/* Should an Mitsuba banner be added to the output image? */
		m_hasBanner = props.getBoolean("banner", true);
	}

	EXRFilm(Stream *stream, InstanceManager *manager) 
		: Film(stream, manager) {
		m_hasAlpha = stream->readBool();
		m_hasBanner = stream->readBool();
		m_pixels = new Pixel[m_cropSize.x * m_cropSize.y];
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		Film::serialize(stream, manager);
		stream->writeBool(m_hasAlpha);
		stream->writeBool(m_hasBanner);
	}

	virtual ~EXRFilm() {
		if (m_pixels)
			delete[] m_pixels;
	}

	void clear() {
		memset(m_pixels, 0, sizeof(Pixel) * m_cropSize.x * m_cropSize.y);
	}

	void fromBitmap(const Bitmap *bitmap) {
		Assert(bitmap->getWidth() == m_cropSize.x 
			&& bitmap->getHeight() == m_cropSize.y);
		Assert(bitmap->getBitsPerPixel() == 128);
		unsigned int lastIndex = m_cropSize.x * m_cropSize.y;

		for (unsigned int index=0; index<lastIndex; ++index) {
			Pixel &pixel = m_pixels[index];
			const float 
				r = bitmap->getFloatData()[index*4+0],
				g = bitmap->getFloatData()[index*4+1],
				b = bitmap->getFloatData()[index*4+2],
				a = bitmap->getFloatData()[index*4+3];
			pixel.spec.fromLinearRGB(r, g, b);
			pixel.alpha = a;
			pixel.weight = 1.0f;
		}
	}

	void toBitmap(Bitmap *bitmap) const {
		Assert(bitmap->getWidth() == m_cropSize.x 
			&& bitmap->getHeight() == m_cropSize.y);
		Assert(bitmap->getBitsPerPixel() == 128);
		unsigned int lastIndex = m_cropSize.x * m_cropSize.y;
		Float r, g, b, a;

		for (unsigned int index=0; index<lastIndex; ++index) {
			Pixel &pixel = m_pixels[index];
			Float invWeight = pixel.weight > 0 ? 1/pixel.weight : 1;
			pixel.spec.toLinearRGB(r, g, b);
			a = pixel.alpha;
			bitmap->getFloatData()[index*4+0] = r*invWeight;
			bitmap->getFloatData()[index*4+1] = g*invWeight;
			bitmap->getFloatData()[index*4+2] = b*invWeight;
			bitmap->getFloatData()[index*4+3] = a*invWeight;
		}
	}

	Spectrum getValue(int xPixel, int yPixel) {
		xPixel -= m_cropOffset.x; yPixel -= m_cropOffset.y;
		if (!(xPixel >= 0 && xPixel < m_cropSize.x && yPixel >= 0 && yPixel < m_cropSize.y )) {
			Log(EWarn, "Pixel out of range : %i,%i", xPixel, yPixel); 
			return Spectrum(0.0f);
		}
		Pixel &pixel = m_pixels[xPixel + yPixel * m_cropSize.x];
		return pixel.weight != 0 ? pixel.spec / pixel.weight : Spectrum(0.0f);
	}

	void putImageBlock(const ImageBlock *block) {
		int entry=0, imageY = block->getOffset().y - 
			block->getBorder() - m_cropOffset.y - 1;
		
		for (int y=0; y<block->getFullSize().y; ++y) {
			if (++imageY < 0 || imageY >= m_cropSize.y) {
				/// Skip a row if it is outside of the crop region
				entry += block->getFullSize().x;
				continue;
			}

			int imageX = block->getOffset().x - block->getBorder()
				- m_cropOffset.x - 1;
			for (int x=0; x<block->getFullSize().x; ++x) {
				if (++imageX < 0 || imageX >= m_cropSize.x) {
					++entry;
					continue;
				}

				Pixel &pixel = m_pixels[imageY * m_cropSize.x + imageX];

				pixel.spec += block->getPixel(entry);
				pixel.alpha += block->getAlpha(entry);
				pixel.weight += block->getWeight(entry++);
			}
		}
	}
	
	void develop(const std::string &destFile) {
		Log(EDebug, "Developing film ..");
		ref<Bitmap> bitmap = new Bitmap(m_cropSize.x, m_cropSize.y, 128);
		float *targetPixels = bitmap->getFloatData();
		Float r, g, b;
		int pos = 0;

		for (int y=0; y<m_cropSize.y; y++) {
			for (int x=0; x<m_cropSize.x; x++) {
				/* Convert spectrum to XYZ colors */
				Pixel &pixel = m_pixels[pos];
				Float invWeight = 1.0f;
				if (pixel.weight != 0.0f)
					invWeight = 1.0f / pixel.weight;
				(pixel.spec * invWeight).toLinearRGB(r, g, b);

				targetPixels[4*pos+0] = std::max(0.0f, (float) r);
				targetPixels[4*pos+1] = std::max(0.0f, (float) g);
				targetPixels[4*pos+2] = std::max(0.0f, (float) b);
				targetPixels[4*pos+3] = m_hasAlpha ? (pixel.alpha*invWeight) : 1.0f;
				++pos;
			}
		}

		if (m_hasBanner && m_cropSize.x > bannerWidth+5 && m_cropSize.y > bannerHeight + 5) {
			int xoffs = m_cropSize.x - bannerWidth - 5, yoffs = m_cropSize.y - bannerHeight - 5;
			for (int y=0; y<bannerHeight; y++) {
				for (int x=0; x<bannerWidth; x++) {
					float value = (1-banner[x+y*bannerWidth]) * 1e5f;
					int pos = 4*((x+xoffs)+(y+yoffs)*m_cropSize.x);
					targetPixels[pos+0] += value;
					targetPixels[pos+1] += value;
					targetPixels[pos+2] += value;
					targetPixels[pos+3] = 1.0f;
				}
			}
		}

		std::string filename = destFile;
		if (!endsWith(filename, ".exr"))
			filename += ".exr";
		Log(EInfo, "Writing image to \"%s\" ..", filename.c_str());
		ref<FileStream> stream = new FileStream(filename, FileStream::ETruncWrite);
		bitmap->save(Bitmap::EEXR, stream);
	}
	
	bool destinationExists(const std::string &baseName) const {
		std::string filename = baseName;
		if (!endsWith(filename, ".exr"))
			filename += ".exr";
		return FileStream::exists(filename);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "EXRFilm[" << std::endl
			<< "  size = " << m_size.toString() << "," << std::endl
			<< "  cropOffset = " << m_cropOffset.toString() << "," << std::endl
			<< "  cropSize = " << m_cropSize.toString() << "," << std::endl
			<< "  alpha = " << m_hasAlpha << "," << std::endl
			<< "  banner = " << m_hasBanner << std::endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(EXRFilm, false, Film)
MTS_EXPORT_PLUGIN(EXRFilm, "High dynamic-range film (EXR)");
MTS_NAMESPACE_END