#include "ncbind.hpp"
#include "lanczosCPU.hpp"
#include "lanczosGPU.hpp"

#define RETURN_IF_FAILED(STATEMENT) \
        { \
        tjs_error error = (STATEMENT); \
        if (TJS_FAILED(error)) { return error; } \
        }

#define TryGetProperty(OBJ, VALUE_NAME, PROP_NAME) \
        tTJSVariant VALUE_NAME; \
        RETURN_IF_FAILED(OBJ->PropGet(0, TJS_W(PROP_NAME), nullptr, &VALUE_NAME, OBJ));

namespace
{
    tjs_error checkDestCondition(iTJSDispatch2 *dest)
    {
        if (dest == nullptr || dest->IsInstanceOf(0, nullptr, nullptr, TJS_W("Layer"), dest) == TJS_S_FALSE) {
            ttstr mes = TJSGetMessageMapMessage(TJS_W("TJSNativeClassCrash"));
            TVPThrowExceptionMessage(mes.c_str());
        }

        TryGetProperty(dest, destHoldAlpha, "holdAlpha");
        if (destHoldAlpha.AsInteger() == 1) {
            TVPThrowExceptionMessage(TJS_W("holdAlpha must be false"));
        }

        TryGetProperty(dest, destHasImage, "hasImage");
        if (destHasImage.AsInteger() == 0) {
            TVPThrowExceptionMessage(TJS_W("hasImage must be true"));
        }
        return TJS_S_OK;
    }
    
    tjs_error checkSrcCondition(iTJSDispatch2 *src)
    {
        if (src == nullptr || src->IsInstanceOf(0, 0, 0, TJS_W("Layer"), src) == TJS_S_FALSE) {
            TVPThrowExceptionMessage(TJS_W("src must be an object of Layer class"));
        }

        TryGetProperty(src, srcHasImage, "hasImage");

        if (srcHasImage.AsInteger() == 0) {
            TVPThrowExceptionMessage(TJS_W("hasImage must be true"));
        }

        return TJS_S_OK;
    }
    
    tjs_error checkAreaCondition(
        const unsigned int dleft, const unsigned int dtop, const unsigned int dwidth, const unsigned int dheight,
        const unsigned int sleft, const unsigned int stop, const unsigned int swidth, const unsigned int sheight)
    {
        if (dwidth <= 0 || dheight <= 0 || dwidth <= 0 || dheight <= 0) {
            TVPThrowExceptionMessage(TJS_W("width and height must be larger than 0"));
        }

        if (dwidth == swidth && dheight == sheight) {
            TVPThrowExceptionMessage(TJS_W("dest size equals src size"));
        }
        return TJS_S_OK;
    }

    template<int W>
    tjs_error stretchCopyByLanczos(tTJSVariant *result, tjs_int numparams, tTJSVariant **param, iTJSDispatch2 *dest)
    {
        if (numparams < 9) {
            return TJS_E_BADPARAMCOUNT;
        }

        RETURN_IF_FAILED(checkDestCondition(dest));

        const unsigned int dleft = static_cast<unsigned int>(param[0]->AsInteger());
        const unsigned int dtop = static_cast<unsigned int>(param[1]->AsInteger());
        const unsigned int dwidth = static_cast<unsigned int>(param[2]->AsInteger());
        const unsigned int dheight = static_cast<unsigned int>(param[3]->AsInteger());
        iTJSDispatch2 *src = param[4]->AsObjectNoAddRef();
        const unsigned int sleft = static_cast<unsigned int>(param[5]->AsInteger());
        const unsigned int stop = static_cast<unsigned int>(param[6]->AsInteger());
        const unsigned int swidth = static_cast<unsigned int>(param[7]->AsInteger());
        const unsigned int sheight = static_cast<unsigned int>(param[8]->AsInteger());
        const bool useGPU = (numparams > 9 && param[9]->AsInteger() != 0);

        RETURN_IF_FAILED(checkSrcCondition(src));
        RETURN_IF_FAILED(checkAreaCondition(dleft, dtop, dwidth, dheight, sleft, stop, swidth, sheight));

        TryGetProperty(dest, destbuffer, "mainImageBufferForWrite");
        TryGetProperty(dest, destpitch, "mainImageBufferPitch");
        TryGetProperty(src, srcbuffer, "mainImageBufferForWrite");
        TryGetProperty(src, srcpitch, "mainImageBufferPitch");

        if (useGPU) {
            GPU::TVPLanczos<W>(
                static_cast<int>(destpitch.AsInteger()), reinterpret_cast<tjs_uint32*>(destbuffer.AsInteger()),
                dleft, dtop, dwidth, dheight,
                static_cast<int>(srcpitch.AsInteger()), reinterpret_cast<tjs_uint32*>(srcbuffer.AsInteger()),
                sleft, stop, swidth, sheight);
        }
        else {
            CPU::TVPLanczos<W>(
                static_cast<int>(destpitch.AsInteger()), reinterpret_cast<tjs_uint8*>(destbuffer.AsInteger()), tTVPRect(dleft, dtop, dleft + dwidth, dtop + dheight),
                static_cast<int>(srcpitch.AsInteger()), reinterpret_cast<tjs_uint8*>(srcbuffer.AsInteger()), tTVPRect(sleft, stop, sleft + swidth, stop + sheight));
        }
        return TJS_S_OK;
    }
}

NCB_ATTACH_FUNCTION(stretchCopyByLanczos2, Layer, stretchCopyByLanczos<2>);
NCB_ATTACH_FUNCTION(stretchCopyByLanczos3, Layer, stretchCopyByLanczos<3>);
