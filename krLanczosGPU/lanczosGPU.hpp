#pragma once

namespace GPU
{
    using tjs_uint32 = unsigned __int32;

    template<int W>
    void TVPLanczos(
        int destpitch, tjs_uint32 * const destbuf,
        const unsigned int destleft, const unsigned int desttop, const unsigned int destwidth, const unsigned int destheight,
        int srcpitch, const tjs_uint32 * const srcbuf,
        const unsigned int srcleft, const unsigned int srctop, const unsigned int srcwidth, const unsigned int srcheight);
}
