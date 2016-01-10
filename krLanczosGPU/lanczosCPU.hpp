#pragma once

#include "ncbind.hpp"

namespace CPU
{
    template<int W>
    void TVPLanczos(
        int destpitch, tjs_uint8 *destbuf, const tTVPRect &destrect,
        int srcpitch, const tjs_uint8 *srcbuf, const tTVPRect &srcrect);
}

