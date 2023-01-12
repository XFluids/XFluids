#pragma once

namespace middle
{
    enum sync_t
    {
        device = 0, /**DeviceSynchronize */
        block = 1   /**__syncthreads in a block */
    };
} // end namespace middle