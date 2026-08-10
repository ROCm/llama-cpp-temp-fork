#include "primitive-capabilities.h"

namespace ggml::hrx {

bool primitive_capability_declared(enum ggml_op operation) {
    switch (operation) {
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_ARGSORT:
        case GGML_OP_CLAMP:
        case GGML_OP_DIV:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
}

}  // namespace ggml::hrx
