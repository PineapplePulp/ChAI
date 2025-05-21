#include "stdchpl.h"
#include "wctype.h"
#include "ctype.h"
#include "ImageHelper/stb_image_helper.h"
void chpl__init_Bridge(int64_t _ln,
                       int32_t _fn);
void chpl__init_NDArray(int64_t _ln,
                        int32_t _fn);
void chpl__init_Standard(int64_t _ln,
                         int32_t _fn);
void chpl__init_Types(int64_t _ln,
                      int32_t _fn);
void chpl__init_Utilities(int64_t _ln,
                          int32_t _fn);
void chpl__init_ndarrayRandom(int64_t _ln,
                              int32_t _fn);
void chpl__init_smol(int64_t _ln,
                     int32_t _fn);
int64_t square(int64_t x);
void printArray(chpl_external_array * a);
void globalLoadModel(void);
chpl_external_array getNewFrame(chpl_external_array * frame,
                                int64_t height,
                                int64_t width,
                                int64_t channels);
