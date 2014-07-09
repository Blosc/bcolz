#include <blosc.h>

#if BLOSC_VERSION_MAJOR == 1 && BLOSC_VERSION_MINOR < 4 && BLOSC_VERSION_RELEASE < 1
#error "Blosc version >= 1.4.1 required"
#endif
