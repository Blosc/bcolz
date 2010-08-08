#include <stdlib.h>
#include "blosc.h"

/* Get version info */
void blosc_get_versions(char **versions)
{
  versions[0] = BLOSC_VERSION_STRING;
  versions[1] = BLOSC_VERSION_DATE;
}
