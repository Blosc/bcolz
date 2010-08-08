/**
  Get version info.

  `versions` must be a char* versions[2] pointer.  In position 0, it
  is returned the BLOSC_VERSION_STRING and in position 1 the
  BLOSC_VERSION_DATE macros.

*/
void blosc_get_versions(char **versions);
