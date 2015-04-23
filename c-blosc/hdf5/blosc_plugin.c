/*
 * Dynamically loaded filter plugin for HDF5 blosc filter.
 *
 * Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
 * Created: 2014
 *
 */


#include <stdint.h>


#define H5Z_class_t_vers 2

#include "blosc_plugin.h"
#include "blosc_filter.h"


// Prototypes for filter function in blosc_filter.c.
size_t blosc_filter(unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t *buf_size, void **buf);

herr_t blosc_set_local(hid_t dcpl, hid_t type, hid_t space);


H5Z_class_t blosc_H5Filter[1] = {{
    H5Z_CLASS_T_VERS,
    (H5Z_filter_t)(FILTER_BLOSC),
    1, 1,
    "blosc",
    NULL,
    (H5Z_set_local_func_t)(blosc_set_local),
    (H5Z_func_t)(blosc_filter)
}};


H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_FILTER;}


const void* H5PLget_plugin_info(void) {return blosc_H5Filter;}

