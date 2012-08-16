# -*- coding: utf-8 -*-
########################################################################
#
#       License: BSD
#       Created: August 16, 2012
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

import os, os.path
import json


ATTRSDIR = "__attrs__"

class attrs(object):
    """Accessor for attributes in carray objects."""

    def __init__(self, rootdir, mode, _new=False):
        self.rootdir = rootdir
        self.mode = mode
        self.attrs = {}
        self.attrsfile = os.path.join(self.rootdir, ATTRSDIR)
        
        if _new:
            self.create()
        else:
            self.open()

    def create(self):
        if self.mode != 'r':
            # Empty the underlying file
            with open(self.attrsfile, 'wb') as rfile:
                rfile.write(json.dumps({}))
                rfile.write("\n")

    def open(self):
        if not os.path.isfile(self.attrsfile):
            if self.mode != 'r':
                # Create a new empty file
                with open(self.attrsfile, 'wb') as rfile:
                    rfile.write("\n")
        # Get the serialized attributes
        with open(self.attrsfile, 'rb') as rfile:
            data = json.loads(rfile.read())
        # JSON returns unicode (?)
        for name, attr in data.items():
            self.attrs[str(name)] = attr

    def update_meta(self):
        """Update attributes on-disk."""
        with open(self.attrsfile, 'wb') as rfile:
            rfile.write(json.dumps(self.attrs))
            rfile.write("\n")

    def getall(self):
        return self.attrs.copy()

    def __getitem__(self, name):
        return self.attrs[name]

    def __setitem__(self, name, carray):
        if self.mode == 'r':
            raise IOError(
                "Cannot modify an attribute in 'r'ead-only mode")
        self.attrs[name] = carray
        self.update_meta()

    def __delitem__(self, name):
        """Remove the `name` attribute."""
        if self.mode == 'r':
            raise IOError(
                "Cannot remove an attribute in 'r'ead-only mode")
        del self.attrs[name]
        self.update_meta()
    
    def __iter__(self):
        return self.attrs.iteritems()

    def __len__(self):
        return len(self.attrs)

    def __str__(self):
        fullrepr = ""
        for name in self.attrs:
            fullrepr += "%s : %s" % (name, self.attrs[name]) 
        return fullrepr

    def __repr__(self):
        fullrepr = ""
        for name in self.attrs:
            fullrepr += "%s : %r\n" % (name, self.attrs[name]) 
        return fullrepr
