"""
Particle Deposition onto Octs




"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
cimport cython
from libc.math cimport sqrt

from fp_utils cimport *
from .oct_container cimport Oct, OctAllocationContainer, OctreeContainer

cdef extern from "platform_dep.h":
    void *alloca(int)
    
cdef inline int gind(int i, int j, int k, int dims[3]):
    # The ordering is such that we want i to vary the slowest in this instance,
    # even though in other instances it varies the fastest.  To see this in
    # action, try looking at the results of an n_ref=256 particle CIC plot,
    # which shows it the most clearly.
    return ((i*dims[1])+j)*dims[2]+k


####################################################
# Standard SPH kernel for use with the Grid method #
####################################################

cdef inline np.float64_t sph_kernel_cubic(np.float64_t x) nogil:
    cdef np.float64_t kernel
    if x <= 0.5:
        kernel = 1.-6.*x*x*(1.-x)
    elif x>0.5 and x<=1.0:
        kernel = 2.*(1.-x)*(1.-x)*(1.-x)
    else:
        kernel = 0.
    return kernel

########################################################
# Alternative SPH kernels for use with the Grid method #
########################################################

# quartic spline
cdef np.float64_t Cquartic = 5.**6/512/np.pi
cdef inline np.float64_t sph_kernel_quartic(np.float64_t x) nogil:
    cdef np.float64_t kernel
    if x < 1:
        kernel = (1.-x)**4
        if x < 3./5:
            kernel -= 5*(3./5-x)**4
            if x < 1./5:
                kernel += 10*(1./5-x)**4
    else:
        kernel = 0.
    return kernel * Cquartic

# quintic spline
cdef np.float64_t Cquintic = 3.**7/40/np.pi
cdef inline np.float64_t sph_kernel_quintic(np.float64_t x) nogil:
    cdef np.float64_t kernel
    if x < 1:
        kernel = (1.-x)**5
        if x < 2./3:
            kernel -= 6*(2./3-x)**5
            if x < 1./3:
                kernel += 15*(1./3-x)**5
    else:
        kernel = 0.
    return kernel * Cquintic

ctypedef np.float64_t (*kernel_func) (np.float64_t)
cdef class ParticleDepositOperation:
    # We assume each will allocate and define their own temporary storage
    cdef kernel_func sph_kernel
    cdef public object nvals
    cdef public int update_values
    cdef void process(self, int dim[3], np.float64_t left_edge[3],
                      np.float64_t dds[3], np.int64_t offset,
                      np.float64_t ppos[3], np.float64_t *fields,
                      np.int64_t domain_ind)
