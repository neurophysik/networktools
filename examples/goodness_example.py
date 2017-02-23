#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from networktools import strengthsurrogates, strengthsurrogate_tester
from scipy.io import loadmat

"""
This is an example for testing the goodness of surrogates for a random network of 24 nodes.
For a stepsize of about 1000, the goodness (chi) assymptotically approaches 1, which indicates that this is a good step size to use for generating surrogates – for networks of this kind and these parameters of the surrogate-generating routine.
"""

B = np.random.random((24,24))
sample_network = B + B.T

refstepsize = 100000
treesteps = 3
chainsteps = 100

numsurr = 2**treesteps * chainsteps
extra_for_ref = 18
c = 10

refsurrs = strengthsurrogates(
					sample_network,
					treesteps,
					10*refstepsize,
					chainsteps + extra_for_ref,
					refstepsize,
					np.random.randint(0,1e10))

tester = strengthsurrogate_tester(refsurrs, numsurr, c)

for stepsize in map(int, np.logspace(1,4,30)):
	surrs = strengthsurrogates(
					sample_network,
					treesteps,
					10*stepsize,
					chainsteps,
					stepsize,
					np.random.randint(0,1e10))
	
	print('stepsize = %(stepsize)d, chi = %(chi)f' % {'stepsize': stepsize, 'chi': tester.chi(surrs)})

