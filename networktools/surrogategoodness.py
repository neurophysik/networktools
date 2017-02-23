#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ._networktools import distances
from math import lgamma, exp, log

lfact = lambda n: lgamma(n+1)

def lptilde(k, a, c, norm=0):
	return (
		  lfact(a)
		- lfact(k)
		- lfact(a-k)
		- lfact(2*a)
		+ lfact(c+k)
		+ lfact(2*a-c-k)
		+ norm
		)

ptilde = lambda k, a, c, norm: exp(lptilde(k, a, c, norm))

class strengthsurrogate_tester(object):
	"""
	Creates a tester for the goodness of strength-preserving surrogates as per Ansmann and Lehnertz, Phys. Rev. E 84, 026103 (2011), chapter C2. The names of parameters are analogous to that section. See the file goodness_example.py for an example of usage.
	
	Parameters
	----------
	QR: list of weighted networks
		These are surrogates generated with a reference method (such as the same method with a very high step size).
	
	a: integer
		The size of the lists of surrogates to be tested.
	
	c: integer
		The number of reference surrogates in each ε-ball used for testing. This should be much smaller than a. A reasonable choice for this parameter is 10.
	"""
	
	def __init__(self, QR, a, c):
		assert c<a
		self.a = a
		self.c = c
		np.random.shuffle(QR)
		Q = QR[:a]
		self.R = R = QR[a:]
		self.b = b = len(R)
		dists = distances(Q,R)
		dists.sort(axis=0)
		
		n = QR.shape[1]
		d = (n*(n-1))/2
		
		# safer and more robust than:
		# epsilons = ((dists[c+1,:]**d + dists[c,:]**d)/2)**(1./d)
		self.epsilons = dists[c,:]*(0.5+0.5*(dists[c+1,:]/dists[c,:])**d)**(1./d)
		if not np.all(np.isfinite(self.epsilons)):
			self.epsilons = dists[c+1,:]

	def chi(self, P):
		"""
		Calculates the quantifier of surrogate goodness χ. If this value is consistently around 1, the tested surrogates are good.
		
		Parameters
		----------
		P: list of a weighted networks
			The surrogates to be tested.
		"""
		
		assert self.a==len(P), "The number of networks in P must be a."
		dists = distances(P,self.R)
		ks = (dists<self.epsilons).sum(axis=0)
		norm = -lptilde(self.c,self.a,self.c)
		return (
			sum(ptilde(k, self.a, self.c, norm) for k in ks)
			* sum(ptilde(j, self.a, self.c, norm) for j in range(self.a+1))
			/ sum(ptilde(j, self.a, self.c, norm)**2 for j in range(self.a+1))
			/ self.b
			)

