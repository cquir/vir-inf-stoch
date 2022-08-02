#!/usr/bin/env python3
# Copyright (C) 2022 Christian Quirouette <cquir@ryerson.ca>
#                    Catherine Beauchemin <cbeau@users.sourceforge.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

from scipy.integrate import solve_ivp
import numpy

def stoch_solver(pdic,V,T,meanfield,pcrit=0.05,tprint=0.05):
	# tprint is time interval (of time in sim) at which to record data
	# 0.001 means every 3.6 seconds

	# Unpack variables and pre-compute some stuff
	pV = pdic['p']
	b = pdic['beta']
	g = pdic['gamma']
	kE = 1.0*pdic['nE']/pdic['tauE']
	dI = 1.0*pdic['nI']/pdic['tauI']

	# Some pre-calcs re rapid dt cal
	# pcrit/max(a,b*T) = pcrit/b/max(a/b,c) = pcob/max(aob,c)
	pcob = 1.0*pcrit/b
	aob = 1.0*max(pdic['c'],kE,dI)/b

	# ensures the very first step is stored
	tnext = -1.0; 
	t = 0.0
	res = []

	if meanfield: #continuous populations (deterministic)
		E = numpy.zeros(pdic['nE'])
		I = numpy.zeros(pdic['nI'])
		rbin = lambda n,p: n*p
		rpoi = lambda lam: lam
		rmul = lambda n,pvals: [n*pvals[0],n*pvals[1],n*(1-pvals[0]-pvals[1])]
	else:  # discrete population (stochastic)
		E = numpy.zeros(pdic['nE'],dtype=int)
		I = numpy.zeros(pdic['nI'],dtype=int)
		rbin = lambda n,p: numpy.random.binomial(n,p)
		rpoi = lambda lam: numpy.random.poisson(lam)
		rmul = lambda n,pvals: numpy.random.multinomial(n,pvals)

	# end simulation when they are no more infectious virions or cells
	Vnabrt  = 0.0 if meanfield else 0
	threshold = 0.01 if meanfield else 0
	while (Vnabrt+V+numpy.sum(E)+numpy.sum(I)) > threshold:

		# store only every "teval" points, and store 1st/last point
		if t >= tnext:
			res.append( (t , V, T, numpy.sum(E), numpy.sum(I)) )
			tnext = t + tprint

		# Determine step size to take based on sensitivity pcrit
		dt = 1.0*pcob/max(aob,T)

		# Sample how many events occur in time step 
		# Last item in pvals makes sure pvals adds to 1 so value irrelevant
		[Vclr,Vabs,Vleft] = rmul(n=V,pvals=[dt*pdic['c'],dt*b*T,0])
		Vnabrt = rbin(n=Vabs,p=g) 
		if (T<1) and meanfield:
			Ninf = 0 ; T = 0
		elif (T>=1) and meanfield:
			Ninf = T*(1.0-((T-1.0)/T)**Vnabrt)
		else:
			Ninf = len(numpy.unique(numpy.random.randint(T,size=Vnabrt)))
		EiOut = rbin(n=E,p=dt*kE)
		IjOut = rbin(n=I,p=dt*dI)
		Vprod = rpoi(lam=dt*pV*numpy.sum(I))

		#update the system
		t += dt
		V = Vleft + Vprod
		T -= Ninf
		E[0]  += Ninf - EiOut[0]
		E[1:] -= numpy.diff(EiOut)
		I[0]  += EiOut[-1] - IjOut[0]
		I[1:] -= numpy.diff(IjOut)

	# If not already in, store the very last step
	if len(res) == 0:
		pass
	elif res[0][-1] != t:
		res.append( (t , V, T, numpy.sum(E), numpy.sum(I)) )

	return numpy.vstack(res).T

def meanfield_solver(pdic,tf,t_eval):
	def ODE(t,y):
		(T,E,I,V) = (y[0],y[1:1+pdic['nE']],y[-1-pdic['nI']:-1],y[-1])
		# Pre-calculation of shared terms
		sI = numpy.sum(I)
		bTV = pdic['beta']*T*V/pdic['S']
		kE = (1.0*pdic['nE']/pdic['tauE'])*E
		dI = (1.0*pdic['nI']/pdic['tauI'])*I
		# ODEs
		dT = -pdic['gamma']*bTV
		dE1 = pdic['gamma']*bTV-kE[0]
		dEi = -numpy.diff(kE)
		dI1 = kE[-1]-dI[0]
		dIi = -numpy.diff(dI)
		dV = pdic['p']*sI-pdic['c']*V-bTV
		return numpy.hstack((dT,dE1,dEi,dI1,dIi,dV))
	y0 = numpy.zeros(pdic['nE']+pdic['nI']+2)
	y0[0] = pdic['Nx']
	y0[-1] = pdic['V0'] 
	sol = solve_ivp(ODE,[0,tf],y0,method='BDF',t_eval=t_eval)
	t = sol.t
	T = sol.y[0] 
	E = numpy.sum(sol.y[1:1+pdic['nE']],axis=0) 
	I = numpy.sum(sol.y[-1-pdic['nI']:-1],axis=0)
	V = sol.y[-1]
	return t,T,E,I,V
