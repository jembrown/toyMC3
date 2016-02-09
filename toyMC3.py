#! /usr/bin/env python

"""
Toy Metropolis-coupling example
@author: jembrown

Starting with a surface consisting of two uniform densities separated by a deep valley. The
width of the valley is bigger than the proposal window, but its density is > 0.
"""

# Import dependencies
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import math
import random

# Define the target distribution ("toy posterior")
# Needs to integrate to 1
peakOneBounds = (0,0.4)
peakOneHeight = 0.5
valleyHeight = 0.000001
peakTwoBounds = (0.9,1.0)
peakTwoHeight = (8.0-(valleyHeight*(peakTwoBounds[0]-peakOneBounds[1])))

			
# Define prior on parameters of interest
def posterior(theta):
	"""
	A function to return the 'posterior' density for a given value of theta.
	"""
	if theta >= peakOneBounds[0] and theta <= peakOneBounds[1]:		# Peak One
		return peakOneHeight
	elif theta >= peakTwoBounds[0] and theta <= peakTwoBounds[1]:	# Peak Two
		return peakTwoHeight
	elif theta >= peakOneBounds[1] and theta <= peakTwoBounds[0]:	# Valley
		return valleyHeight
	else:															# Outside range
		return 0.0

	
# Define proposal distribution
def drawTheta(thetaCurr):
	"""
	This function provides proposed values for p, given a current value.
	"""
	propWindowSize = 0.1
	newTheta = uniform.rvs(loc=thetaCurr-(propWindowSize/2.0),scale=propWindowSize) # min = loc, max = loc+scale
	return newTheta
	
def chainSwap(chains):
	"""
	This function attempts a Metropolis-Hastings swap between the positions of two chains.
	"""
	swapChainOne = random.choice(chains)
	swapChainTwo = random.choice(chains)
	while (swapChainOne == swapChainTwo):	# Making sure same chain not selected twice
		swapChainTwo = random.choice(chains)
	chainR = math.pow( (swapChainTwo.thetaPost/swapChainOne.thetaPost) , ((1/swapChainOne.temp)-(1/swapChainTwo.temp)) )
	chainRanUnifDraw = uniform.rvs()
	if (chainRanUnifDraw <= chainR):
		thetaOne = swapChainOne.theta
		thetaTwo = swapChainTwo.theta
		swapChainOne.theta = thetaTwo
		swapChainTwo.theta = thetaOne
		swapChainOne.thetaProp = drawTheta(swapChainOne.theta)
		swapChainTwo.thetaProp = drawTheta(swapChainTwo.theta)
		swapChainOne.swapCount = swapChainOne.swapCount + 1
		swapChainTwo.swapCount = swapChainTwo.swapCount + 1

	
class chain(object):
	"""
	A class to define instances of Metropolis-coupled Markov chains.
	"""	
	
	def __init__(self,lam,num,name,theta):
		self.name = name
		self.theta = theta
		self.thetaPost = posterior(self.theta)
		self.thetaProp = drawTheta(self.theta)
		self.lam = lam   
		self.num = num
		self.temp = 1.0 + (lam*(self.num-1))
		self.samples = []
		self.swapCount = 0
		self.reportState(debug=False)
	
	def reportState(self,debug=False):
		print ""
		print "Chain Name: %s" % self.name
	 	print "Chain Number: %s" % self.num
		print "Chain Lambda: %s" % self.lam
		print "Chain Temperature: %s" % self.temp
		print "Chain Theta: %f" % self.theta
		print "Chain Probability Density: %f" % self.thetaPost
		print "Chain Swap Count: %d" % self.swapCount
		if (debug):
			print "Chain %s Temp Test: %f" % (self.name,math.pow(0.9,(1.0/self.temp)))
		print ""
	
	def update(self):
		self.thetaPropPost = posterior(self.thetaProp)
		self.thetaPost = posterior(self.theta)
		if (self.thetaPropPost >= self.thetaPost): # If proposed value has posterior density > curr value
			self.theta = self.thetaProp
		elif (self.thetaPropPost < self.thetaPost): # If proposed value has posterior density < curr value
			self.r = math.pow((self.thetaPropPost/self.thetaPost),(1.0/self.temp))
			self.ranUnifDraw = uniform.rvs()
			if (self.ranUnifDraw <= self.r):
				self.theta = self.thetaProp
		else:
			print "Problem calculating proposal ratio for chain %s!" % self.name
			print (self.thetaProp,self.thetaPropPost)
			print (self.theta,self.thetaPost)
			print ""
		self.thetaProp = drawTheta(self.theta) # Finish by drawing new proposed value of theta
		self.thetaPropPost = posterior(self.thetaProp)
		self.thetaPost = posterior(self.theta)
		self.samples.append(self.theta)

# Set chain length and run it
ngens = 100000				# Total length of chain
sampleFreq = 100			# Change this to something >1 if you want to space out samples.
updateFreq = ngens*0.1		# Frequency of screen updates to make sure chain is running.
samples = []				# Vector to hold sampled values from cold chain

# Initiate chains
print ""
print "Starting chain information: "
chainOne = chain(lam=0.3,num=1,name="One",theta=uniform.rvs(loc=0,scale=1))	# Cold chain
chainTwo = chain(lam=0.3,num=2,name="Two",theta=uniform.rvs(loc=0,scale=1))	# First "heated" chain 
chainThree = chain(lam=0.3,num=3,name="Three",theta=uniform.rvs(loc=0,scale=1))
chainFour = chain(lam=0.3,num=4,name="Four",theta=uniform.rvs(loc=0,scale=1))
chainFive = chain(lam=0.3,num=5,name="Five",theta=uniform.rvs(loc=0,scale=1))
chainSix = chain(lam=0.3,num=6,name="Six",theta=uniform.rvs(loc=0,scale=1))
chains = [chainOne,chainTwo,chainThree,chainFour,chainFive,chainSix]	# Create list holding all chains

# Run chains
for gen in range(ngens):
	if (uniform.rvs() <= 0.5):
		for chain in chains:
			chain.update()					# Running Metropolis-Hastings sampling within each chain
	else:
		chainSwap(chains)
	if (gen % sampleFreq == 0):				# Appending latest sample from cold chain
		samples.append(chainOne.theta)
	if (gen % updateFreq == 0):				# Reporting update to screen
		print "Generation %s" % gen
		
for chain in chains:
	chain.reportState()

# Summarizing MCMC samples

# Numerical summaries
burnin = int((ngens/sampleFreq)*0.1)
postBurnSamples = samples[burnin+1:]
print("Posterior Mean: %f" % np.mean(postBurnSamples))
postBurnSamples.sort()  # post-burnin samples will be sorted after this is called
print("Posterior 95% credibility interval: "+"(%f,%f)" % (postBurnSamples[int(len(postBurnSamples)*0.025)],postBurnSamples[int(len(postBurnSamples)*0.975)]))
print("Estimated probability of peak one: %f" % (float(sum(1 for samp in postBurnSamples if samp <= peakOneBounds[1]))/float(len(postBurnSamples))))
print("Estimated probability of peak two: %f" % (float(sum(1 for samp in postBurnSamples if samp >= peakTwoBounds[0]))/float(len(postBurnSamples))))

# Marginal histogram
pl.figure()
pl.xlim(0,1)
pl.hist(postBurnSamples)
pl.show()

# Trace plot
plt.figure()
plt.plot(range(ngens/sampleFreq),samples)
plt.ylim(0,1)
plt.ylabel("Theta")
plt.xlabel("Sample Number")
plt.show()	  
