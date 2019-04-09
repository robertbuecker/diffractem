#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:43:51 2018

@author: dekockmi
"""

@custom_model
def Hyperbolic(x,y, amplitude=1.0,x_0=0.0,y_0=0.0, scale=1.0, shape=1.0,asym=1.0):
    radius = (((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0 + 1)**0.5
    return amplitude*special.kn(1,shape*radius)/radius/special.kn(1,1)

@custom_model
def Voight(x,y,amplitude=1.0,x_0=0.0,y_0=0.0,gauss_scale=1.0,cauchy_scale=1.0):
    """
    custom_model: Voight line profile, 
    which is a convolution of a Lorentz and Gaussian distributions.
    :param x: x coordinate
    :param y: y coordinate
    :param amplitude: normalisation of distribution as a free parameter
    :param x_0: x location
    :param y_0: y location
    :param gauss_scale: scale parameter for the Gaussian
    :param cauchy_scale: scale parameter for the Cauchy
    """
    radius = ((x-x_0)**2.0 + (y-y_0)**2.0)**0.5 + 1j*cauchy_scale
    return amplitude*np.real(special.wofz(radius/gauss_scale))

@custom_model
def MixtureGaussCauchy(x, y, amplitude=1.0,mix=1.0, x_0=0.0, y_0=0.0, shape=2.0,gauss_scale=20.0,cauchy_scale=20.0):
    """
    Custom model: Two-dimensional Cauchy model with variance-covariance matrix 
    ((xscale, xy_scale),(xy_scale, y_scale))
    :param x: x coordinate
    :param y: y coordinate
    :param amplitude: normalisation of distribution as a free parameter
    :param x_0: x location
    :param y_0: y_location
    :param shape: shape parameter for Cauchy distribution
    :param x_scale: scale of the x-coordinate
    :param y_scale: scale of the y-coordinate
    :param corr: correlation between x and y, must be between -1.0 and 1.0
    """
    radius = (x-x_0)**2.0 + (y-y_0)**2.0
    return amplitude*mix*(1+radius/cauchy_scale**2.0)**(shape/2.0) + amplitude*(1-mix)*np.exp(-radius/gauss_scale**2.0)


@custom_model
def Cauchy2D(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, shape=2.0,x_scale=1.0,y_scale=1.0,corr=0.0):
    """
    Custom model: Two-dimensional Cauchy model with variance-covariance matrix 
    ((xscale, xy_scale),(xy_scale, y_scale))
    :param x: x coordinate
    :param y: y coordinate
    :param amplitude: normalisation of distribution as a free parameter
    :param x_0: x location
    :param y_0: y_location
    :param shape: shape parameter for Cauchy distribution
    :param x_scale: scale of the x-coordinate
    :param y_scale: scale of the y-coordinate
    :param corr: correlation between x and y, must be between -1.0 and 1.0
    """
    determinant = (x_scale**2)*(y_scale**2)*(1-corr**2)
    if (determinant <= 0.0):
        raise InputParameterError("correlations must be between -1 and 1")
    radius = ((x-x_0)*(y_scale))**2.0 + ((y-y_0)*(x_scale))**2.0-2*(x-x_0)*(y-y_0)*corr*x_scale*y_scale
    return amplitude*((1+radius/determinant)**(-shape/2.0))


@custom_model
@jit
def CauchyRadial(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, shape=4.0, scale=20.0):
    """
    Custom model: Radial Cauchy Distribution
    :param x: x coordinate
    :param y: y coordinate
    :param amplitude: normalisation of distribution as a free parameter
    :param x_0: x location
    :param y_0: y_location
    :param shape: shape parameter for Cauchy distribution
    :param scale: radial scale
    """
    return amplitude * (1 + ((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0)**(-shape/2.0)

class CauchyRadial(Component):
    """Two-dimensional Cauchy distribution with arbitrary shape parameter"""
    def __init__(self, amplitude=1., x_0=0.0, y_0=0.0, scale=1.0, shape=2.0):
        Component.__init__(self,('amplitude','x_0','y_0','scale','shape'))
        self.amplitude.value = amplitude
        self.x_0.value=x_0
        self.y_0.value=y_0
        self.scale.value=scale
        self.shape.value=shape
        
        # Boundaries
        self.amplitude.bmin = 0
        self.amplitude.bmax = None
        self.x_0.bmin = None
        self.x_0.bmax = None
        self.y_0.bmin = None
        self.y_0.bmax = None
        self.scale.bmin = 0.1
        self.scale.bmax = None
        self.shape.bmin = 1.0
        self.shape.bmax = None
        
        self.isbackground = False
        self.isconvolved = True
        # Gradients
        self.amplitude.grad = self.grad_amplitude
        self.x_0.grad = self.grad_x0
        self.y_0.grad = self.grad_y0
        self.scale.grad = self.grad_scale
        self.shape.grad = self.grad_shape
        
    def function(self,x, y):
        amp = self.amplitude.value
        x_0 = self.x_0.value
        y_0 = self.y_0.value
        scale=self.scale.value
        shape=self.shape.value
        ### DEBUG ### print(amp,x_0,y_0,scale,shape)
        return amp*((1+((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0)**(-shape/2.0))
        
    def grad_amplitude(self,x,y):
        return self.function(x,y) / self.amplitude.value
        
    def grad_x0(self,x,y):
        amp = self.amplitude.value
        x_0 = self.x_0.value
        y_0 = self.y_0.value
        scale=self.scale.value
        shape=self.shape.value
        return amp*((1+((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0)**(-shape/2.0-1.0))*shape*(x-x_0)/(scale**2.0)
        
    def grad_y0(self,x,y):
        amp = self.amplitude.value
        x_0 = self.x_0.value
        y_0 = self.y_0.value
        scale=self.scale.value
        shape=self.shape.value
        return amp*((1+((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0)**(-shape/2.0-1.0))*shape*(y-y_0)/(scale**2.0)
        
    def grad_scale(self,x,y):
        amp = self.amplitude.value
        x_0 = self.x_0.value
        y_0 = self.y_0.value
        scale=self.scale.value
        shape=self.shape.value
        return amp*((1+((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0)**(-shape/2.0-1.0))*shape*((x-x_0)**2.0+(y-y_0)**2.0)/(scale**3.0)
        
    def grad_shape(self,x,y):
        amp = self.amplitude.value
        x_0 = self.x_0.value
        y_0 = self.y_0.value
        scale=self.scale.value
        shape=self.shape.value
        return -0.5*amp*((1+((x-x_0)/scale)**2.0 + ((y-y_0)/scale)**2.0)**(-shape/2.0))*np.log(1+((x-x_0)/scale)**2.0+((y-y_0)/scale)**2.0 )
