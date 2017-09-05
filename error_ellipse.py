from __future__ import division
import numpy as np
from numpy.random import uniform
from scipy.interpolate import griddata
import numpy.ma as ma
import math
from scipy.stats import chisquare
import matplotlib.pyplot as plt

#load the data file

data=np.loadtxt("test_data1.dat");
x=data[:,0];                       #getting the variable in x-direction
y=data[:,1];                       #getting the variable in y-direction
z=data[:,2];                       #correlation values

#calculation of the covariance matrix
data_points=np.vstack((x,y));    

M=np.cov(data_points);      #covariance matrix
w,v=np.linalg.eig(M);       #eigenvalues and eigenvectors

eig_max=max(w);
eiv_max=v[:,w.argmax()];
eiv_max_index=w.argmax();

#getting the larger eigenvalue and correcsponding eigenvector 
if eiv_max_index==0:
    eig_min=w[1];
    eiv_min=v[:,1];
else:
    eig_min=w[0];
    eiv_min=v[:,0];

#angle of inclination of the ellipse with x-axis

theta=math.atan2(eiv_max[1],eiv_max[0]);
if theta<0.0:
    theta=theta+2*np.pi;

#centre of the ellipse
X0=np.mean(x);
Y0=np.mean(y);

#s-value of error ellipse (depicting the percentage of total points enclosed)
s_value=9.21;  #for 97.5% confidence intervals

#semi major and semi minor axes of the error ellipse

a=2*np.sqrt(s_value*eig_max);
b=2*np.sqrt(s_value*eig_min);

angle_array=np.linspace(0,2*np.pi,2000); #angle array to draw ellipse

#equation of the ellipse
X00=a*np.cos(theta)*np.cos(angle_array)-b*np.sin(theta)*np.sin(angle_array)+X0;
Y00=a*np.sin(theta)*np.cos(angle_array)+b*np.cos(theta)*np.sin(angle_array)+Y0;

#draw the figure

fig1=plt.figure()
plt.plot(X00,Y00)
plt.plot(x,y,'.')
plt.show()

