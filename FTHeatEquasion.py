# -*- coding: utf-8 -*-

"""
Created on Sun Jun 14 21:32:08 2020

@author: Pranav Devarinti
"""
# Imports
import numpy as np
import scipy
from scipy import integrate
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
from numba import njit
from sys import exit
import tqdm
# In[]
# Variables to alter:
mn_level = 5 # Fourier Transform Variable

start = -2
stop = 2.0001
step =.05 #.00625

t=1.5
dt=.003

k = .2

fn = 'Sample4'
fps = 30

#Function here:
def InitialHeatEquasion(x,y):
    return np.sin(x+y)
# In[] 
# Don't edit these

x_start=start
x_stop=stop
x_step =step
lambda_x = x_stop-x_start

y_start=start
y_stop=stop
y_step = step
lambda_y = y_stop-y_start

# In[]
# 3D Fourier Approximation Code


class FourierTransformCalculations():
    def __init__(self,initialFunc,x_start,x_stop,y_start,y_stop,mn_level):
        self.initialFunc = initialFunc
        self.x_0 = x_start
        self.l_x = x_stop-x_start
        self.y_0 = y_start
        self.l_y = y_stop-y_start
        self.mn_level = mn_level       
        self.c_0 = 4/(self.l_x*self.l_y)
        self._define_coeff_funcs()
    def _define_coeff_funcs(self):
        self._alpha_func = lambda x,y,n,m:(self.initialFunc(x,y)*(np.cos(2*np.pi*n*x/self.l_x))*(np.cos(2*np.pi*m*y/self.l_y)))
        self._beta_func = lambda x,y,n,m:(self.initialFunc(x,y)*(np.cos(2*np.pi*n*x/self.l_x))*(np.sin(2*np.pi*m*y/self.l_y)))
        self._gamma_func = lambda x,y,n,m:(self.initialFunc(x,y)*(np.sin(2*np.pi*n*x/self.l_x))*(np.cos(2*np.pi*m*y/self.l_y)))
        self._delta_func = lambda x,y,n,m:(self.initialFunc(x,y)*(np.sin(2*np.pi*n*x/self.l_x))*(np.sin(2*np.pi*m*y/self.l_y)))
    def alpha_coeff_calc(self,n,m):
        datapoints = np.array(np.meshgrid(np.arange(0,n),np.arange(0,m))).transpose() 
        place_grid = np.zeros(datapoints.shape[:-1])
        for row in range(0,len(datapoints)):
            for column in range(0,len(datapoints[row])):
                if row == 0 and column==0:
                    c_0 = self.c_0/4
                elif row == 0 or column==0:
                    c_0 = self.c_0/2
                else:
                    c_0 = self.c_0
                place_grid[row,column] = c_0*integrate.dblquad(self._alpha_func,self.x_0,self.x_0+self.l_x,lambda x:self.y_0,lambda x:self.y_0+self.l_y,args=(row,column))[0]
        return place_grid
    def beta_coeff_calc(self,n,m):
        datapoints = np.array(np.meshgrid(np.arange(0,n),np.arange(0,m))).transpose() 
        place_grid = np.zeros(datapoints.shape[:-1])
        for row in range(0,len(datapoints)):
            for column in range(0,len(datapoints[row])):
                if row == 0 and column==0:
                    c_0 = self.c_0/4
                elif row == 0 or column==0:
                    c_0 = self.c_0/2
                else:
                    c_0 = self.c_0
                    
                place_grid[row,column] = c_0*integrate.dblquad(self._beta_func,self.x_0,self.x_0+self.l_x,lambda x:self.y_0,lambda x:self.y_0+self.l_y,args=(row,column))[0]
        return place_grid
    def gamma_coeff_calc(self,n,m):
        datapoints = np.array(np.meshgrid(np.arange(0,n),np.arange(0,m))).transpose() 
        place_grid = np.zeros(datapoints.shape[:-1])
        for row in range(0,len(datapoints)):
            for column in range(0,len(datapoints[row])):
                if row == 0 and column==0:
                    c_0 = self.c_0/4
                elif row == 0 or column==0:
                    c_0 = self.c_0/2
                else:
                    c_0 = self.c_0
                place_grid[row,column] = c_0*integrate.dblquad(self._gamma_func,self.x_0,self.x_0+self.l_x,lambda x:self.y_0,lambda x:self.y_0+self.l_y,args=(row,column))[0]
        return place_grid
    def delta_coeff_calc(self,n,m):
        datapoints = np.array(np.meshgrid(np.arange(0,n),np.arange(0,m))).transpose() 
        place_grid = np.zeros(datapoints.shape[:-1])
        for row in range(0,len(datapoints)):
            for column in range(0,len(datapoints[row])):
                if row == 0 and column==0:
                    c_0 = self.c_0/4
                elif row == 0 or column==0:
                    c_0 = self.c_0/2
                else:
                    c_0 = self.c_0
                place_grid[row,column] = c_0*integrate.dblquad(self._delta_func,self.x_0,self.x_0+self.l_x,lambda x:self.y_0,lambda x:self.y_0+self.l_y,args=(row,column))[0]
        return place_grid
    
    def get_coefficents(self):
        c0 = self.alpha_coeff_calc(self.mn_level,self.mn_level)
        c1 = self.beta_coeff_calc(self.mn_level,self.mn_level)
        c2 = self.gamma_coeff_calc(self.mn_level,self.mn_level)
        c3 = self.delta_coeff_calc(self.mn_level,self.mn_level)
        rt_val = np.array([c0,c1,c2,c3])
        return rt_val

class TwoDimFourierEquasion():
    def __init__(self,coefficents):
        self.coefficents = coefficents
        self.n_list = None
    def _alpha_calc(self,x,y,l_x,l_y):
        nval =self.coefficents[0].copy()
        for row in range(0,len(nval)):
            for column in range(0,len(nval[row])):
                nval[row,column] = nval[row,column]*np.cos(2*np.pi*(row)*x/l_x)*np.cos(2*np.pi*(column)*y/l_y)
        return np.sum(nval)
    def _beta_calc(self,x,y,l_x,l_y):
        nval =self.coefficents[1].copy()
        for row in range(0,len(nval)):
            for column in range(0,len(nval[row])):
                nval[row,column] = nval[row,column]*np.cos(2*np.pi*(row)*x/l_x)*np.sin(2*np.pi*(column)*y/l_y)
        return np.sum(nval)
    def _gamma_calc(self,x,y,l_x,l_y):
        nval =self.coefficents[2].copy()
        for row in range(0,len(nval)):
            for column in range(0,len(nval[row])):
                nval[row,column] = nval[row,column]*np.sin(2*np.pi*(row)*x/l_x)*np.cos(2*np.pi*(column)*y/l_y)
        return np.sum(nval)
    def _delta_calc(self,x,y,l_x,l_y):
        nval =self.coefficents[3].copy()
        for row in range(0,len(nval)):
            for column in range(0,len(nval[row])):
                nval[row,column] = nval[row,column]*np.sin(2*np.pi*(row)*x/l_x)*np.sin(2*np.pi*(column)*y/l_y)
        return np.sum(nval)
    
    def alpha_calc(self,x,y,l_x,l_y):
        nval =self.coefficents[0].copy()
        n_list = np.meshgrid(np.arange(0,nval.shape[0]),np.arange(0,nval.shape[1]))
        nval[:] = nval[:]*np.cos(2*np.pi*(n_list[0])*x/l_x)*np.cos(2*np.pi*(n_list[1])*y/l_y)
        return np.sum(nval)
    
    def beta_calc(self,x,y,l_x,l_y):
        nval =self.coefficents[1].copy()
        n_list = np.meshgrid(np.arange(0,nval.shape[0]),np.arange(0,nval.shape[1]))
        nval[:] = nval[:]*np.cos(2*np.pi*(n_list[0])*x/l_x)*np.sin(2*np.pi*(n_list[1])*y/l_y)
        return np.sum(nval)
    
    def gamma_calc(self,x,y,l_x,l_y):
        nval=self.coefficents[2].copy()
        n_list = np.meshgrid(np.arange(0,nval.shape[0]),np.arange(0,nval.shape[1]))
        nval[:] = nval[:]*np.sin(2*np.pi*(n_list[0])*x/l_x)*np.cos(2*np.pi*(n_list[1])*y/l_y)
        return np.sum(nval)
    
    def delta_calc(self,x,y,l_x,l_y):
        nval=self.coefficents[3].copy()
        n_list = np.meshgrid(np.arange(0,nval.shape[0]),np.arange(0,nval.shape[1]))
        nval[:] = nval[:]*np.sin(2*np.pi*(n_list[0])*x/l_x)*np.sin(2*np.pi*(n_list[1])*y/l_y)
        return np.sum(nval)
    
    def calc_point(self,x,y,l_x,l_y):
        return self.alpha_calc(x,y,l_x,l_y)+self.beta_calc(x,y,l_x,l_y)+self.gamma_calc(x,y,l_x,l_y)+self.delta_calc(x,y,l_x,l_y)
    
    def calc_point_faster(self,x,y,l_x,l_y):
        if type(self.n_list) == type(None):
            self.n_list = np.mgrid[0:self.coefficents.shape[1],0:self.coefficents.shape[2]]
        n_list = self.n_list
        c1 = (2*np.pi*x/l_x)*(n_list[0])
        c2 = (2*np.pi*y/l_y)*(n_list[1])
        s1 = np.sin(c1)
        s2 = np.sin(c2)
        cs1 = np.cos(c1)
        cs2 = np.cos(c2)
        return np.sum(self.coefficents[3]*s1*s2+self.coefficents[0]*cs1*cs2+self.coefficents[2]*s1*cs2+self.coefficents[1]*cs1*s2)

        
    def get_values(self,x_start,x_stop,y_start,y_stop,x_step=.01,y_step=.01):
        vals = np.transpose(np.meshgrid(np.arange(x_start,x_stop,x_step),np.arange(y_start,y_stop,y_step)))
        return [[(column[0],column[1],self.calc_point_faster(column[0],column[1],x_stop-x_start,y_stop-y_start)) for column in row] for row in vals]
    
    def plotEquasion(self,x_start,x_stop,y_start,y_stop,y_step=.01,x_step=.01):
        x_l = x_stop-x_start
        y_l = y_stop-y_start
        values = self.get_values(x_start,x_stop,y_start,y_stop,x_step,y_step)
        Vals = np.array(values).transpose()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(Vals[0], Vals[1], Vals[2], 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
                
class HeatEquasion():
    def __init__(self,intial_state_function,x_start,x_stop,y_start,y_stop,F_depth):
        self.coeffCalcs = FourierTransformCalculations(InitialHeatEquasion,x_start,x_stop,y_start,y_stop,F_depth)
        self.coeffs = self.coeffCalcs.get_coefficents()
        self.x_start = x_start
        self.x_stop = x_stop
        
        self.y_start = y_start
        self.y_stop = y_stop
    @staticmethod
    def coefficent_reduction_factor(a,n,m,l_x,l_y):
        dd_x = (-4*(np.pi**2)*a*(n**2))/(l_x**2)
        dd_y = (-4*(np.pi**2)*a*(m**2))/(l_y**2)
        return (dd_x,dd_y)
    
    def rate_temp_change_calc(self):
        cf = self.coeffs.copy()
        for c_index in range(0,len(cf)):
            for row in range(0,len(cf[c_index])):
                for column in range(0,len(cf[c_index,row])):
                    cf[c_index,row,column] = np.sum(self.coefficent_reduction_factor(cf[c_index,row,column],row,column,self.coeffCalcs.l_x,self.coeffCalcs.l_y))
        return cf

    def coeff_temp_change_calc(self,dt,k):
        return self.rate_temp_change_calc()*dt*k

    def coeff_temp_change(self,t,dt,k=1):
        heat_equasion_times = []
        for time in np.arange(0,t,dt):
            heat_equasion_times.append(TwoDimFourierEquasion(self.coeffs))
            self.coeffs = self.coeffs+self.coeff_temp_change_calc(dt,k)
        return heat_equasion_times
    
# In[] Setup Models

H = HeatEquasion(InitialHeatEquasion,x_start,x_stop,y_start,y_stop,mn_level)
M = H.coeff_temp_change(t,dt,k)

# In[] Make Arrays with 2D values
import tqdm

def get_vals(obj):
    return obj[0].get_values(x_start,x_stop,y_start,y_stop)
values = [np.transpose(i.get_values(x_start,x_stop,y_start,y_stop,x_step,y_step)) for i in tqdm.tqdm(M)]
values = np.array(values)
zarray= values[:,2].T
x = values[0,0]
y = values[0,1]

# In[] Plot


#Thanks to this article for the animated plotting code https://pythonmatplotlibtips.blogspot.com/2018/11/animation-3d-surface-plot-funcanimation-matplotlib.html

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="seismic")
    ax.view_init(elev=50, azim=frame_number*.25)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0,1.1)
ani = animation.FuncAnimation(fig, update_plot, len(zarray[0,0]), fargs=(zarray, plot), interval=1000/fps)

print("Saving")
ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
exit()

