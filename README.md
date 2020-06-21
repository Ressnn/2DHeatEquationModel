# HeatEquasionModel
A python model of the 2D heat equation

# Usage
Under the section of code titled variables to alter, change the the variables to the way you want them to be.
The section should initially look like this :
```
mn_level = 5 

start = -2
stop = 2.0001
step =.05 

t=1.5
dt=.003

k = .2

fn = 'Sample4'
fps = 30

#Function here:
def InitialHeatEquasion(x,y):
    return np.sin(x+y)
```
Each variable alters the code like so:
### Function plotting
  - mn_level -> the value of m and n of the fourier series (Read about it "How it works") or [here](https://math.stackexchange.com/questions/211689/real-valued-2d-fourier-series)
  - start -> the value to start plotting the function
  - stop -> the value to stop plotting the function
  - step -> the step of the function plot
### Time
  - t -> the time to stop the simulation
  - dt -> how much to step the time per frame (higher values might crash or result in odd plots, lower values slower but better)
### Temprature
  - a -> the  thermal diffusivity of the material
### Saving
  - fn -> the file name
  - fps -> frames per second of animation
### InitialHeatEquasion
  - This equation determines the initial state of the simulator must take in two parameters (x,y) and return the z coordinate which will be the temprature

# How it Works

This program first does a 2D fourier approximation of the the given 2D equation.
This is because the heat equation takes takes the second derivative of the function and subtracts it from the function.
With a normal function this would result in a long chain of subtracted derivatives but because the second derivative of a
fourier coefficent is a multiple of itself (by a decimal factor) it makes it much easier to compute. After the approximation is calculated, at the given mn_level (more on what I did in the second answer [here](https://math.stackexchange.com/questions/211689/real-valued-2d-fourier-series)),
we then store how the function would morph with the diffusion of heat over time (in the TwoDimFourierEquasion class) and calculate it for exact points last.
