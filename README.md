# HeatEquasionModel
A python model of the 2D heat equation

# Usage
Under the section of code titled variables to alter, change the the variables to the way you want them to be.
The section should initially look like this :
```
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
```
Each variable here has a useage
  - n_level -> the value of m and n of the fourier series (Read about it "How it works" or here: 
-
