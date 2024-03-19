"""Tests movement of a rod connected with collar joints to a fixed vertical bar, 
and to a horizontal bar connected to and rotating around the vertical bar. 
Verifies implementation of a constraint locking translations only along given axes. 
The angular acceleration of the tilted rod is compared to the analytical solution, 
from Boström (2018), exercise 3.2"""

import sys
sys.path.append("..")
import MBS
from MBS import constraints as c
from MBS import positioning
from MBS import simulation as sim
from MBS import visualization as v
from MBS import loads as l
from MBS.transformations import R, skew
import numpy as np
import matplotlib.pyplot as plt

def test_collar_rotation():
  dt = 1e-5
  seconds = 0.01
  steps = int(seconds/dt)
  rodmass = 2.53 # has no influence on results
  barmass = rodmass
  collar_zvel = 0.2
  collar_zpos = 0.075
  length = 0.125
  collar_ypos = np.sqrt(length**2-collar_zpos**2)
  collar_yvel = -collar_zvel*collar_zpos/collar_ypos
  omega_x = collar_zvel/collar_ypos
  Omega = 2.0
  phi = np.arcsin(collar_zpos/length) # initial angle of rod
  
  rod = MBS.Body(rodmass, np.diag([rodmass*length**2/12, 0.001*rodmass*length**2/12, rodmass*length**2/12]), r=[0,collar_ypos*0.5,collar_zpos*0.5], n=[1,0,0], phi=-phi, v=[-collar_ypos*0.5*Omega,0.5*collar_yvel,0.5*collar_zvel], omega=[-omega_x,-np.sin(phi)*Omega,np.cos(phi)*Omega])
  bar = MBS.Body(rodmass, np.diag([rodmass*length**2/12, 0.001*rodmass*length**2/12, rodmass*length**2/12]), omega=[0,0,Omega], torque=[0,0,-0.050594])
  bodies = [rod, bar]
  ground = MBS.Ground()
  
  up = MBS.Point([0,-0.5*length,0], rod)
  lp = MBS.Point([0,0.5*length,0], rod)
  gp = MBS.Point([0,0,0], ground)
  gup = MBS.Point([0,0,length], ground)
  bp = MBS.Point([0,0,0], bar)
  blp = MBS.Point([0,length,0], bar)
  points = [up, lp, gp, bp, gup, blp]
  points.append(MBS.Point([0,-length,0], bar))
  
  constraints = []
  constraints.append(c.Spherical(bar, bp, ground, gp))
  constraints.append(c.Slider(bar, bp, ground, gp, constaxes=[[1,0,0],[0,1,0]]))
  constraints.append(c.Translational(ground, gup, rod, up, constaxes=[[1,0,0],[0,1,0]], coord='global'))
  constraints.append(c.Translational(bar, blp, rod, lp, constaxes=[[1,0,0],[0,0,1]], coord='local1'))
  
  loads = []
  # loads = [l.PointLoad(rod, up, ground, gp, [0,0,0.649])]
  
  s, qdot = positioning.get_state(bodies)
  ssaved, qdotsaved = sim.simulate(dt, int(steps), s, qdot, bodies=bodies, constraints=constraints, loads=loads)
  print((R(rod.psaved[1]).dot(rod.omegasaved[1]) - R(rod.psaved[0]).dot(rod.omegasaved[0]))/dt)
  print((rod.omegasaved[1] - rod.omegasaved[0])/dt)
  print(R(rod.psaved[1]).dot(skew(rod.omegasaved[1]).dot(up.u))+rod.vsaved[1])
  print(bar.omegasaved[1])
  quit()
  
  # upper_point_z_position = np.array([r[2,0]+(R(p).dot(up.u))[2,0] for r, p in zip(rod.rsaved, rod.psaved)])
  # horizontal_time_index = np.argmin(abs(upper_point_z_position))
  # upper_point_velocity = np.array([v+R(p).dot(skew(o).dot(up.u)) for v, o, p in zip(rod.vsaved, rod.omegasaved, rod.psaved)])
  # speed_at_horizontal = upper_point_velocity[horizontal_time_index][2]
  # analytical_solution = -np.sqrt(12*g*a)
  # speed_error = abs((speed_at_horizontal - analytical_solution)/analytical_solution)
  # print(speed_error)
  return dt, ssaved, bodies, seconds
  # assert(speed_error<0.003)
  
if __name__ == '__main__':
  dt, ssaved, bodies, seconds = test_collar_rotation()
  # t = np.linspace(0, seconds, len(rod.vsaved))
  # plt.plot(t, upper_point_velocity[:,0])
  # plt.plot(t, upper_point_velocity[:,1])
  # plt.plot(t, upper_point_velocity[:,2])
  # plt.show()
  
  v.animate(ssaved, bodies, pt=int(0.01/dt), zoom=6, rotspeed=0., n=[0.0,0.0,1.0], show_on_screen=True, savepics=True, directory='C:\\Users\\jaxas\\Documents\\Vidframes')
  
