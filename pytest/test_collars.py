"""Tests movement of a rod under gravitational load connected with collar joints
to two fixed rods, one horizontal and one vertical. Verifies implementation of 
a constraint locking translations only along given axes. The speed of the vertical
collar is compared to the analytical solution, from Boström (2018), exercise 4.19"""

import sys
sys.path.append("..")
import MBS
from MBS import constraints as c
from MBS import positioning
from MBS import simulation as sim
from MBS import visualization as v
from MBS.transformations import R, skew
import numpy as np
import matplotlib.pyplot as plt

def test_collar_joints():
  dt = 1e-2
  seconds = 3.8
  steps = int(seconds/dt)
  g = 9.81
  rodmass = 1.53 # has no influence on results
  a = 1. # length scale of rods
  length = 5*a
  
  rod = MBS.Body(rodmass, np.diag([1., rodmass*length**2/12, rodmass*length**2/12]), force=[0.,0.005*g*rodmass,-g*rodmass], r=[1.5*a,0,2*a], n=[0,1,0], phi=np.arccos(3*a/length)) 
  # initial tilted placement of the rod. A small horizontal force is applied to stir the equilibrium
  bodies = [rod]
  ground = MBS.Ground()
  
  up = MBS.Point([-0.5*length,0,0], rod)
  lp = MBS.Point([0.5*length,0,0], rod)
  gup = MBS.Point([0,0,0], ground)
  glp = MBS.Point([3*a,0,0], ground)
  points = [up, lp, gup, glp]
  
  constraints = [] # collars
  constraints.append(c.Translational(rod, up, ground, gup, constaxes=[[1,0,0],[0,1,0]], coord='global'))
  constraints.append(c.Translational(rod, lp, ground, glp, constaxes=[[1,0,0],[0,0,1]], coord='global'))
  
  s, qdot = positioning.get_state(bodies)
  ssaved, qdotsaved = sim.simulate(dt, int(steps), s, qdot, bodies=bodies, constraints=constraints, gamma=0.25, beta=0.25, correction=True)
  
  upper_point_z_position = np.array([r[2,0]+(R(p).dot(up.u))[2,0] for r, p in zip(rod.rsaved, rod.psaved)])
  horizontal_time_index = np.argmin(abs(upper_point_z_position))
  upper_point_velocity = np.array([v+R(p).dot(skew(o).dot(up.u)) for v, o, p in zip(rod.vsaved, rod.omegasaved, rod.psaved)])
  speed_at_horizontal = upper_point_velocity[horizontal_time_index][2]
  analytical_solution = -np.sqrt(12*g*a)
  speed_error = abs((speed_at_horizontal - analytical_solution)/analytical_solution)
  print(speed_error)
  return rod, upper_point_velocity, ssaved, bodies, seconds
  assert(speed_error<0.003)
  
if __name__ == '__main__':
  rod, upper_point_velocity, ssaved, bodies, seconds = test_collar_joints()
  t = np.linspace(0, seconds, len(rod.vsaved))
  plt.plot(t, upper_point_velocity[:,0])
  plt.plot(t, upper_point_velocity[:,1])
  plt.plot(t, upper_point_velocity[:,2])
  plt.show()
  
  v.animate(ssaved, bodies, pt=30, zoom=0.15, rotspeed=0., n=[0.0,0.0,1.0], show_on_screen=True, savepics=False, directory='C:\\Users\\jaxas\\Documents\\Vidframes')
  
