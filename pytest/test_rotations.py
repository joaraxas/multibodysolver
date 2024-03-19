"""Two rods attached with a cylindrical constraint. One rod attached to 
the ground with a cylindrical constraint. Given initial rotational 
velocities, the displacement after a time is measured and compared to an
analytical solution. Tests constraints, quaternions and simulation."""

import sys
sys.path.append("..")
from MBS import Body, Ground, Point
from MBS.constraints import Spherical, Slider
from MBS.positioning import get_state, adjust_to_constraints
from MBS.simulation import simulate
from MBS.transformations import R
import numpy as np

# Imports for running the test locally
import matplotlib.pyplot as plt
import MBS.visualization as v

def common_setup(l1, l2, o1, o2):
  shaft = Body(m=2, I=np.eye(3), force=[0.0, 0.0, -1000.0], omega=[0.0, 0.0, o1], v=[0.0, o1*l1*0.5, 0.0])
  rotor = Body(m=3, I=np.eye(3), n=[0.0, 1.0, 0.0], phi=np.pi/2, force=[0.0, 0.0, -1000.0], omega=[-o1, 0.0, o2], v=[0.0, o1*l1, 0.0])
  bodies = [shaft, rotor]
  ground = Ground()
  
  p1 = Point([-l1*0.5, 0.0, 0.0], shaft)
  p2 = Point([l1*0.5, 0.0, 0.0], shaft)
  p3 = Point([-l2*0.5, 0.0, 0.0], rotor)
  p4 = Point([l2*0.5, 0.0, 0.0], rotor)
  p5 = Point([0.0, 0.0, 0.0], rotor)
  gp = Point([0.0, 0.0, 0.0], ground)
  points = [p1,p2,p3,p4,gp]
  
  leftc = Spherical(shaft, p1, ground, gp)
  lefta = Slider(shaft, p1, ground, gp, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  rightc = Spherical(shaft, p2, rotor, p5)
  righta = Slider(shaft, p2, rotor, p5, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  constraints = [leftc, lefta, rightc, righta]
  
  return bodies, points, constraints

def test_shaft_rotations():
  # Lengths of rods
  l1 = 1.8
  l2 = 1.6
  # angular speeds of rods
  o1 = 10.0
  o2 = 30.0
  
  dt = 1e-4
  #~ t = 0.02
  t = 0.02
  steps = int(t/dt)
  
  bodies, points, constraints = common_setup(l1, l2, o1, o2)
  
  shaft, rotor = bodies
  edge = points[3]
  s, qdot = get_state(bodies)
  adjust_to_constraints(s, bodies, constraints)
  ssaved, qdotsaved = simulate(dt, steps, s, qdot, bodies=bodies, constraints=constraints)
  edgepos = rotor.r+R(ssaved[-1][10:14]).dot(edge.u)
  analytical = [np.cos(o1*t)*l1 - np.sin(o1*t)*np.sin(o2*t)*l2*0.5, np.sin(o1*t)*l1 + np.cos(o1*t)*np.sin(o2*t)*l2*0.5, -np.cos(o2*t)*l2*0.5] # analytical solution
  position_error = np.linalg.norm(np.array(analytical) - edgepos.T)
  print(position_error)
  assert(position_error < 1e-3)
  return ssaved, bodies

if __name__ == "__main__":
  ssaved, bodies = test_shaft_rotations()
  v.animate(ssaved, bodies, pt=10, zoom=0.5)
  
