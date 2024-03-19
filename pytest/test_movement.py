"""Tests movement of an unconstrained body subject to translational and
rotational load. The total displacement is measured and compared to an
analytical and a previous result."""

import sys
sys.path.append("..")
from MBS import Body, Ground, Point
from MBS.loads import PointLoad, Torque
from MBS.positioning import get_state, adjust_to_constraints
from MBS.simulation import simulate
from MBS.transformations import R
import numpy as np

# Imports for running the test locally
import matplotlib.pyplot as plt
import MBS.visualization as v

def common_setup(mass, inertiaTensor, force, torque):
  shaft = Body(m=mass, I=inertiaTensor, torque=torque)
  bodies = [shaft]
  ground = Ground()
  
  p1 = Point([1.0, 0.0, 0.0], shaft)
  p2 = Point([0.0, 0.0, 0.0], shaft)
  gp = Point([0.0, 0.0, 0.0], ground)
  points = [p1,p2,gp]
  
  loads = []
  loads.append(PointLoad(shaft, p1, ground, gp, force))
  
  return bodies, points, loads

def test_displacement():
  mass = 3
  inertiaTensor = np.diag([2.0, 3.0, 5.0])
  force = [3.0, 2.0, 8.0]
  torque = [1.0, 4.0, 6.0]
  
  dt = 1e-3
  seconds = 1.0
  steps = int(seconds/dt)
  
  bodies, points, loads = common_setup(mass, inertiaTensor, force, torque)
  shaft = bodies[0]
  
  s, qdot = get_state(bodies)
  ssaved, qdotsaved = simulate(dt, steps, s, qdot, bodies=bodies, loads=loads)
  analyticaltrans = np.array(force)/mass*seconds**2*0.5         # analytical result
  measuredrot = np.array([0.97156646, -0.55264952, 1.55450075]) # measured in an earlier simulation
  #~ translation_error = np.linalg.norm(ssaved[-1][0:3].T - analyticaltrans)
  translation_error = np.linalg.norm(shaft.r.T - analyticaltrans)
  rotation_error = np.linalg.norm(shaft.omega.T - measuredrot)
  print(translation_error)
  print(rotation_error)
  assert(translation_error < 1.0e-2)
  assert(rotation_error < 1.0e-2)
  
  return ssaved, bodies, qdotsaved

if __name__ == "__main__":
  ssaved, bodies, qdotsaved = test_displacement()
  v.animate(ssaved, bodies, pt=100)
  print(ssaved[-1][0:3].T)
  print(qdotsaved[-1][3:6].T)
  
