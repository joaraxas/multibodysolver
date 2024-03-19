"""Tests the spring and damper classes in a 1D problem and compares to
analytical solution."""

import sys
sys.path.append("..")
from MBS import Body, Ground, Point
from MBS.loads import Spring, Damper
from MBS.simulation import simulate
from MBS.positioning import get_state
import numpy as np

# Imports for running the test locally
import matplotlib.pyplot as plt
import MBS.visualization as v

def common_setup(k1, k2, c1, c2, m1, m2, F, x1, x2):
  left = Body(m=m1, I=np.eye(3), r=[x1, 0.0, 0.0])
  right = Body(m=m2, I=np.eye(3), r=[x2, 0.0, 0.0], force=[F, 0.0, 0.0])
  bodies = [left, right]
  g = Ground()
  
  llinkljoint = Point([-x1, 0.0, 0.0], left)
  llinkrjoint = Point([x1, 0.0, 0.0], left)
  rlinkljoint = Point([x1*2-x2, 0.0, 0.0], right)
  rlinkrjoint = Point([-x1*2+x2, 0.0, 0.0], right)
  groundpoint = Point([0.0, 0.0, 0.0], g)
  points = [llinkljoint, llinkrjoint, rlinkljoint, rlinkrjoint, groundpoint]
  
  leftspring = Spring(left, llinkljoint, g, groundpoint, k1)
  rightspring = Spring(left, llinkrjoint, right, rlinkljoint, k2)
  leftdamp = Damper(left, llinkljoint, g, groundpoint, c1)
  rightdamp = Damper(left, llinkrjoint, right, rlinkljoint, c2)
  loads = [leftspring, rightspring, leftdamp, rightdamp]
  
  return bodies, points, loads

def test_spring():
  # spring stiffnesses and damping coefficients
  k1 = 10000.0
  k2 = 5000.0
  c1 = 1000.0
  c2 = 500.0
  # rod masses
  m1 = 3.0
  m2 = 5.0
  # load
  F = 80.0
  # initial positions
  x1 = 0.5
  x2 = 2.0
  dt = 1.0e-3
  steps = 100
  
  bodies, points, loads = common_setup(k1, k2, c1, c2, m1, m2, F, x1, x2)
  
  # Time-integrated rigid body system for comparison
  x = np.array([x1,x2])
  x0 = np.array([x1,x2])
  xp = np.array([0.,0.])
  K = np.array([[(-k1-k2), k2], [k2, -k2]]) # stiffness matrix
  C = np.array([[(-c1-c2), c2], [c2, -c2]]) # damping matrix
  Q = np.array([0, F])  #load vector
  M = np.array([[m1, 0], [0, m2]]) # mass matrix
  xsaved = []
  for t in range(steps): # "Analytical" (integrated) solution
    xp += np.linalg.inv(M).dot(dt*(K.dot(x-x0)+C.dot(xp)+Q))
    x += dt*xp
    xsaved.append(x.copy())
  
  s, qdot = get_state(bodies)
  ssaved,_ = simulate(dt, steps, s, qdot, bodies=bodies, loads=loads)
  assert abs(ssaved[-1][0,0] - xsaved[-1][0]) < 1e-8
  return ssaved, xsaved

if __name__ == "__main__":
  ssaved, xsaved = test_spring()
  plt.plot(range(len(ssaved)), [ss[0] for ss in ssaved])
  plt.plot(range(len(ssaved)), [xs[0] for xs in xsaved])
  plt.show()
  print([ss[0] for ss in ssaved])
  print([ss[0] for ss in xsaved])
  print(ssaved[-1][0])
  print(xsaved[-1][0])
  
