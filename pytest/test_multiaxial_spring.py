"""Tests the multiaxial spring class in a 2D problem. Asserts that a tangential
displacement gives no force from a multiaxial spring with zero tangential 
stiffness."""

from MBS import Body, Ground, Point
from MBS.loads import MultiaxialSpring
from MBS.simulation import simulate
from MBS.positioning import get_state
from MBS.visualization import animate
import numpy as np

# Imports for running the test locally
import matplotlib.pyplot as plt
import MBS.visualization as v

def common_setup(kn, kt, x1, x2, z1, z2, l1, l2, phi1, phi2):
  left = Body(m=3., I=np.eye(3), r=[x1, 0.0, z1], n=[0,-1,0], phi=phi1)
  right = Body(m=5., I=np.eye(3), r=[x2, 0.0, z2], n=[0,-1,0], phi=phi2)
  bodies = [left, right]
  g = Ground()
  
  llinkljoint = Point([-l1/2, 0.0, 0.0], left)
  llinkrjoint = Point([l1/2, 0.0, 0.0], left)
  rlinkljoint = Point([-l2/2, 0.0, 0.0], right)
  rlinkrjoint = Point([l2/2, 0.0, 0.0], right)
  groundpoint = Point([0.0, 0.0, 0.0], g)
  points = [llinkljoint, llinkrjoint, rlinkljoint, rlinkrjoint, groundpoint]
  
  spring = MultiaxialSpring(left, llinkrjoint, right, rlinkljoint, kn, kt, [1,0,0])
  loads = [spring]
  
  return bodies, points, loads

def test_spring():
  # spring stiffnesses
  kn = 500000.0
  kt = 0.0
  # initial positions
  phi1 = -np.pi/6
  phi2 = 0.
  l1 = 0.4
  l2 = 0.2
  x1 = 0.0
  x2 = l1/2/np.cos(phi1)+l2/2
  z1 = 0.0
  z2 = 0.0
  dt = 1.0e-3
  steps = 100
  
  bodies, points, loads = common_setup(kn, kt, x1, x2, z1, z2, l1, l2, phi1, phi2)
  
  s, qdot = get_state(bodies)
  ssaved, qdotsaved = simulate(dt, steps, s, qdot, bodies=bodies, loads=loads)
  assert np.linalg.norm(qdotsaved[-1][:,0]) < 1e-8
  return ssaved, bodies#, xsaved

if __name__ == "__main__":
  #~ ssaved, xsaved = test_spring()
  ssaved, bodies = test_spring()
  plt.plot(range(len(ssaved)), [ss[0] for ss in ssaved])
  plt.show()
  animate(ssaved, bodies, pt=1)
  print([ss[0] for ss in ssaved])
  print([ss[0] for ss in xsaved])
  print(ssaved[-1][0])
  print(xsaved[-1][0])
  
