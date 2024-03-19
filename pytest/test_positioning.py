"""Tests the positioning functionality. Two rods are rotated and constrained
to one another and it is asserted that they end up at the correct positions."""

import sys
sys.path.append("..")
from MBS import Body, Ground, Point
from MBS.constraints import Spherical, Fixed
from MBS.positioning import get_state, adjust_to_constraints
import numpy as np

# Imports for running the test locally
import matplotlib.pyplot as plt
import MBS.visualization as v

def common_setup(xgp, n1, phi1, n2, phi2, l1, l2):
  left = Body(m=1, I=np.eye(3), n=n1, phi=phi1)
  right = Body(m=1, I=np.eye(3), n=n2, phi=phi2)
  bodies = [left, right]
  ground = Ground()
  
  llinkljoint = Point([-l1*0.5, 0.0, 0.0], left)
  llinkrjoint = Point([l1*0.5, 0.0, 0.0], left)
  rlinkljoint = Point([-l2*0.5, 0.0, 0.0], right)
  rlinkrjoint = Point([l2*0.5, 0.0, 0.0], right)
  groundpoint = Point(xgp, ground)
  points = [llinkljoint, llinkrjoint, rlinkljoint, rlinkrjoint, groundpoint]
  
  leftc = Spherical(left, llinkljoint, ground, groundpoint)
  rightc = Fixed(left, llinkrjoint, right, rlinkljoint)
  constraints = [rightc]
  positioningconstraints = [leftc]
  
  return bodies, points, constraints, positioningconstraints

def test_positioning():
  # lengths of rods
  l1 = 0.8
  l2 = 2.3
  # position of fixation of rod 1 to ground
  xgp = [1.2, -1.9, 3.4]
  # rotations of rods
  n1 = [0.0, 1.0, 0.0]
  n2 = [0.0, 0.0, 1.0]
  phi1 = np.pi/7
  phi2 = np.pi/3
  
  # Analytical solutions
  pos1 = [xgp[0] + l1*np.cos(phi1)*0.5, xgp[1], xgp[2]-l1*np.sin(phi1)*0.5]
  lnkpos = [xgp[0] + l1*np.cos(phi1), xgp[1], xgp[2]-l1*np.sin(phi1)]
  pos2 = [lnkpos[0] + l2*np.cos(phi2)*0.5, lnkpos[1] + l2*np.sin(phi2)*0.5, lnkpos[2]]
  
  bodies, points, constraints, positioningconstraints = common_setup(xgp, n1, phi1, n2, phi2, l1, l2)
  
  rod1, rod2 = bodies
  s, qdot = get_state(bodies)
  adjust_to_constraints(s, bodies, constraints, positioningconstraints)
  assert np.linalg.norm(np.array(pos1) - rod1.r.T) < 1e-8
  assert np.linalg.norm(np.array(pos2) - rod2.r.T) < 1e-8
  return s, bodies, pos2

if __name__ == "__main__":
  s, bodies, pos2 = test_positioning()
  print(s[7:10].T)
  print(pos2)
  v.plot_bodies(bodies, s)
  plt.show()
  
