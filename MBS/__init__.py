import numpy as np
from MBS.transformations import get_p

class Point():
  '''Body-specific local coordinate references for constraints and loads.'''
  def __init__(self, vector, owner):
    """Point(vector, owner)
    
    Args:
      vector (3-list or array): Local coordinates for the point.
      owner (Body or Ground obj): Specifies the coordinate system.
    
    (Currently the owner property only affects visualisation. You can use a Point object 
    for other bodies than its owner, but it is not recommended.)
    
    """
    self.u = np.reshape(vector, (3,1))
    self.owner = owner
    owner.points.append(self)

class Body():
  '''Bodies expand the mass matrix and the number of dofs. They have an optional force
  and torque for convenience. The force is in the MC of the body in global coords, and 
  the torque is local. Initial position, velocity and angular velocity can be set.
  Initial rotation can be set by passing a normal vector n and a rotation phi in radians.'''
  def __init__(self, m, I, force=[0.,0.,0.], torque=[0.,0.,0.], n=[0.,0.,1.], phi=0, r=[0,0,0], v=[0,0,0], omega=[0,0,0]):
    """Body(m, I, force=[0.,0.,0.], torque=[0.,0.,0.], n=[0.,0.,1.], phi=0, r=[0,0,0], v=[0,0,0], omega=[0,0,0])
    
    Args:
      m (float): mass
      I (array 3x3): Inertia tensor in local coordinates
      force (3-list or 3x1 array, optional): A constant force in global 
        coords. E.g. gravity. Defaults to [0,0,0].
      torque (3-list or 3x1 array, optional): A constant torque in local 
        coords. Defaults to [0,0,0].
      n (3-list or 3x1 array, optional): A normal vector for initial rotation.
         Defaults to [0,0,1].
      phi (float, optional): Angle in radians for initial rotation about n
        Defaults to 0.
      r (3-list or 3x1 array, optional): initial position (global). Defaults
        to [0,0,0].
      v (3-list or 3x1 array, optional): initial translational velocity 
        (global). Defaults to [0,0,0].
      omega (3-list or 3x1 array, optional): initial angular velocity 
        (local). Defaults to [0,0,0].
    
    Note that setting r may still lead to problems with adjust_to_constraints.
      
    """
    self.m = m
    self.I = I
    self.force = np.reshape(force, (3,1))
    self.torque = np.reshape(torque, (3,1))
    self.r = np.reshape(r, (3,1))
    self.v = np.reshape(v, (3,1))
    self.omega = np.reshape(omega, (3,1))
    self.p = get_p(n, phi)
    self.rsaved = []
    self.vsaved = []
    self.omegasaved = []
    self.psaved = []
    self.points = []
    
class Ground():
  '''Special object that can be passed to constraints and loads but has 
  no dofs and does not move.'''
  def __init__(self):
    """Ground()"""
    self.r = np.array([[0.,0.,0.]]).T
    self.v = np.array([[0.,0.,0.]]).T
    self.omega = np.array([[0.,0.,0.]]).T
    self.p = np.array([[1.,0.,0.,0.]]).T
    self.points = []

class Regulator():
  '''PID regulator.'''
  def __init__(self, r, y, K, Ti, Td, regulated_load=-1):
    """Regulator(K, invTi, Td)
    
    Args:
      r (variable reference): reference signal, goal
      y (variable reference): insignal
      K (float): constant for all regulator parts
      Ti (float): Ti, time constant for I regulator
      Td (float): time constant for D regulator
      regulated_load (Load object, optional): the Load object whose fvector 
        will be regulated
    
    u = K(e + 1/Ti*integral(e) + Td*de/dt)
      
    """
    self.r, self.y, self.K, self.Ti, self.Td, self.regulated_load = r, y, K, Ti, Td, regulated_load
    self.Iold = 0.
    self.eold = self.r - self.y
  def get_signal(self, dt):
    """get_signal(dt)
    
    Args:
      dt (float): time step, sampling time
    
    Returns:
      u (float): outsignal
    
    u = K(e + 1/Ti*integral(e) + Td*de/dt)
      
    """
    e = self.r - self.y
    I = self.Iold + dt/self.Ti*e
    u = self.K*(e + I + self.Td/dt*(e-self.eold))
    self.Iold = I
    self.eold = e
    return u
  def set_signal(self, dt):
    self.regulated_load.fvector = self.get_signal(dt)
    
