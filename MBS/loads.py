import numpy as np

from MBS.transformations import R, skew#, doublecross
from MBS import Ground
norm = np.linalg.norm

def get_force_vector(body, point, fvector):
  '''Takes a point load and returns the resulting forces and moments.
  
  If a force F is applied at a point u in local coords, the resulting
  moment on the body is u x F.
  
  Args:
    body (Body obj): Used to get the orientation of its local coord system.
    point (Point obj): where the force is applied.
    fvector (array 3x1): The force vector in global coordinates.
  
  Returns:
    Q (array 6x1): Vector of force and resulting moment.
  
  '''
  Q = np.zeros((6,1))
  u = point.u
  p = body.p
  Q[0:3] = fvector
  Q[3:6] = skew(u).dot(R(p).T.dot(fvector))
  return Q


class Load():
  '''Parent class for all loads.'''
  def __init__(self, body1, body2, fvector):
    """Sets instance attributes body1, body2, fvector"""
    self.body1, self.body2, self.fvector = body1, body2, np.array(fvector).reshape((3,1))

class PointLoad(Load):
  '''Adds a load between two bodies in two points. The fvector is in the global system.'''
  def __init__(self, body1, point1, body2, point2, fvector):
    """PointLoad(body1, point1, body2, point2, fvector)
    
    Args:
      body1, body2 (Body or Ground obj): The load is applied to both bodies 
        with opposite direction (Newton 3).
      point1, point2 (Point obj): Points where the load is applied. Does 
        not define force direction.
      fvector (array (3) or 3-list): Load vector, global.
    
    Note that body1 and body2 currently override the points' owner 
    attributes. This increases readability when defining loads.
    Usually, one of the bodies is Ground, but this class could also be 
    used for user-defined springs etc.
    
    """
    super().__init__(body1, body2, fvector)
    self.point1, self.point2 = point1, point2
  
  def get_F(self):
    '''Q = get_F()
    Returns:
      Q (array 12x1): the load vector contribution of forces and moments.
    
    '''
    Q = np.zeros((12,1))
    Q[0:6] = get_force_vector(self.body1, self.point1, self.fvector)
    Q[6:12] = get_force_vector(self.body2, self.point2, -self.fvector)
    return Q
 
class Torque(Load):
  '''Adds a free torque to a body. The coordinate system can be specified.'''
  def __init__(self, body1, body2, fvector, coord='global'):
    """PointLoad(body1, body2, fvector, coord='global')
    
    Args:
      body1, body2 (Body or Ground obj): The torque is applied to both bodies 
        with opposite direction (Newton 3).
      fvector (array (3) or 3-list): Torque vector.
      coord (string, optional): Defines the coordinate system of the torque vector.
        Can be either "global", "local1" (for body1) or "local2" (for body2). 
        Defaults to "global".
    
    """
    super().__init__(body1, body2, fvector)
    self.coord = coord
    if coord not in ['global', 'local', 'local1', 'local2']:
      print('incorrect coordinate system definition in Torque')
      quit()
  
  def get_F(self):
    '''Q = get_F()
    Returns:
      Q (array 12x1): the load vector contribution of moments.
    
    '''
    p1, p2 = self.body1.p, self.body2.p
    F = np.zeros((12,1))
    if self.coord == 'local' or self.coord == 'local1':
      F[3:6] = self.fvector
      F[9:12] = -R(p2).T.dot(R(p1)).dot(self.fvector)
    elif self.coord == 'local2':
      F[3:6] = -R(p1).T.dot(R(p2)).dot(self.fvector)
      F[9:12] = self.fvector
    elif self.coord == 'global':
      F[3:6] = R(p1).T.dot(self.fvector)
      F[9:12] = -R(p2).T.dot(self.fvector)
    return F
    
class NormalLoad(Load):
  '''Repulsive or attractive load in the direction between two points.'''
  def __init__(self, body1, point1, body2, point2, magnitude):
    """NormalLoad(body1, point1, body2, point2, magnitude)
    
    Args:
      body1, body2 (Body or Ground obj): The load is applied to both bodies 
        with opposite direction (Newton 3).
      point1, point2 (Point obj): Points connected by the spring. This 
        defines the force direction. 
      magnitude (float): Force magnitude in N. Positive values for repulsion.
    
    Note that body1 and body2 currently override the points' owner 
    attributes. This increases readability when defining loads.
    
    """
    super().__init__(body1, body2, [0.0, 0.0, 0.0])
    self.point1, self.point2, self.magnitude = point1, point2, magnitude
  def get_F(self):
    '''Q = get_F()
    Returns:
      Q (array 12x1): the load vector contribution of forces and moments.
    
    '''
    u1bar, u2bar = self.point1.u, self.point2.u
    p1, p2 = self.body1.p, self.body2.p
    r1, r2 = self.body1.r, self.body2.r
    u1 = r1 + R(p1).dot(u1bar)
    u2 = r2 + R(p2).dot(u2bar)
    self.fvector = -self.magnitude*(u2-u1)/norm(u2-u1)
    F = np.zeros((12, 1))
    F[0:6] = get_force_vector(self.body1, self.point1, self.fvector)
    F[6:12] = get_force_vector(self.body2, self.point2, -self.fvector)
    return F
        
class Spring(Load):
  '''Load between two bodies relative to the distance between two points'''
  def __init__(self, body1, point1, body2, point2, stiffness):
    """Spring(body1, point1, body2, point2, stiffness)
    
    Args:
      body1, body2 (Body or Ground obj): The load is applied to both bodies 
        with opposite direction (Newton 3).
      point1, point2 (Point obj): Points connected by the spring. This 
        defines the force direction. 
      stiffness (float): N/m
    
    Note that body1 and body2 currently override the points' owner 
    attributes. This increases readability when defining loads.
    
    """
    super().__init__(body1, body2, [0.0, 0.0, 0.0])
    self.point1, self.point2, self.k = point1, point2, stiffness
  def get_F(self):
    '''Q = get_F()
    Returns:
      Q (array 12x1): the load vector contribution of forces and moments.
    
    '''
    u1bar, u2bar = self.point1.u, self.point2.u
    p1, p2 = self.body1.p, self.body2.p
    r1, r2 = self.body1.r, self.body2.r
    u1 = r1 + R(p1).dot(u1bar)
    u2 = r2 + R(p2).dot(u2bar)
    self.fvector = self.k*(u2-u1)
    F = np.zeros((12, 1))
    F[0:6] = get_force_vector(self.body1, self.point1, self.fvector)
    F[6:12] = get_force_vector(self.body2, self.point2, -self.fvector)
    return F
    
class Damper(Load):
  '''Load between two bodies relative to the velocity difference of two points'''
  def __init__(self, body1, point1, body2, point2, dampingcoefficient):
    """Damper(body1, point1, body2, point2, dampingcoefficient)
    
    Args:
      body1, body2 (Body or Ground obj): The load is applied to both bodies 
        with opposite direction (Newton 3). The load direction is v2-v1.
      point1, point2 (Point obj): Points connected by the dashpot.
      dampingcoefficient (float): kg/s
    
    Note that body1 and body2 currently override the points' owner 
    attributes. This increases readability when defining loads.
    
    """
    super().__init__(body1, body2, [0.0, 0.0, 0.0])
    self.point1, self.point2, self.c = point1, point2, dampingcoefficient
  def get_F(self):
    '''Q = get_F()
    Returns:
      Q (array 12x1): the load vector contribution of forces and moments.
    
    '''
    u1bar, u2bar = self.point1.u, self.point2.u
    p1, p2 = self.body1.p, self.body2.p
    r1, r2 = self.body1.r, self.body2.r
    o1, o2 = self.body1.omega, self.body2.omega
    v1, v2 = self.body1.v, self.body2.v
    u1dot = v1 + R(p1).dot(skew(o1)).dot(u1bar)
    u2dot = v2 + R(p2).dot(skew(o2)).dot(u2bar)
    self.fvector = self.c*(u2dot-u1dot)
    F = np.zeros((12,1))
    F[0:6] = get_force_vector(self.body1, self.point1, self.fvector)
    F[6:12] = get_force_vector(self.body2, self.point2, -self.fvector)
    return F
     
class MultiaxialSpring(Load):
  '''Spring with different stiffnesses in different directions'''
  def __init__(self, body1, point1, body2, point2, normal_stiffness, 
                  tangential_stiffness, normal_direction):
    """MultiaxialSpring(body1, point1, body2, point2, normal_stiffness, 
                  tangential_stiffness)
    
    Args:
      body1, body2 (Body or Ground obj): The load is applied to both bodies 
        with opposite direction (Newton 3).
      point1, point2 (Point obj): Points connected by the spring, where the 
        force is applied
      normal_stiffness (float): N/m 
      tangential_stiffness (float): N/m
      normal_direction (3-list): normal direction of the spring, in the
        coordinate system of body1
    
    Note that body1 and body2 currently override the points' owner 
    attributes. This increases readability when defining loads.
    
    """
    super().__init__(body1, body2, [0.0, 0.0, 0.0])
    self.point1, self.point2 = point1, point2
    self.kn, self.kt = normal_stiffness, tangential_stiffness
    self.n = np.reshape(normal_direction, (3,1))
    self.n = self.n/norm(self.n)
  def get_F(self):
    '''Q = get_F()
    Returns:
      Q (array 12x1): the load vector contribution of forces and moments.
    
    '''
    u1bar, u2bar = self.point1.u, self.point2.u
    p1, p2 = self.body1.p, self.body2.p
    r1, r2 = self.body1.r, self.body2.r
    u1 = r1 + R(p1).dot(u1bar)
    u2 = r2 + R(p2).dot(u2bar)
    displacement = u2-u1
    global_normal = R(p1).dot(self.n)
    normal_displacement = displacement.T.dot(global_normal)*global_normal
    tangential_displacement = displacement - normal_displacement
    self.fvector = self.kn*normal_displacement + self.kt*tangential_displacement
    F = np.zeros((12, 1))
    F[0:6] = get_force_vector(self.body1, self.point1, self.fvector)
    F[6:12] = get_force_vector(self.body2, self.point2, -self.fvector)
    return F
