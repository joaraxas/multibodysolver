import numpy as np
from MBS import Ground
from MBS.transformations import R, skew, doublecross

class Constraint():
  '''Parent class for all constraints.'''
  def __init__(self, body1, point1, body2, point2):
    """Sets instance attributes body1, body2, point1, point2"""
    self.body1, self.body2, self.point1, self.point2 = body1, body2, point1, point2

class Spherical(Constraint):
  '''Locks all translational dofs between two points in two entities (bodies or ground)'''
  def __init__(self, body1, point1, body2, point2, constaxes=np.eye(3), coord='global'):
    """Spherical(body1, point1, body2, point2, constaxes=np.eye(3), coord='global')
    
    Args:
      body1, body2 (Body or Ground obj): Connected bodies.
      point1, point2 (Point obj): Points that must coincide
    
    Note that body1 and body2 currently override the points' owner attributes. This
    increases readability when defining constraints.
    
    """
    super().__init__(body1, point1, body2, point2)
    
  def get_J(self):
    '''J, Fc = get_J()
    Constraint Jacobian and constraint force.
    
    Returns:
      J (array 3x12): Jacobian of the constraint equations C. To be added to the mass
        matrix if a Lagrange multiplier scheme is used for simulation.
      Fc (array 3x1): Constraint force vector. A number of terms that appear when 
        differentiating C and that must be added to the load vector.
    
    '''
    u1, u2 = self.point1.u, self.point2.u
    r1, r2 = self.body1.r, self.body2.r
    v1, v2 = self.body1.v, self.body2.v
    p1, p2 = self.body1.p, self.body2.p
    o1, o2 = self.body1.omega, self.body2.omega
    J = np.zeros([3,12])
    Fc = np.zeros([3,1])
    J[:, 0:3] = -np.eye(3)
    J[:, 3:6] = -R(p1).dot(skew(u1).T)
    Fc += R(p1).dot(doublecross(o1)).dot(u1)
    J[:, 6:9] = np.eye(3)
    J[:, 9:12] = R(p2).dot(skew(u2).T)
    Fc += -R(p2).dot(doublecross(o2)).dot(u2)
    return J, Fc
    
  def get_C(self):
    '''C = get_C()
    Value of the constraint function (only for positioning or error estimation)
    
    Returns:
      C (array 3x1): The distance between the two points that are supposed to coincide.'''
    r1, r2 = self.body1.r, self.body2.r
    p1, p2 = self.body1.p, self.body2.p
    u1, u2 = self.point1.u, self.point2.u
    C = np.zeros((3,1))
    if not isinstance(self.body1, Ground):
      C += -r1 - R(p1).dot(u1)
    if not isinstance(self.body2, Ground):
      C +=  r2 + R(p2).dot(u2)
    return C

class Translational(Constraint):
  '''Locks translational dofs between two points in two entities (bodies or ground)'''
  def __init__(self, body1, point1, body2, point2, constaxes=np.eye(3), coord='global'):
    """Translational(body1, point1, body2, point2, constaxes=np.eye(3), coord='global')
    
    Args:
      body1, body2 (Body or Ground obj): Connected bodies
      point1, point2 (Point obj): Points that must coincide along constaxes
      constaxes (list of 3-lists or arrays, optional): Axes along which no 
        translation takes place
      coord ('global' or 'local1'): coordinate system of constaxes
    
    Note that body1 and body2 currently override the points' owner attributes. This
    increases readability when defining constraints.
    
    Example:
      Translational(b1, p1, b2, p2, constaxes=[[0,1,0], [0,0,1], coord='local1']
          Allows translation between points p1 and p2 of bodies b1 and b2 only 
          along the axis [1,0,0] as defined in the coordinate system of b1.
      Translational(ground, gp, b, p, constaxes[[0,-1/sqrt(2),1/sqrt(2)], 
        coord='global')
          Allows translation of point p in body b only along the diagonal [1,0,1]
          in the x-z plane, and free translation in the y-plane
    
    """
    super().__init__(body1, point1, body2, point2)
    self.constaxes = np.array(constaxes).T
    self.coord = coord
    
  def get_J(self):
    '''J, Fc = get_J()
    Constraint Jacobian and constraint force.
    
    Returns:
      J (array nx12): Jacobian of the constraint equations C. To be added to the mass
        matrix if a Lagrange multiplier scheme is used for simulation.
      Fc (array nx1): Constraint force vector. A number of terms that appear when 
        differentiating C and that must be added to the load vector.
    
    '''
    u1, u2 = self.point1.u, self.point2.u
    r1, r2 = self.body1.r, self.body2.r
    v1, v2 = self.body1.v, self.body2.v
    p1, p2 = self.body1.p, self.body2.p
    o1, o2 = self.body1.omega, self.body2.omega
    if self.coord == 'global':
      rotate_all = np.eye(3)
      d = np.array([[0,0,0]]).T # set distance to zero, since these terms will anyway disappear if the coordsys is global
      ddot = np.array([[0,0,0]]).T
    elif self.coord == 'local1':
      rotate_all = R(p1)
      d = -R(p1).T.dot(r1+R(p1).dot(u1)-r2-R(p2).dot(u2)) # distance between points
      ddot = -R(p1).T.dot(v1+R(p1).dot(skew(o1).dot(u1)) -v2-R(p2).dot(skew(o2).dot(u2))) + skew(o1).dot(d)
    elif self.coord == 'local2':
      rotate_all = R(p2)
      print('Spherical: local2 coords currently not supported, please change order of the arguments')
    self.ATRT = self.constaxes.T.dot(rotate_all.T)
    equations = np.shape(self.constaxes)[1]
    J = np.zeros([equations,12])
    Fc = np.zeros([equations,1])
    J[:, 0:3] = -self.ATRT
    J[:, 3:6] = -self.ATRT.dot(R(p1)).dot(skew(u1).T+skew(d).T)
    Fc += self.ATRT.dot(R(p1)).dot(doublecross(o1).dot(u1) + doublecross(o1).dot(d) + 2*skew(o1).dot(ddot))
    J[:, 6:9] = self.ATRT
    J[:, 9:12] = self.ATRT.dot(R(p2)).dot(skew(u2).T)
    Fc += -self.ATRT.dot(R(p2)).dot(doublecross(o2)).dot(u2)
    return J, Fc
    
  def get_C(self):
    '''C = get_C()
    Value of the constraint function (only for positioning or error estimation)
    
    Returns:
      C (array nx1): The distance between the two points that are supposed to coincide.'''
    r1, r2 = self.body1.r, self.body2.r
    p1, p2 = self.body1.p, self.body2.p
    u1, u2 = self.point1.u, self.point2.u
    equations = 3 # function not yet supported
    C = np.zeros((equations,1))
    if not isinstance(self.body1, Ground):
      C += -r1 - R(p1).dot(u1)
    if not isinstance(self.body2, Ground):
      C +=  r2 + R(p2).dot(u2)
    return C

class Slider(Constraint):
  '''Locks rotations between two bodies along the 1, 2, or 3 axes defined in constaxes.''' # TODO: Remove point arguments?
  def __init__(self, body1, point1, body2, point2, constaxes):
    """Slider(body1, point1, body2, point2, constaxes)
    
    Args:
      point1, point2 (Point obj): Required but currently not used (Rotation constraint is independent of location)
      body1, body2 (Body or Ground obj): Connected bodies.
      constaxes (list of normalised 3-lists): The locked axes of rotation. These axes are defined 
        in the coord system of body1.
    
    Examples:
      Slider(body1, point1, body2, point2, [[0,1,0], [0,0,1]])
        Locks two rotations, but allows rotation between body1 and body2 along the 
        x axis of body1's coord system. 
      Slider(body1, point1, body2, point2, numpy.eye(3))
        Locks all rotations.
    
    """
    super().__init__(body1, point1, body2, point2)
    self.constaxes = np.array(constaxes).T
    
  def get_J(self):
    '''J, Fc = get_J()
    Constraint Jacobian and constraint force. n is the number of constant axes.
    
    Returns:
      J (array nx12): Jacobian of the constraint equations C. To be added to the mass
        matrix if a Lagrange multiplier scheme is used for simulation.
      Fc (array nx1): Constraint force vector. A number of terms that appear when 
        differentiating C and that must be added to the load vector.
    
    '''
    p1, p2 = self.body1.p, self.body2.p
    o1, o2 = self.body1.omega, self.body2.omega
    m = self.constaxes # one to three unit vectors perpendicular to the free axes
    J = np.zeros([m.shape[1],12])
    Fc = np.zeros([m.shape[1],1])
    J[:, 3:6] = -m.T
    J[:, 9:12] = m.T.dot(R(p1).T).dot(R(p2))
    Fc += m.T.dot(skew(o1)).dot(R(p1).T).dot(R(p2)).dot(o2)
    return J, Fc
    
  def get_C(self):
    '''C = get_C()
    Value of the constraint function (only for positioning or error estimation)
    n is the number of constant axes.
    
    Returns:
      C (array nx1): Currently returns zeros, since constraints on angular velocity are 
        fulfilled for all translations. (To be extended if "velocity adjustment" functions
        are developed)
    
    '''
    m = self.constaxes
    C = np.zeros((m.shape[1],1))
    return C
    
class Fixed(Constraint):
  '''Combination of a Spherical and 3-axes Slider constraint. Locks all dofs.'''
  def __init__(self, body1, point1, body2, point2):
    """Fixed(body1, point1, body2, point2)
    
    Args:
      body1, body2 (Body or Ground obj): Connected bodies.
      point1, point2 (Point obj): Points that must coincide
    
    Note that body1 and body2 currently override the points' owner attributes. This
    increases readability when defining constraints.
    
    """
    super().__init__(body1, point1, body2, point2)
    self.constaxes = np.eye(3)
    self.coord = 'global'
    
  def get_J(self):
    '''J, Fc = get_J()
    Constraint Jacobian and constraint force.
    
    Returns:
      J (array 6x12): Jacobian of the constraint equations C. To be added to the mass
        matrix if a Lagrange multiplier scheme is used for simulation.
      Fc (array 6x1): Constraint force vector. A number of terms that appear when 
        differentiating C and that must be added to the load vector.
    
    '''
    J = np.zeros([6, 12])
    Fc = np.zeros([6, 1])
    Jtemp, Fctemp = Spherical.get_J(self)
    J[0:3, :] = Jtemp
    Fc[0:3] = Fctemp
    Jtemp, Fctemp = Slider.get_J(self)
    J[3:6, :] = Jtemp
    Fc[3:6] = Fctemp
    return J, Fc
    
  def get_C(self):
    '''C = get_C()
    Value of the constraint function (only for positioning or error estimation)
    
    Returns:
      C (array 6x1): See Spherical.get_C and Slider.get_C'''
    C = np.zeros((6,1))
    C[0:3] = Spherical.get_C(self)
    C[3:6] = Slider.get_C(self)
    return C
