import numpy as np

norm = np.linalg.norm
sin = np.sin
cos = np.cos

def Gbar(theta):
  '''omegabar = Gbar(p).dot(pdot) yields omega in local coords.

  Gbar corresponds to 2G in Graf (2008), 2Ebar in Shabana (2005).
  
  Args:
    theta (array 4x1): Quaternion
  
  Returns:
    Gbar (array 3x4): Transformation matrix
  
  '''
  th = [t[0,] for t in theta]
  return 2*np.array([[-th[1], th[0], th[3], -th[2]], [-th[2], -th[3], th[0], th[1]], [-th[3], th[2], -th[1], th[0]]])

def G(theta):
  '''omega = G(p).dot(pdot) yields omega in global coords.
  
  G corresponds to 2E in Graf (2008), 2E in Shabana (2005)
  
  Args:
    theta (array 4x1): Quaternion
  
  Returns:
    G (array 3x4): Transformation matrix
  
  '''
  th = [t[0,] for t in theta]
  return 2*np.array([[-th[1], th[0], -th[3], th[2]], [-th[2], th[3], th[0], -th[1]], [-th[3], -th[2], th[1], th[0]]])

def R(theta):
  '''Returns rotation matrix.
  
  u = R(p).dot(ubar) yields a local vector u in global coords.
  
  Args:
    theta (array 4x1): Quaternion
  
  Returns:
    R (array 3x3): Rotation matrix
  
  '''
  return 0.25*G(theta).dot(Gbar(theta).T)

def get_p(n, phi):
  '''Returns a quaternion corresponding to a vector rotation.
  
  p = [cos(phi/2), sin(phi/2)*nx, sin(phi/2)*ny, sin(phi/2)*nz]
  
  Args:
    n (3-list): Vector to rotate about (normalised)
    phi (float): Rotation in radians
  
  Returns:
    p (array 4x1): Quaternion that corresponds to n-phi rotation.
  
  '''
  n = np.reshape(n, (3,1))
  n = n/norm(n)
  p = np.zeros((4,1))
  p[0] = cos(0.5*phi)
  p[1:4] = sin(0.5*phi)*n
  return p

def skew(rr):
  '''Skew-symmetric tensor, skew(u).dot(v) = cross(u,v)
  
  Args:
    rr (array 3x1): vector to transform
  
  Returns:
    rrskew (array 3x3): skew-symmetric matrix
  
  '''
  r = [t[0,] for t in rr]
  return np.array([[0., -r[2], r[1]],[r[2], 0., -r[0]],[-r[1], r[0], 0.]])
  
def doublecross(x):
  '''Double skew tensor, doublecross(u).dot(v) = cross(u,cross(u,v))
  
  Args:
    x (array 3x1): vector to transform
  
  Returns:
    rrdouble (array 3x3): matrix representation
    
  '''
  return np.array([[-x[1,0]**2-x[2,0]**2, x[0,0]*x[1,0], x[0,0]*x[2,0]], [x[1,0]*x[0,0], -x[2,0]**2-x[0,0]**2, x[1,0]*x[2,0]], [x[2,0]*x[0,0], x[2,0]*x[1,0], -x[0,0]**2-x[1,0]**2]])

def omega_to_pdot(p):
  '''Returns the transformation matrix Q = Gbar.T/4
  
  Args:
    p (array 4x1): Quaternion
  
  Returns:
    Q (array 4x3): Transformation matrix from omega to pdot
  
  Examples:
    pdot = omega_to_pdot(p).dot(omega)
    
  '''
  return 0.25*Gbar(p).T

def qdot_to_sdot(p):
  '''Returns the transformation matrix S = [1 0;0 Q]
  
  Args:
    p (array 4x1): Quaternion
  
  Returns:
    S (array 7x6): Transformation matrix from q to s
  
  Examples:
    sdot = qdot_to_sdot(s[3:7]).dot(qdot)
    s = qdot_to_sdot(p).dot(q)
    
  '''
  S = np.zeros((7,6))
  S[0:3,0:3] = np.eye(3)
  S[3:7,3:7] = omega_to_pdot(p)
  return S
  
def R_x(phi):
  '''Returns rotation matrix about x axis.
  
  Args:
    phi (float): euler angle about x axis in radians
  
  Returns:
    R (array 3x3): Rotation matrix
  
  '''
  cp = cos(phi)
  sp = sin(phi)
  R = np.array([[1., 0., 0. ],
                [0., cp, -sp],
                [0., sp, cp]])
  return(R)
  
def R_y(theta):
  '''Returns rotation matrix about y axis.
  
  Args:
    theta (float): euler angle about y axis in radians
  
  Returns:
    R (array 3x3): Rotation matrix
  
  '''
  cp = cos(theta)
  sp = sin(theta)
  R = np.array([[cp, 0., sp],
                [0., 1., 0. ],
                [-sp, 0., cp]])
  return(R)
  
def R_z(psi):
  '''Returns rotation matrix about z axis.
  
  Args:
    psi (float): euler angle about z axis in radians
  
  Returns:
    R (array 3x3): Rotation matrix
  
  '''
  cp = cos(psi)
  sp = sin(psi)
  R = np.array([[cp, -sp, 0.],
                [sp, cp, 0.],
                [0., 0., 1.]])
  return(R)
  
  
def R_rpy(rpy):
  '''Returns rotation matrix.
  
  u = R(rpy).dot(ubar) yields a local vector u in global coords.
  
  Args:
    rpy (array 3x1): euler angles in the order z-y-x in radians
  
  Returns:
    R (array 3x3): Rotation matrix
  
  '''
  phi = rpy[2] # x axis
  theta = rpy[1] # y axis
  psi = rpy[0] # z axis
  R = R_z(psi).dot(R_y(theta)).dot(R_x(phi))
  return R

