import numpy as np
from MBS import Ground
from MBS.transformations import skew, qdot_to_sdot
norm = np.linalg.norm

def assemble_constraints(bodies, constraints):
  """Iterates over constraints and assembles constraint tensors
    
    Args:
      bodies (m-list of Body obj)
      constraints (n-list of Constraint obj, locking nc dofs)
    
    Returns:
      J (array nc x m): Constraint Jacobian
      Fc (array nc x 1): Constraint resulting forces e.g. centripetal
      C (array nc x 1): Constraint fulfillment vector
    
    See also constraints module get_C, get_J
      
    """
  ndof = len(bodies)*6
  J = np.zeros((0, ndof))
  Fc = np.zeros((0, 1))
  C = np.zeros((0, 1))
  for cons in constraints:
      Je, Fce = cons.get_J()
      Jeexp = np.zeros((Je.shape[0], ndof))
      if not isinstance(cons.body1, Ground):
        rw = bodies.index(cons.body1)*6
        Jeexp[:, rw:rw+6] = Je[:, 0:6]
      if not isinstance(cons.body2, Ground):
        rw = bodies.index(cons.body2)*6
        Jeexp[:, rw:rw+6] = Je[:, 6:12]
      J = np.vstack((J, Jeexp))
      Fc = np.vstack((Fc, Fce))
      Ce = cons.get_C()
      C = np.vstack((C, Ce))
  return J, Fc, C

def stack_symmetric(M, J, cornerdiag):
  """Stack matrices as [M, J.T]
                       [J, c*1]
    
    Args:
      M (array n x n): top-left matrix
      J (array n x m): side matrix
      cornerdiag (float): the right bottom corner is filled as np.eye*cornerdiag
    
    Returns:
      Ml (array (n+m) x (n+m)): stacked matrix
      
    """
  Ml1 = np.hstack((M, J.T))
  Ml2 = np.hstack((J, cornerdiag*np.eye(J.shape[0])))
  return np.vstack((Ml1, Ml2))

def get_mass_matrix(dt, s, bodies, constraints, loads, regulators):
  ndof = 6*len(bodies)
  nsdof = 7*len(bodies)
  M = np.zeros((ndof, ndof))
  Fext = np.zeros((ndof, 1))
  Fv = np.zeros((ndof, 1))
  J = np.zeros((0, ndof))
  Fc = np.zeros((0, 1))
  S = np.zeros((nsdof, ndof))
  
  for iBody in range(len(bodies)):
    rw = 6*iBody
    body = bodies[iBody]
    M[rw:rw+3, rw:rw+3] = np.eye(3)*body.m
    M[rw+3:rw+6, rw+3:rw+6] = body.I
    rw = 6*iBody
    rs = 7*iBody
    s[rs+3:rs+7] = s[rs+3:rs+7] / norm(s[rs+3:rs+7]) # normalise quaternions
    body = bodies[iBody]
    Fext[rw:rw+3] = body.force
    Fext[rw+3:rw+6] = body.torque
    Fv[rw+3:rw+6] = -skew(body.omega).dot((body.I).dot(body.omega))
    S[rs:rs+7, rw:rw+6] = qdot_to_sdot(body.p) # conversion matrix from 6-system to 7-system
  
  J, Fc, _ = assemble_constraints(bodies, constraints)
  
  for regulator in regulators:
    regulator.set_signal(dt)

  for iForce in range(len(loads)):
    force = loads[iForce]
    Fe = force.get_F()
    if not isinstance(force.body1, Ground):
      rw = bodies.index(force.body1)*6
      Fext[rw:rw+6] += Fe[0:6]
    if not isinstance(force.body2, Ground):
      rw = bodies.index(force.body2)*6
      Fext[rw:rw+6] += Fe[6:12]
  
  Ml = stack_symmetric(M, J, 0.0)
  Fl = np.vstack((Fext+Fv, Fc))
  return Ml, Fl, S
    

def simulate(dt, steps, s, qdot, bodies=[], constraints=[], loads=[], regulators=[], log_level=1, gamma=0, beta=0.5, correction=False):
  '''Solves a dynamic system of m bodies with nc constraint eqs.
  
  Args:
    dt (float): Timestep [s]
    steps (int): Number of steps to simulate
    s (array 7m x 1): State vector of translations and rotations 
      (quaternion-formulated), as [[rx,ry,rz,q0,q1,q2,q3,...]].T
      Defines the initial state. Updated during simulation!
    qdot (array 6m x 1): Velocity vector of translational and rotational
      velocities (vector-formulated), as [[vx,vy,vz,wx,wy,wz,...]].T
      Defines the initial velocity. Updated during simulation!
    bodies (list of Body obj, optional): m Bodies to simulate.
    constraints (list of Constraint obj, optional): Constraints to add.
    loads (list of Load obj, optional): Loads to apply.
    regulators (list of Regulator obj, optional): Regulators to apply.
    log_level (int in {0,1}, optional): If 1, writes out time during simulation.
    gamma (float in [0,1], optional): Weight of the acceleration of last 
      timestep to use in the time-integration. 0.25 is a good choice if
      correction is applied.
    beta (float in (0,0.5], optional): Parameter for correction. 0.25 is 
      recommended if correction is applied.
    correction (Boolean, optional): Whether to apply correction according to 
      Newmark's method (for non-linear systems). This can reduce drifting.
  
  Returns:
    ssaved (list of array 7mx1): steps no. of entries of state arrays
    qdotsaved (list of array 6mx1): steps no. of entries of qdot arrays
  
  Examples:
    ssaved, qdotsaved = simulate(1e-4, 10000, s, qdot, bodies)
      Runs 1 second of simulation of "bodies", but with no loads or constraints.
      
  '''
  ssaved = []
  qdotsaved = []
  ndof = 6*len(bodies)
  _, Fc, _ = assemble_constraints(bodies, constraints)
  nc = np.shape(Fc)[0]
  tol = 1e-8
  ql = np.zeros((ndof+nc, 1))
  
  for step in range(steps):
    qlold = ql.copy()
    Ml, Fl, S = get_mass_matrix(dt, s, bodies, constraints, loads, regulators)
    ql = np.linalg.solve(Ml, Fl)
    qdot += dt*((1-gamma)*ql[:ndof] + gamma*qlold[:ndof])
    sdot = S.dot(qdot)
    sdotdot = S.dot(ql[:ndof])
    s += sdot*dt + (0.5-beta)*dt**2*sdotdot
    while correction:
      Ml, Fl, S = get_mass_matrix(dt, s, bodies, constraints, loads, regulators)
      r_error = Ml.dot(ql) - Fl
      qlerror = norm(r_error)
      if qlerror < tol:
        break
      correction_matrix = Ml/dt**2/beta
      deltaql = -np.linalg.solve(correction_matrix, r_error)
      deltaq = deltaql[:ndof]
      s += S.dot(deltaq)
      qdot += deltaq/dt*gamma/beta
      ql += deltaql/dt**2/beta
    
    ssaved.append(s.copy())
    qdotsaved.append(qdot.copy())
    for body in bodies:
      body.rsaved.append(body.r.copy())
      body.psaved.append(body.p.copy())
      body.vsaved.append(body.v.copy())
      body.omegasaved.append(body.omega.copy())
    if not step % 100 and log_level>0:
      print('simulating timestep',step,'/',steps)
  return ssaved, qdotsaved
  
