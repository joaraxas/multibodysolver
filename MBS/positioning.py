import numpy as np
from MBS import Ground
from MBS.transformations import omega_to_pdot
from MBS.simulation import assemble_constraints, stack_symmetric

def get_state(bodies):
  '''Initialises state 7m x 1 vector s and the 6m x 1 velocity qdot.
  
  s and qdot are set as references to r, p, v and omega in bodies. They 
  point to the same values.
  
  Args:
    bodies (m-list of Body obj)
  
  Returns:
    s (array 7m x 1): state vector s.T = [r1.T, p1.T, ..., rm.T, pm.T].T
    qdot (array 6m x 1): velocity vector.
      qdot.T = [v1.T, omega1.T, ..., vm.T, omegam.T].T
  
  '''
  nbods = len(bodies)
  nsdof = 7*nbods
  ndof = 6*nbods
  s = np.zeros((nsdof,1))
  qdot = np.zeros((ndof,1))
  for iBody in range(nbods):
    rw = 6*iBody
    rs = 7*iBody
    s[rs:rs+3] = bodies[iBody].r
    s[rs+3:rs+7] = bodies[iBody].p
    qdot[rw:rw+3] = bodies[iBody].v
    qdot[rw+3:rw+6] = bodies[iBody].omega
    bodies[iBody].r = s[rs:rs+3]
    bodies[iBody].p = s[rs+3:rs+7]
    bodies[iBody].v = qdot[rw:rw+3]
    bodies[iBody].omega = qdot[rw+3:rw+6]
  return s, qdot


def adjust_to_constraints(s, bodies, const, posconst=[]):
  '''Adjusts the translations in s to fulfill the constraints in const and posconst.
  
  posconst is a list of constraints that are used for positioning but not 
  in the simulation. The translations of all bodies are adjusted so that all
  constraints are satisfied. Note that the system MUST be fully and sufficiently
  constrained, i.e. in this function, NO translations may be unconstrained.
  The rotational dofs are not touched. Function is slightly experimental.
  
  Args:
    s (array 7m x 1): state vector s.T = [r1.T, p1.T, ..., rm.T, pm.T].T
    bodies (m-list of Body obj)
    const (m-list of Constraint obj)
    posconst (m-list of Constraint obj, optional): Has the same effect as
      const, but you may want to keep them separate
  
  Returns:
    No returns but s is altered.
  
  '''
  # TODO: maybe better to add cons and poscons together in this function
  constraints = const + posconst
  nbods = len(bodies)
  nsdof = 7*nbods
  ndof = 6*nbods #TODO: Also set v based on omega?
  
  dof = list(range(ndof))
  sdof = list(range(nsdof))
  fdof = dof[0::6] + dof[1::6] + dof[2::6] # the "free" dofs in 6-system (that are to be adjusted to constraints)
  pdof = dof[3::6] + dof[4::6] + dof[5::6] # the "prescribed" dofs in 6-system (that are not to be changed, i.e. rotations)
  fsdof = sdof[0::7] + sdof[1::7] + sdof[2::7] # the "free" dofs in 7-system
  fdof.sort(), pdof.sort(), fsdof.sort()
  g = np.zeros((ndof, 1)) # the rhs of the positioning system (~biases, shifts of dofs)
  nc = 0
  for cons in constraints:
    # Find the number of constraint eqs. and align all positions according to any ground constraint
    # TODO: Multiple ground constraints?
    nc += cons.get_C().shape[0]
    if isinstance(cons.body1, Ground):
      g[fdof[0::3]] = cons.point1.u[0]
      g[fdof[1::3]] = cons.point1.u[1]
      g[fdof[2::3]] = cons.point1.u[2]
    if isinstance(cons.body2, Ground):
      g[fdof[0::3]] = cons.point2.u[0]
      g[fdof[1::3]] = cons.point2.u[1]
      g[fdof[2::3]] = cons.point2.u[2]
  
  g = np.vstack((g, np.zeros((nc, 1))))
  
  for iBody in range(nbods):
    # set the rotation biases to the prescribed values but in the 6-system
    rw = 6*iBody
    rs = 7*iBody
    Q = omega_to_pdot(s[rs+3:rs+7])
    g[rw+3:rw+6] = Q.T.dot(s[rs+3:rs+7])
  f = g.copy()
  
  # Assemble Jacobian and solve for free coordinates f that satisfy constraints
  # (could possibly be iterated to solve for arbitrary rotations, but not prioritised)
  
  J, _, C = assemble_constraints(bodies, constraints)
  f[ndof:] = C # f measures the unfulfillment of constraints, we want f=g
  Jac = np.eye(ndof)
  Jac[fdof,fdof] = 0.0 # Zeros in the Jacobian correspond to unconstrained dofs
  
  Jac = stack_symmetric(Jac, J, 0.0)
  
  df = np.linalg.solve(Jac, g-f)
  df[pdof] = 0.0
  f += df
  s[fsdof] = f[fdof]
