import MBS
from MBS import constraints as c
from MBS import loads as l
from MBS import positioning
from MBS import simulation as sim
from MBS import visualization as v
import numpy as np
import matplotlib.pyplot as plt
import os
pi = np.pi

def roller_setup(g, freq, roller_v, fixed):
  massfactor = 1.0
  rammass = massfactor*3159.0
  drivmass = massfactor*220.4
  vibbmass = massfactor*108.2
  valsmass = massfactor*3765.8
  axlemass = massfactor*88.074
  axleexc = massfactor*7.31/axlemass
  discstiffnesstangential = 200000
  vibbstiffnesstangential = 250000
  discstiffnessnormal = 1400000
  vibbstiffnessnormal = 1750000
  damping = 310000*0.5*0.1
  if fixed:
    damping = 0.
  amplitude = axlemass*axleexc*(2*pi*freq)**2/(discstiffnesstangential*8+vibbstiffnesstangential*6-(valsmass+vibbmass)*(2*pi*freq)**2)
  print('m', axlemass)
  print('M', valsmass+vibbmass)
  print('r_o', axleexc)
  print('k', discstiffnesstangential*8+vibbstiffnesstangential*6)
  print('omega', 2*np.pi*freq)
  print('A', amplitude)

  roller_omega = roller_v/0.759
  
  # Global coords taken from CAD
  CAD = {} # mm, y-z-x
  CAD['rammc'] = np.array([[-18, 49, -69]])
  CAD['drivmc'] = np.array([[782, 0, 0]])
  CAD['vibbmc'] = np.array([[-649, -13, 7]])
  CAD['valsmc'] = np.array([[16, 0, 0]])
  CAD['excenter'] = np.array([[0, -83, 0]])
  CAD['ramdrivlager'] = np.array([[776,0,0]])
  CAD['valsvibblager'] = np.array([[-606,0,0]])
  CAD['exclagerleft'] = np.array([[-305,0,0]])
  CAD['exclagerright'] = np.array([[305,0,0]])
  CAD['gummivibb'] = np.array([[-676, 164, 267], [-676, -22, 260], [-676, -208, 254]])
  CAD['gummivibb'] = np.vstack((CAD['gummivibb'], CAD['gummivibb']))
  CAD['gummivibb'][3:, 1] = -CAD['gummivibb'][3:, 1]
  CAD['gummidriv'] = np.array([[747, 299, 480], [747, 459, 330]])
  CAD['gummidriv'] = np.vstack((CAD['gummidriv'], CAD['gummidriv'], CAD['gummidriv'], CAD['gummidriv']))
  CAD['gummidriv'][2:4, 2] = -CAD['gummidriv'][2:4, 2]
  CAD['gummidriv'][4:6, 1] = -CAD['gummidriv'][4:6, 1]
  CAD['gummidriv'][6:8, 1] = -CAD['gummidriv'][6:8, 1]
  CAD['gummidriv'][6:8, 2] = -CAD['gummidriv'][6:8, 2]
  for p in CAD:
    CAD[p][:,[0, 2]] = CAD[p][:,[2, 0]] # switch y <-> x indices to match our coord syst
    CAD[p][:,[1, 2]] = CAD[p][:,[2, 1]] # switch z <-> y indices to match our coord syst
    CAD[p] = np.round(CAD[p]*0.001,3)   # convert to meters
  
  Inertia = {} # kgmm^3
  Inertia['ram'] = [2.955e9, -4.1417e6, 1.508e7, 4.988e9, -2.683e8, 2.3146e9]
  Inertia['driv'] = [3.74e7, 0, 0, 1.93e7, 0, 1.93e7]
  Inertia['vibb'] = [6.33e6, 9.75e4, 8.40e5, 6.47e6, -1.08e5, 4.02e6]
  Inertia['vals'] = [1.709e9, 0, 0, 2.074e9, 0, 2.074e9]
  Inertia['excenter'] = [6.226e5, 1.678e3, -1.709e4, 1.481e7, -9.883e3, 1.497e7]
  for I in Inertia:
    Inertia[I] = [massfactor*II*1e-6 for II in Inertia[I]]
    Inertia[I] = np.round(np.array([[Inertia[I][5], Inertia[I][2], Inertia[I][4]], [Inertia[I][2], Inertia[I][0], Inertia[I][1]], [Inertia[I][4], Inertia[I][1], Inertia[I][3]]]), 2)
  # Bodies
  bodies = []
  axel = MBS.Body(axlemass, Inertia['excenter'], omega=(0,2*np.pi*freq,0), v=(-1.025*(valsmass+vibbmass)*axleexc*2*np.pi*freq/(valsmass+vibbmass+axlemass),0,0))
  vals = MBS.Body(valsmass, Inertia['vals'], v=(1.025*axlemass*axleexc*2*np.pi*freq/(valsmass+vibbmass+axlemass),0,0))
  drivskiva = MBS.Body(drivmass, Inertia['driv'], omega=vals.omega)
  vibbplat = MBS.Body(vibbmass, Inertia['vibb'], v=vals.v)
  ram = MBS.Body(rammass, Inertia['ram'])
  bodies = [axel, vals, drivskiva, vibbplat, ram]
  for body in bodies:
    body.force = np.array([[0,0,-g*body.m]]).T
  
  ground = MBS.Ground()
  
  # Points
  points = []
  # for constraints
  axelvalsp_l = MBS.Point(CAD['exclagerleft']-CAD['excenter'], axel)
  axelvalsp_r = MBS.Point(CAD['exclagerright']-CAD['excenter'], axel)
  valsaxelp_l = MBS.Point(CAD['exclagerleft']-CAD['valsmc'], vals)
  valsaxelp_r = MBS.Point(CAD['exclagerright']-CAD['valsmc'], vals)
  valsdrivp = MBS.Point(CAD['ramdrivlager']-CAD['valsmc'], vals)
  valsvibbp = MBS.Point(CAD['valsvibblager']-CAD['valsmc'], vals)
  drivramp = MBS.Point(CAD['ramdrivlager']-CAD['drivmc'], drivskiva)
  drivvalsp = MBS.Point(CAD['ramdrivlager']-CAD['drivmc'], drivskiva)
  vibbvalsp = MBS.Point(CAD['valsvibblager']-CAD['vibbmc'], vibbplat)
  vibbcp = MBS.Point([0.,0.,0.], vibbplat)
  drivcp = MBS.Point([0.,0.,0.], drivskiva)
  axelcp = MBS.Point([0.,0.,0.], axel)
  valscp = MBS.Point([0.,0.,0.], vals)
  ramcp = MBS.Point([0.,0.,0.], ram)
  ramdrivp = MBS.Point(CAD['ramdrivlager']-CAD['rammc'], ram)
  ramvibbp = MBS.Point(CAD['vibbmc']-CAD['rammc'], ram)
  gp = MBS.Point([0.,0.,0.], ground)
  groundramp = MBS.Point(CAD['rammc'], ground)
  
  springdrivvalsp = []
  springvalsdrivp = []
  [springdrivvalsp.append(MBS.Point(gummip-CAD['drivmc'], drivskiva)) for gummip in CAD['gummidriv']]
  [springvalsdrivp.append(MBS.Point(gummip-CAD['valsmc'], vals)) for gummip in CAD['gummidriv']]
  
  springvibbramp = []
  springramvibbp = []
  [springvibbramp.append(MBS.Point(gummip-CAD['vibbmc'], vibbplat)) for gummip in CAD['gummivibb']]
  [springramvibbp.append(MBS.Point(gummip-CAD['rammc'], ram)) for gummip in CAD['gummivibb']]
  
  # for visualisation
  MBS.Point([0.,0.,1.], drivskiva)
  points.append(MBS.Point([1.,0.,0.], drivskiva))
  points.append(MBS.Point([-1.,0.,0.], drivskiva))
  points.append(MBS.Point([0.,0.,-1.], drivskiva))
  points.append(MBS.Point([0.,-1.,1.], vals))
  points.append(MBS.Point([0.83,-1.,-0.5], vals))
  points.append(MBS.Point([-0.83,-1.,-0.5], vals))
  points.append(MBS.Point([0.,1.,1.], vals))
  points.append(MBS.Point([0.83,1.,-0.5], vals))
  points.append(MBS.Point([-0.83,1.,-0.5], vals))
  points.append(MBS.Point([0.,0.,1.], vibbplat))
  points.append(MBS.Point([0.83,0.,-0.5], vibbplat))
  points.append(MBS.Point([-0.83,0.,-0.5], vibbplat))
  points.append(MBS.Point([1.2,1.2,0.], ram))
  points.append(MBS.Point([-1.2,1.2,0.], ram))
  points.append(MBS.Point([1.2,-1.2,0.], ram))
  points.append(MBS.Point([-1.2,-1.2,0.], ram))
  
  # Constraints
  # for simulation
  constraints = []
  positioningconstraints = []
  constraints.append(c.Spherical(axel, axelvalsp_l, vals, valsaxelp_l))
  constraints.append(c.Slider(axel, axelvalsp_r, vals, valsaxelp_r, [[1,0,0], [0,0,1]]))
  if fixed:
    constraints.append(c.Fixed(ram, ramcp, ground, groundramp))
  else:
    # constraints.append(c.Slider(ram, ramcp, ground, groundramp, [[1,0,0], [0,1,0], [0,0,1]]))
    constraints.append(c.Slider(ram, ramcp, ground, groundramp, [[0,1,0], [0,0,1]]))
  constraints.append(c.Spherical(ram, ramdrivp, drivskiva, drivramp))
  constraints.append(c.Slider(ram, ramdrivp, drivskiva, drivramp, [[1,0,0], [0,0,1]]))
  constraints.append(c.Spherical(vals, valsvibbp, vibbplat, vibbvalsp))
  constraints.append(c.Slider(vals, valsvibbp, vibbplat, vibbvalsp, [[1,0,0], [0,0,1]]))
  
  # for initial positioning (these are removed later)
  if not fixed:
    positioningconstraints.append(c.Spherical(ram, ramcp, ground, groundramp))
  positioningconstraints.append(c.Spherical(vals, valsdrivp, drivskiva, drivvalsp))
  
  # Loads
  loads = []
  [loads.append(l.MultiaxialSpring(drivskiva, dv, vals, vd, discstiffnessnormal, discstiffnesstangential, [0,1,0])) for dv, vd in zip(springdrivvalsp, springvalsdrivp)]
  [loads.append(l.MultiaxialSpring(ram, rv, vibbplat, vr, vibbstiffnessnormal, vibbstiffnesstangential, [0,1,0])) for rv, vr in zip(springramvibbp, springvibbramp)]
  loads.append(l.Damper(vals, valsdrivp, drivskiva, drivvalsp, damping))
  loads.append(l.Damper(ram, ramvibbp, vibbplat, vibbcp, damping))
  gravelload = l.PointLoad(vals, valscp, ground, gp, [0.0, 0.0, 0.0])
  graveltorque = l.Torque(vals, ground, [0.0, 0.0, 0.0], 'global')
  drivetorque = l.Torque(ram, drivskiva, [0.0, 0.0, 0.0], 'local1')
  driveforce = l.PointLoad(ram, ramcp, ground, gp, [0.0, 0.0, 0.0])
  axletorque = l.Torque(vibbplat, axel, [0.0, 0.0, 0.0], 'local1')
  loads += [gravelload, graveltorque, drivetorque, driveforce, axletorque]
  
  s, qdot = positioning.get_state(bodies)
  positioning.adjust_to_constraints(s, bodies, constraints, positioningconstraints)
  
  axel.r[2] -= amplitude
  vals.r[2] -= amplitude
  vibbplat.r[2] -= amplitude
  '''adjust roller by an approximate initial displacement opposite to the excenter'''
  
  qdot[9:12] = np.array([0, roller_omega, 0]).reshape((3,1))
  qdot[6*2+3:6*2+6] = np.array([0, roller_omega, 0]).reshape((3,1))
  '''Rotation of roller and driving disc'''
  
  qdot[0::6] += roller_v
  
  ram.r[2] -= 2*(rammass+drivmass)*g/(discstiffnesstangential*8+vibbstiffnesstangential*6)
  drivskiva.r[2] -= 2*(rammass+drivmass)*g/(discstiffnesstangential*8+vibbstiffnesstangential*6)
  '''Adjust the frame position to account for suspension'''
  
  goal_drum_rotation = np.array([[0.,roller_omega,0.]]).T
  Kdrum = -3e4
  Tidrum = np.inf
  Tddrum = 0
  drumregulator = MBS.Regulator(goal_drum_rotation, drivskiva.omega, Kdrum, Tidrum, Tddrum, regulated_load=drivetorque)

  goal_axle_rotation = np.array([[0.,2*pi*freq,0.]]).T
  Kaxle = -70
  Tiaxle = np.inf
  Tdaxle = 0.
  axleregulator = MBS.Regulator(goal_axle_rotation, axel.omega, Kaxle, Tiaxle, Tdaxle, regulated_load=axletorque)
  regulators = [drumregulator, axleregulator]
  
  return bodies, constraints, loads, s, qdot, vals, axel, ram, gravelload, graveltorque, drivetorque, driveforce, regulators

if __name__ == "__main__":
  dt = 1e-3
  seconds = 1.0
  steps = int(seconds/dt)
  g = 9.81*0
  freq = 29
  roller_v = 1.0*0
  ssaved = []
  fixed = True
  
  bodies, constraints, loads, s, qdot, vals, axel, ram, gravelload, graveltorque, drivetorque, driveforce, regulators = roller_setup(
        g, freq, roller_v, fixed)
  
  gravelload.fvector = np.array([[0.,0.,7341.474]]).T*g
  ssaved, _ = sim.simulate(dt, int(steps), s, qdot, bodies, constraints, loads, regulators, gamma=0.25, beta=0.25, correction=True)
  
  v.animate(ssaved, bodies, pt=8, zoom=0.5, rotspeed=0.1, n=[0.0,0.0,1.0], show_on_screen=True, savepics=False, directory='vidframes')
  
