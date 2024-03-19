import MBS
from MBS import constraints as c
from MBS import loads as l
from MBS import positioning
from MBS import simulation as sim
from MBS import visualization as v
import numpy as np

dt = 1e-3
seconds = 4.0
steps = int(seconds/dt)

upper = MBS.Body(1., np.eye(3), force=[0.,0.,-9.8*1.])
mid = MBS.Body(1., np.eye(3), force=[0.,0.,-9.8*1.])
lower = MBS.Body(1., np.eye(3), force=[0.,0.,-9.8*1.])
bodies = [upper, mid, lower]
ground = MBS.Ground()

gu = MBS.Point([0,0,0], ground)
ug = MBS.Point([0,0.5,0], upper)
um = MBS.Point([0,-0.5,0], upper)
mu = MBS.Point([0.5,0,0], mid)
ml = MBS.Point([-0.5,0,0], mid)
lm = MBS.Point([0,0,-0.5], lower)
lp = MBS.Point([0,0,0.5], lower)
points = [ug, um, mu, ml, lm]

constraints = []
constraints.append(c.Spherical(upper, ug, ground, gu))
constraints.append(c.Spherical(mid, mu, upper, um))
constraints.append(c.Spherical(lower, lm, mid, ml))

loads = []

s, qdot = positioning.get_state(bodies)
positioning.adjust_to_constraints(s, bodies, constraints)

ssaved, qdotsaved = sim.simulate(dt, int(steps), s, qdot, bodies=bodies, constraints=constraints, loads=loads)

v.animate(ssaved, bodies, pt=30, zoom=0.25, rotspeed=0., n=[0.0,0.0,1.0], show_on_screen=True, savepics=False, directory='vidframes')
