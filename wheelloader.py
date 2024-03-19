import MBS
from MBS import constraints as c
from MBS import loads as l
from MBS import positioning
from MBS import simulation as sim
from MBS import visualization as v
import numpy as np

dt = 1e-3
seconds = 8.0
steps = int(seconds/dt)

piston_upper = MBS.Body(1., np.eye(3), force=[0.,0.,-9.8*1.], r=[-1,0,1])
piston_lower = MBS.Body(1., np.eye(3), force=[0.,0.,-9.8*1.], r=[-1,0,-1], torque=[0,-5000.,0])
bucket = MBS.Body(300., np.eye(3)*300., force=[0.,0.,-9.8*300.], r=[1,0,0])
bodies = [piston_upper, piston_lower, bucket]
ground = MBS.Ground()

uplp = MBS.Point([-1,0,0], piston_upper)
lplp = MBS.Point([-1,0,0], piston_lower)
uprp = MBS.Point([1,0,0], piston_upper)
lprp = MBS.Point([1,0,0], piston_lower)
blp = MBS.Point([-1,0,-1], bucket)
bup = MBS.Point([-1,0,1], bucket)
glp = MBS.Point([-2,0,-1], ground)
gup = MBS.Point([-2,0,1], ground)
points = [uplp, lplp, uprp, lprp, bup, blp, glp, gup]
points.append(MBS.Point([-1.5,-2.0,1.5], bucket))
points.append(MBS.Point([-1.5,-2.0,-2.0], bucket))
points.append(MBS.Point([1.0,-2.0,1.0], bucket))
points.append(MBS.Point([-1.5,2.0,1.5], bucket))
points.append(MBS.Point([-1.5,2.0,-2.0], bucket))
points.append(MBS.Point([1.0,2.0,1.0], bucket))
points.append(MBS.Point([2.5,0,0], piston_upper))

constraints = []
constraints.append(c.Spherical(piston_lower, lplp, ground, glp))
constraints.append(c.Spherical(piston_lower, lprp, bucket, blp))
constraints.append(c.Spherical(piston_upper, uplp, ground, gup))
constraints.append(c.Translational(piston_upper, uprp, bucket, bup, constaxes=[[0,1,0],[0,0,1]], coord='local1'))

loads = []
loads.append(l.Damper(piston_upper, uprp, bucket, bup, 7e1))
loads.append(l.Spring(piston_upper, uprp, bucket, bup, 3e3))

s, qdot = positioning.get_state(bodies)

ssaved, qdotsaved = sim.simulate(dt, int(steps), s, qdot, bodies=bodies, constraints=constraints, loads=loads)

v.animate(ssaved, bodies, pt=30, zoom=0.25, rotspeed=0., n=[0.0,0.0,1.0], show_on_screen=True, savepics=False, directory='vidframes')
