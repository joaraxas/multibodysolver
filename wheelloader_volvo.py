import MBS
from MBS import constraints as c
from MBS import loads as l
from MBS import positioning
from MBS import simulation as sim
from MBS import visualization as v
import numpy as np

dt = 1e-3
seconds = 6.0
steps = int(seconds/dt)

# Geometry extracted from CAD
# global coordinates of points
coord = {}
coord['t_arm'] = np.array([3.0,0.0,0.1])
coord['t_arm_bucket'] = np.array([3.15,0.0,0.05])
coord['boom'] = np.array([2.6,0.0,0.5])
coord['boom_hydcyl2'] = np.array([2.4,0.0,0.5])
coord['boom_frame'] = np.array([1.5,0.0,1.0])
coord['boom_bucket'] = np.array([3.3,0.0,-0.15])
coord['boom_bellcrank'] = np.array([2.95,0.0,0.6])
coord['frame'] = np.array([1.6,0.0,0.4])
coord['frame_hydcyl1'] = np.array([1.65,0.0,0.9])
coord['frame_hydcyl2'] = np.array([1.5,0.0,0.7])
coord['bucket'] = np.array([3.5,0.0,0.0])
coord['bellcrank'] = np.array([2.9,0.0,0.5])
coord['bellcrank_hydcyl1'] = np.array([3.1,0.0,1.0])
coord['bellcrank_t_arm'] = np.array([2.5,0.0,0.2])

coord['offset'] = np.array([-2.6,0.0,0.0])

t_arm = MBS.Body(m=8.0, I=1.*np.eye(3))
boom = MBS.Body(m=130.0, I=100.*np.eye(3))
frame = MBS.Body(m=650.0, I=500.*np.eye(3))
bucket = MBS.Body(m=170.0, I=80.*np.eye(3))
bellcrank = MBS.Body(m=60.0, I=20.*np.eye(3))
bodies = [t_arm, boom, frame, bucket, bellcrank]
for body in bodies:
  body.force = np.array([[0,0,-9.81*body.m]]).T
ground = MBS.Ground()

ground_p = MBS.Point(coord['frame'] + coord['offset'], ground)
t_arm_bellcrank_p = MBS.Point(coord['bellcrank_t_arm'] - coord['t_arm'], t_arm)
t_arm_bucket_p = MBS.Point(coord['t_arm_bucket'] - coord['t_arm'], t_arm)
boom_hydcyl2_p = MBS.Point(coord['boom_hydcyl2'] - coord['boom'], boom)
boom_frame_p = MBS.Point(coord['boom_frame'] - coord['boom'], boom)
boom_bucket_p = MBS.Point(coord['boom_bucket'] - coord['boom'], boom)
boom_bellcrank_p = MBS.Point(coord['boom_bellcrank'] - coord['boom'], boom)
frame_cp = MBS.Point([0,0,0], frame)
frame_hydcyl1_p = MBS.Point(coord['frame_hydcyl1'] - coord['frame'], frame)
frame_hydcyl2_p = MBS.Point(coord['frame_hydcyl2'] - coord['frame'], frame)
frame_boom_p = MBS.Point(coord['boom_frame'] - coord['frame'], frame)
bucket_boom_p = MBS.Point(coord['boom_bucket'] - coord['bucket'], bucket)
bucket_t_arm_p = MBS.Point(coord['t_arm_bucket'] - coord['bucket'], bucket)
bellcrank_hydcyl1_p = MBS.Point(coord['bellcrank_hydcyl1'] - coord['bellcrank'], bellcrank)
bellcrank_boom_p = MBS.Point(coord['boom_bellcrank'] - coord['bellcrank'], bellcrank)
bellcrank_t_arm_p = MBS.Point(coord['bellcrank_t_arm'] - coord['bellcrank'], bellcrank)

points = [t_arm_bellcrank_p, t_arm_bucket_p, boom_hydcyl2_p, boom_frame_p, boom_bucket_p, boom_bellcrank_p, frame_cp, frame_hydcyl1_p, frame_hydcyl2_p,frame_boom_p, bucket_boom_p, bucket_t_arm_p, bellcrank_hydcyl1_p, bellcrank_boom_p, bellcrank_t_arm_p]

# Extra points for visualizing body geometries
points.append(MBS.Point(np.array([2.2,0,0.25]) - coord['frame'], frame))
points.append(MBS.Point(np.array([1.1,0,0.4]) - coord['frame'], frame))
points.append(MBS.Point(np.array([4,0,0.4]) - coord['bucket'], bucket))
points.append(MBS.Point(np.array([3.2,0,0.7]) - coord['bucket'], bucket))
for p in points: # Add depth to bodies
  MBS.Point(p.u - np.array([[0],[1],[0]]), p.owner)

constraints = []
constraints.append(c.Spherical(t_arm, t_arm_bucket_p, bucket, bucket_t_arm_p))
constraints.append(c.Spherical(frame, frame_boom_p, boom, boom_frame_p))
constraints.append(c.Spherical(boom, boom_bucket_p, bucket, bucket_boom_p))
constraints.append(c.Spherical(boom, boom_bellcrank_p, bellcrank, bellcrank_boom_p))
constraints.append(c.Spherical(bellcrank, bellcrank_t_arm_p, t_arm, t_arm_bellcrank_p))
constraints.append(c.Fixed(frame, frame_cp, ground, ground_p))

hydcyl1 = l.NormalLoad(frame, frame_hydcyl1_p, bellcrank, bellcrank_hydcyl1_p, 2000.)
hydcyl2 = l.NormalLoad(frame, frame_hydcyl2_p, boom, boom_hydcyl2_p, 18000.)
loads = [hydcyl1, hydcyl2]

s, qdot = positioning.get_state(bodies)
positioning.adjust_to_constraints(s, bodies, constraints)

if __name__ == '__main__':
  
  ssaved, qdotsaved = sim.simulate(dt, int(steps), s, qdot, bodies=bodies, constraints=constraints, loads=loads)
  
  v.animate(ssaved, bodies, pt=40, zoom=0.6, rotspeed=0.5, n=[0.0,0.0,1.0], show_on_screen=True, savepics=False)
  
