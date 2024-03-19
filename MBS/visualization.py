import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from MBS.transformations import R, get_p


def draw_body(body, s, ax, col='black', n=[0.0, 0.0, 1.0], phi=0):
  '''Draws lines between all points of one body. The view can be rotated.
  
  Args:
    body (Body obj): Body to plot.
    s (array 7x1) Translation and quaternion of body.
    ax (Matplotlib axis obj): Where to plot.
    col (Matplotlib color argument)
    n (3-list, optional): Normalised vector of rotation. Defaults to [0,0,1]
    phi (float, optional): View rotation in radians. Defaults to 0.
  
  
  '''
  pos = s[0:3]
  rot = s[3:7]
  R_global = R(get_p(n, phi))
  posr = R_global.dot(pos)
  ax.plot(posr[0,0], posr[2,0], 'o', color=col)
  vertices = list(combinations([point.u for point in body.points], 2))
  for i in range(len(vertices)):
    u1 = R_global.dot(pos + R(rot).dot(vertices[i][0]))
    u2 = R_global.dot(pos + R(rot).dot(vertices[i][1]))
    ax.plot([u1[0,0], u2[0,0]], [u1[2,0], u2[2,0]], color=col)

def plot_bodies(bodies, s, ax=-1, zoom=1, n=[0.0, 0.0, 1.0], phi=0):
  '''Draws m bodies in the list bodies. The view can be rotated.
  
  Args:
    bodies (m-list of Body obj): Bodies to plot.
    s (array 7mx1): Translational and rotational state of bodies.
    ax (Matplotlib axis obj, optional): Where to plot. Defaults to current axis.
    zoom (float, optional): Magnification factor of plot. Defaults to 1.
    n (3-list, optional): Normalised vector of rotation. Defaults to [0,0,1]
    phi (float, optional): View rotation in radians. Defaults to 0.
    
  Examples:
    plot_bodies(bodies, s)
      A quick look on the configuration, e.g. for simulation setup.'''
  if ax == -1:
    ax = plt.gca()
  ax.cla()
  ax.axis('scaled')
  ax.axis(1/zoom*np.array([-1.0, 1.0, -1.0, 1.0]))
  [draw_body(bodies[iBody], s[7*iBody:7*iBody+7], ax, col='C'+str(iBody), n=n, phi=phi) for iBody in range(len(bodies))]

def animate(ssaved, bodies, pt=100, zoom=1, rotspeed=0, n=[0.0, 0.0, 1.0], show_on_screen=True, savepics=False, directory=os.getcwd(), fname='pic'):
  '''Plots a series of saved states in two views.
  
  Optionally plays the animation. Optionally saves figures of the animation, 
  for subsequent generation of video files. You can add a rotational 
  velocity to the animation to make it look cool.
  
  Args:
    ssaved (list of array 7mx1): time-listed state arrays
    bodies (list of Body obj): Bodies to plot.
    pt (int, optional): Every pt time-state in ssaved will be drawn. 
      Defaults to 100.
    zoom (float, optional): Magnification factor of plot. Defaults to 1.
    rotspeed (float, optional): Revolutions per animation to rotate the
      drawn bodies. 
    n (3-list, optional): Normalised vector of rotation. Defaults to [0.0, 0.0, 1.0]
    show_on_screen (bool, optional): Whether to display the animation. Defaults to True.
    savepics (bool, optional): Whether to save the animation frames to file. 
      Defaults to False.
    directory (bool, optional): Location to save animation frames. Defaults
      to current work directory.
    fname (string, optional): Filename of saved animation frames. The 
      files are named fname0000.png, fname0001.png, ... Defaults to 'pic'.
  
  Examples:
    animate(ssaved, bodies, pt=25, rotspeed=1)
      Animates the states in ssaved, showing every 25 timesteps and applying
      a visual rotation of in total one revolution.
    animate(ssaved, bodies, show_on_screen=False, savepics=True, fname='simul')
      Generates simulXXXX.png images in the current directory, without 
      displaying all frames to the user during generation. 
      
  '''
  fig, (ax1, ax2) = plt.subplots(1, 2)
  frames = int(np.ceil(len(ssaved)/pt))
  
  for frame in range(frames):
    vs = frame*pt
    print('visualising', vs,' / ',len(ssaved))
    phi = rotspeed*frame/frames*2.*np.pi
    plot_bodies(bodies, ssaved[vs], ax=ax1, zoom=zoom, phi=0+phi, n=n)
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    plot_bodies(bodies, ssaved[vs], ax=ax2, zoom=zoom, phi=-np.pi/2+phi, n=n)
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    if show_on_screen:
      plt.show(block=False)
      plt.pause(0.001)
    if savepics:
      plt.savefig(os.path.join(directory,fname+str(frame).rjust(4, '0')+'.png'))
  if show_on_screen:
    plt.show()
  else:
    plt.close()
  
