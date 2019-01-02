from .train_op import import_matplotlib
import_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
import io
import scipy.misc
import numpy as np
from utils import train_op, torch_op
import PIL.Image
import cv2

def plotSeries(x,y,xlabel=None,ylabel=None,legend=None):
  if not isinstance(x,list):
    x=[x]
  if not isinstance(y,list):
    y=[y]
  if not isinstance(legend,list):
    legend=[legend]
  assert(len(x)==len(y))
  fig = plt.figure()
  plot = fig.add_subplot ( 111 )
  nplot=len(x)
  try:
    for i in range(nplot):
        plot.plot(x[i],y[i],marker='x',label=legend[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
  except:
    print("nothing to draw!")
  fig.canvas.draw()
  visfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  visfig = visfig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.clf()
  return visfig

def plotHistogram(x,xlabel=None,ylabel=None,legend=None):
    if not isinstance(x,list):
        x=[x]
    if not isinstance(legend,list):
        legend=[legend]
    assert(len(x)==len(legend))
    fig = plt.figure()
    plot = fig.add_subplot ( 111 )
    nplot=len(x)
    for i in range(nplot):
        values, base = np.histogram(x[i], bins=40)
        values = values/sum(values)
        plot.plot(base[:-1], values,label=legend[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.canvas.draw()
    visfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    visfig = visfig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return visfig

def plotCummulative(x,xlabel=None,ylabel=None,legend=None,xmin=None,xmax=None,bins=40):
    if not isinstance(x,list):
        x=[x]
    if not isinstance(legend,list):
        legend=[legend]
    fig = plt.figure()
    plot = fig.add_subplot ( 111 )
    nplot=len(x)
    for i in range(nplot):
        values, base = np.histogram(x[i], bins=bins)
        cumulative = np.cumsum(values)/sum(values)
        plot.plot(base[:-1], cumulative,label=legend[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    axes = plt.gca()
    if xmin is not None and xmax is not None:
        axes.set_xlim([xmin,xmax])
    #axes.set_ylim([ymin,ymax])
    fig.canvas.draw()
    visfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    visfig = visfig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return visfig


def vox_to_image(vox_,imname, mode = 'fix_size'):
    """ vox_ : dim, dim, dim, 1
    """
    print(abs(vox_).mean())
    if mode == 'fix_size':
        vox = (vox_ > 0.5).astype(int)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        dim = vox.shape[0]
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.set_zlim(0, dim)
        #xs,ys,zs = get_points(vox)
        zs,ys,xs = get_points(vox)
        ss = 10
        ax.scatter(xs,ys,zs, s=ss)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.savefig(imname)
    else:
        vox = vox_
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        dim = vox.shape[0]
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.set_zlim(0, dim)
        xs,ys,zs = get_points(vox)
        ss = vox[np.where(vox > 0)] * 1.
        ax.scatter(xs,dim-1-ys, dim-1-zs, s=ss)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.savefig(imname)


# 3D visualization
def draw_3d_mlab(voxel,  savepath):
    obj_res = voxel.shape[0]
    voxel = np.reshape(voxel, (obj_res, obj_res, obj_res))
    xx, yy, zz = np.where(voxel >= 0)
    ss = voxel[np.where(voxel >= 0)] * 1.
    mlab.options.offscreen = True
    mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(400, 400))
    s = mlab.points3d(xx, yy, zz, ss,
                      mode="cube",
                      colormap='bone',
                      scale_factor=2)
    #mlab.view(120, 290, 85)
    s.scene.light_manager.lights[0].activate  = True
    s.scene.light_manager.lights[0].intensity = 1.0
    s.scene.light_manager.lights[0].elevation = 30
    s.scene.light_manager.lights[0].azimuth   = -30
    s.scene.light_manager.lights[1].activate  = True
    s.scene.light_manager.lights[1].intensity = 0.3
    s.scene.light_manager.lights[1].elevation = -60
    s.scene.light_manager.lights[1].azimuth   = -30
    s.scene.light_manager.lights[2].activate  = False
    if savepath == 0:
        return mlab.show()
    return  mlab.savefig(savepath)


def show3D(ax, points, edges, c = (255, 0, 0)):
    J = points.shape[1]
    x, y, z = np.zeros((3, J)) 
    for j in range(J):
      x[j] = points[0,j] 
      y[j] = points[1,j] 
      z[j] = points[2,j]
    ax.scatter(x, y, z)
    for e in edges:
      if (x[e][0] == -1 and y[e][0] == -1 and z[e][0] == -1) or (x[e][1] == -1 and y[e][1] == -1 and z[e][1] == -1):
        continue
      else:
        ax.plot(x[e], y[e], z[e])

def visualize_keypoint(keypoints, mode,fig,edges):
    
    if mode == 'align':
        plt.clf()
        n = keypoints.shape[0]
        keypoints_tgt = keypoints[0:1,:,:].repeat(n,1,1)
        r,t,_ = rigid_transform_np(keypoints, keypoints_tgt)
        ax = fig.add_subplot(111, projection='3d')
        r = torch_op.npy(r)
        keypoints = torch_op.npy(keypoints)
        t = torch_op.npy(t)

        for i in range(n):
            newpoint = np.matmul(r[i],keypoints[i]) + t[i]
            #ax.scatter(newpoint[0,:], newpoint[1,:], newpoint[2,:])
            show3D(ax, newpoint, edges)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
    else:
        plt.clf()
        n = keypoints.shape[0]
        keypoints = torch_op.npy(keypoints)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n):
            #ax.scatter(keypoints[i,0,:], keypoints[i,1,:], keypoints[i,2,:])
            show3D(ax, keypoints[i,:,:], edges)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
    return buf

def visualize_keypoint_show(keypoints, mode,fig,edges):
    
    if mode == 'align':
        plt.clf()
        n = keypoints.shape[0]
        keypoints_tgt = keypoints[0:1,:,:].repeat(n,1,1)
        r,t,_ = rigid_transform_np(keypoints, keypoints_tgt)
        ax = fig.add_subplot(111, projection='3d')
        r = torch_op.npy(r)
        keypoints = torch_op.npy(keypoints)
        t = torch_op.npy(t)
        for i in range(n):
            newpoint = np.matmul(r[i],keypoints[i]) + t[i]
            show3D(ax, newpoint, edges)

    else:
        plt.clf()
        n = keypoints.shape[0]
        keypoints = torch_op.npy(keypoints)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n):
            #ax.scatter(keypoints[i,0,:], keypoints[i,1,:], keypoints[i,2,:])
            show3D(ax, keypoints[i,:,:], edges)
    plt.show()

def plt2npy():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = PIL.Image.open(buf)
    img = np.array(img)[:,:,:3]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def PlotContour(levelset,N=20):
    plt.clf()
    plt.axis('off')
    ys,xs = np.meshgrid(range(levelset.shape[0]),range(levelset.shape[1]),indexing='ij')
    plt.contour(xs,ys,levelset,N)
    plt.colorbar()
    contour = plt2npy()
    #contour = contour[::-1,:,:]
    return contour