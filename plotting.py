import matplotlib.pyplot as plt
from matplotlib.colors import LightSource as ls

# from IPython.display import display, Math, Latex
# import matplotlib.font_manager
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"]})

import gif
# import matplotlib.animation as animation

import os
import numpy as np
from enum import Enum


class QOI_Plot(Enum):
  position = 1
  velocity = 2
  pNumber = 3


class PlotStyle(Enum):
  scatter = 1
  histogram = 2


class HistStyle(Enum):
  flat = 1
  bars = 2
  none = 3


# Calculate the most compact tiling for the subplots
# NOTE: written by bot
def most_compact_tiling(n):
  import math
  cols = math.ceil(math.sqrt(n))
  rows = math.ceil(n / cols)
  return rows, cols


def set_qoi(p, qoi):
  if qoi == QOI_Plot.position:
    q = p.tSeries_X
    pcase = '2d'
    xname = r'$X$-Position [L]'
    yname = r'$Y$-Position [L]'
  elif qoi == QOI_Plot.velocity:
    q = p.tSeries_vel
    pcase = '2d'
    xname = r'$X$-Velocity [L/T]'
    yname = r'$Y$-Velocity [L/T]'
  elif qoi == QOI_Plot.pNumber:
    q = p.tSeries_Np
    pcase = '1d'
    xname = 'Time Step'
    yname = 'Number of Particles'
  else:
    raise ValueError("Invalid value for time series qoi.\n" +
                     "Valid choices are: {position, velocity, pNumber}")
  return q, pcase, xname, yname


def get_bounds_onetime(q):
  bounds_x = [min(q[:, 0]), max(q[:, 0])]
  bounds_y = [min(q[:, 1]), max(q[:, 1])]
  return bounds_x, bounds_y


def get_bounds_alltime(q, axlim_pad):
  bounds_x = [np.min(np.min(q[:, 0, :], axis=0)),
              np.max(np.max(q[:, 0, :], axis=0))]
  bounds_y = [np.min(np.min(q[:, 1, :], axis=0)),
              np.max(np.max(q[:, 1, :], axis=0))]

  bounds_x[0] -= axlim_pad
  bounds_x[1] += axlim_pad
  bounds_y[0] -= axlim_pad
  bounds_y[1] += axlim_pad
  return bounds_x, bounds_y


def plot_scatter_2d(ax, q, frame, xname, yname, bx, by):
  ax.plot(q[:, 0, frame], q[:, 1, frame],
          marker='o', linestyle='None', markersize=6)
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)
  ax.set_xlim(bx)
  ax.set_ylim(by)


def plot_hist2d_flat(ax, q, frame, xname, yname, bx, by):
  nbin = 40
  ax.hist2d(q[:, 0, frame], q[:, 1, frame],
            bins=nbin, range=[bx, by], cmap='Blues')
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)


def plot_hist2d_bars(ax, q, frame, xname, yname, bx, by):
  nbin = 40
  hist, xedges, yedges = np.histogram2d(q[:, 0, frame], q[:, 1, frame],
                                        bins=nbin, range=[bx, by],
                                        density=False)

  xx = (xedges[:-1] + xedges[1:]) / 2
  yy = (yedges[:-1] + yedges[1:]) / 2
  XX, YY = np.meshgrid(xx, yy)

  XX = XX.ravel()
  YY = YY.ravel()
  ZZ = np.zeros_like(XX)

  dx = xedges[1] - xedges[0]
  dy = yedges[1] - yedges[0]
  dz = hist.T.ravel()

  azval = 285
  altval = 30
  bl_cmap = plt.get_cmap('Blues')

  light = ls(altdeg=altval + 15, azdeg=azval + 225)
  rgbshade = light.shade(hist.T, cmap=bl_cmap,
                         blend_mode='overlay')
  # rgba = [rgbshade((k - min_height) / max_height) for k in dz]
  shape_shade = np.shape(rgbshade)
  rgba = np.reshape(rgbshade, (shape_shade[0] * shape_shade[1], 4))

  max_height = np.max(dz)
  sz = np.sum(dz)
  for i, z in enumerate(dz):
    dz[i] = z / sz

  ax.bar3d(XX, YY, ZZ, dx, dy, dz, color=rgba, zsort='average',
           lightsource=light)
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)
  ax.set_zlim([0, 0.05])
  ax.view_init(elev=altval, azim=azval)
  return max_height


def plot_scatter_snapshots_2d(nPlot, frames, q, xname, yname, axlim_pad):
  rows, cols = most_compact_tiling(nPlot)
  fig, ax = plt.subplots(rows, cols, figsize=(cols*4, rows*4),
                         gridspec_kw={'wspace': 0.4, 'hspace': 0.3})
  [axlimX, axlimY] = get_bounds_alltime(q[:, :, frames[0]:frames[-1] + 1],
                                        axlim_pad)
  for i, frame in enumerate(frames):
    plot_scatter_2d(ax[i // cols, i % cols], q, frame,
                    xname, yname, axlimX, axlimY)


def plot_scatter_1d(ax, q, frame, xname, yname):
  ax.plot(frame, q[frame], marker='o', linestyle='None', markersize=6)
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)


def plot_scatter_snapshots_1d(frames, q, xname, yname):
  rows = 1
  cols = 1
  fig, ax = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
  for i, frame in enumerate(frames):
    plot_scatter_1d(ax, q, frame, xname, yname)


def plot_scatter_snapshots(p, frames, qoi):
  nPlot = len(frames)
  [q, pcase, xname, yname] = set_qoi(p, qoi)
  match pcase:
    case '2d':
      plot_scatter_snapshots_2d(nPlot, frames, q, xname, yname, axlim_pad=2)
    case '1d':
      plot_scatter_snapshots_1d(frames, q, xname, yname)


def plot_hist_snapshots(p, frames, qoi, hstyle):
  nPlot = len(frames)
  axlim_pad = 2
  [q, pcase, xname, yname] = set_qoi(p, qoi)
  rows, cols = most_compact_tiling(nPlot)
  match hstyle:
    case HistStyle.flat:
      fig, ax = plt.subplots(rows, cols, figsize=(cols*4, rows*4),
                             gridspec_kw={'wspace': 0.4, 'hspace': 0.3})
      [axlimX, axlimY] = get_bounds_alltime(q, axlim_pad)
      for i, frame in enumerate(frames):
        plot_hist2d_flat(ax[i // cols, i % cols], q, frame, xname, yname,
                         axlimX, axlimY)
      for x in ax.flat:
        x.set_xlim(axlimX)
        x.set_ylim(axlimY)
    case HistStyle.bars:
      fig, ax = plt.subplots(rows, cols, figsize=(cols*6, rows*6),
                             gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                             subplot_kw={'projection': '3d'})
      [axlimX, axlimY] = get_bounds_alltime(q, axlim_pad)
      for i, frame in enumerate(frames):
        plot_hist2d_bars(ax[i // cols, i % cols], q, frame, xname, yname,
                         axlimX, axlimY)
      for x in ax.flat:
        x.set_xlim(axlimX)
        x.set_ylim(axlimY)
    case HistStyle.none:
      raise ValueError("Invalid value for histogram style.\n" +
                       "Valid choices are: {flat, bars}")


def plot_snapshots(p, frames, qoi, pstyle, hstyle=HistStyle.none):
  match pstyle:
    case PlotStyle.scatter:
      plot_scatter_snapshots(p, frames, qoi)
    case PlotStyle.histogram:
      plot_hist_snapshots(p, frames, qoi, hstyle)


def get_hist_frame(q, hstyle, frame, xname, yname, bx, by, params):
  match hstyle:
    case HistStyle.flat:
      fig, ax = plt.subplots(1, 1, figsize=(6, 6))
      plot_hist2d_flat(ax, q, frame, xname, yname, bx, by)
      ax.text(0.5, 0.8,
              s=f'time = {frame * params.saveInterval} ' +
                f'of {params.maxT}',
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes)
    case HistStyle.bars:
      fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                             subplot_kw={'projection': '3d'})
      plot_hist2d_bars(ax, q, frame, xname, yname, bx, by)
      ax.text(0.5, 0.8, 0.9,
              s=f'time = {frame * params.saveInterval} ' +
                f'of {params.maxT}',
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes)
    case HistStyle.none:
      raise ValueError("Invalid value for histogram style.\n" +
                       "Valid choices are: {flat, bars}")
  return ax


def get_scatter_frame(q, frame, xname, yname, bx, by, params):
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))
  plot_scatter_2d(ax, q, frame, xname, yname, bx, by)
  ax.text(0.5, 0.8,
          s=f'time = {frame * params.saveInterval} of {params.maxT}',
          horizontalalignment='center', verticalalignment='center',
          transform=ax.transAxes)


def get_frame(p, qoi, frame, pstyle, hstyle):
  [q, pcase, xname, yname] = set_qoi(p, qoi)
  [axlimX, axlimY] = get_bounds_alltime(q, axlim_pad=2)
  match pstyle:
    case PlotStyle.scatter:
      ax = get_scatter_frame(q, frame, xname, yname, axlimX, axlimY, p.params)
    case PlotStyle.histogram:
      ax = get_hist_frame(q, hstyle, frame, xname, yname, axlimX, axlimY,
                          p.params)
  ax.set_xlim(axlimX)
  ax.set_ylim(axlimY)


def make_qoi_tGIF(p, qoi, pstyle, hstyle=HistStyle.none, gif_name='test.gif',
                  make_gif=False):
  if make_gif:
    if os.path.exists(gif_name):
      os.remove(gif_name)

    @gif.frame
    def plotter(p, qoi, frame):
      get_frame(p, qoi, frame, pstyle, hstyle)

    images = []
    for frame in range(0, p.params.nSaveSteps):
      images.append(plotter(p, qoi, frame))
    gif.save(images, gif_name, duration=100)
