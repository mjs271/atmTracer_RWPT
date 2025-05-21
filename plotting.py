import matplotlib.pyplot as plt

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


def plot_hist_2d(ax, q, frame, xname, yname, bx, by):
  nbin = 40
  ax.hist2d(q[:, 0, frame], q[:, 1, frame],
            bins=nbin, range=[bx, by], cmap='Blues')
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)


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


def plot_hist_snapshots(p, frames, qoi):
  nPlot = len(frames)
  axlim_pad = 2
  [q, pcase, xname, yname] = set_qoi(p, qoi)
  rows, cols = most_compact_tiling(nPlot)
  fig, ax = plt.subplots(rows, cols, figsize=(cols*4, rows*4),
                         gridspec_kw={'wspace': 0.4, 'hspace': 0.3})
  [axlimX, axlimY] = get_bounds_alltime(q, axlim_pad)
  for i, frame in enumerate(frames):
    plot_hist_2d(ax[i // cols, i % cols],
                 q, frame, xname, yname, axlimX, axlimY)
  for x in ax.flat:
    x.set_xlim(axlimX)
    x.set_ylim(axlimY)


def plot_snapshots(p, frames, qoi, pstyle):
  match pstyle:
    case PlotStyle.scatter:
      plot_scatter_snapshots(p, frames, qoi)
    case PlotStyle.histogram:
      plot_hist_snapshots(p, frames, qoi)


def get_frame(ax, q, frame, pstyle, xname, yname, axlimX, axlimY):
  match pstyle:
    case PlotStyle.scatter:
      plot_scatter_2d(ax, q, frame, xname, yname, axlimX, axlimY)
    case PlotStyle.histogram:
      plot_hist_2d(ax, q, frame, xname, yname, axlimX, axlimY)


def make_qoi_tGIF(p, qoi, pstyle, gif_name, make_gif):
  if make_gif:
    if os.path.exists(gif_name):
      os.remove(gif_name)
    [q, pcase, xname, yname] = set_qoi(p, qoi)
    [axlimX, axlimY] = get_bounds_alltime(q, axlim_pad=2)

    @gif.frame
    def plotter(q, frame):
      fig, ax = plt.subplots(1, 1, figsize=(6, 6))
      get_frame(ax, q, frame, pstyle, xname, yname, axlimX, axlimY)
      ax.set_xlim(axlimX)
      ax.set_ylim(axlimY)
      ax.text(0.5, 0.8,
              f'time = {frame * p.params.saveInterval} of {p.params.maxT}',
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes)

    images = []
    for frame in range(0, p.params.nSaveSteps):
        images.append(plotter(q, frame))
    gif.save(images, gif_name, duration=100)
