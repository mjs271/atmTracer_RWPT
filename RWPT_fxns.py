import RWPT_params as PTp
import numpy as np
import userParams_RWPT as userP


# getter for the Input Parameters object--intializes it from
# the User Input class
def get_params():
  input = userP.ParamsIn()
  p = PTp.InputParams(input)
  return p


# define a local params to use for default values below
f_params = get_params()


# calculates the differential velocity update (dU) according to the
# Langevin Eq. from Pope
def dVel_func(velStar, velBar=f_params.mean_vel, sigma=f_params.sigma_vel,
              TL_inv=f_params.invTL, delta_t=f_params.dt, d=f_params.dim,
              Np=f_params.N):
  RV = np.random.standard_normal((Np, d))
  return (velBar[0:Np, :] - velStar) * delta_t * TL_inv \
         + np.sqrt(2 * sigma[0:Np, :] * delta_t * TL_inv) * RV


# saves current state to a timeSeries array every `saveInterval`
# number of model time steps
def interval_save(velVec, i, outVec, interval=f_params.saveInterval):
  saveIdx = i / interval
  if saveIdx.is_integer():
    outVec[:, :, int(saveIdx)] = velVec
  return outVec


# imposes the periodic boundary conditions
# TODO: any reason to test/check that a particle hasn't wrapped the boundary
# twice?... seems unlikely
def apply_BC(vel, bbox=f_params.Omega, d=f_params.dim):
  lower = bbox[0, :]
  upper = bbox[1, :]
  dim_len = upper - lower
  for i in range(d):
    vel[:, i] = np.mod(vel[:, i] - lower[i], dim_len[i])
    vel[:, i] += lower[i]
  return vel


# get the initial velocities for all particles
def get_v0(vel=f_params.v0):
  vel = apply_BC(vel)
  return vel


# get the initial particle positions
def get_X0(X=f_params.X0):
  return X


# initialize the arrays that hold the time series data to the proper shape
def init_tSeries(tShape=f_params.shapeSave):
  ts = np.zeros(tShape)
  return ts


# simple forward-Euler integration of dX/dt = v(t)
def advect_tracer(X, vel, dt=f_params.dt):
  X = X + vel * dt
  return X


# moves the tracer emitter (e.g., ship emitting exhaust) according to its
# defined velocity
def move_emitter(emX, emVel=f_params.emitterVel, dt=f_params.dt):
  emX = emX + emVel * dt
  return emX
