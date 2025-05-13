# import RWPT_params as PTp
import numpy as np
# import userParams_RWPT as userP


# getter for the Input Parameters object--intializes it from
# the User Input class
# def get_params():
#   input = userP.ParamsIn()
#   p = PTp.InputParams(input)
#   return p


# define a local params to use for default values below
# f_params = get_params()


# calculates the differential velocity update (dU) according to the
# Langevin Eq. from Pope
def dVel_func(velStar, params):
  RV = np.random.standard_normal((params.N, params.dim))
  return (params.mean_vel - velStar) * params.dt * params.invTL \
         + np.sqrt(2 * params.sigma_vel * params.dt * params.invTL) * RV


def is_saveStep(i, saveInterval):
  saveIdx = i / saveInterval
  # expected behavior is that integer division results in float if the result
  # is not an integer
  if saveIdx.is_integer():
    val = saveIdx
  else:
    val = -42
  return int(val)


# imposes the periodic boundary conditions
# TODO: any reason to test/check that a particle hasn't wrapped the boundary
# twice?... seems unlikely
def apply_BC(vel, params):
  lower = params.Omega[0, :]
  upper = params.Omega[1, :]
  dim_len = upper - lower
  for i in range(params.dim):
    vel[:, i] = np.mod(vel[:, i] - lower[i], dim_len[i])
    vel[:, i] += lower[i]
  return vel


# # get the initial velocities for all particles
# def get_v0():
#   vel = f_params.v0
#   return vel


# # get the initial particle positions
# def get_X0():
#   X = f_params.X0
#   return X


# initialize the arrays that hold the time series data to the proper shape
def init_tSeries(params):
  return np.zeros(params.shapeSave), \
         np.zeros(params.shapeSave), \
         np.zeros(params.nSaveSteps)


# def init_tSeries_Np(params):
#   return np.zeros(params.nSaveSteps)

# def alt_init_tSeries(saveSteps=f_params.nSaveSteps):
#   ts = np.empty(saveSteps, object)
#   return ts


# def advect_tracer(X, vel, Np, dt=f_params.dt, dim=f_params.dim):
#   outX = np.ndarray(np.shape(X))
#   for d in range(dim):
#     outX[0:Np, d] = X[0:Np, d] + vel[0:Np, d] * dt
#   return outX

# simple forward-Euler integration of dX/dt = v(t)
def advect_tracer(X, vel, params):
  X = X + vel * params.dt
  return X


# moves the tracer emitter (e.g., ship emitting exhaust) according to its
# defined velocity
def move_emitter(emX, params):
  emX = emX + params.emitterVel * params.dt
  return emX


def init_X(params):
    X0 = np.ndarray(params.shapeA)
    for p in range(params.shapeA[0]):
      X0[p, :] = params.initial_pos
    return X0


# NOTE: the initial velocity needs to be divergence-free, but for now, we
# randomly scatter according to the distribution of initial velocity
def init_velocity(params):
  if params.vel_IC is params.IC_Case.Point:
    velVec = params.mean_v0 * np.ones(params.shapeA)
  elif params.vel_IC is params.IC_Case.Gaussian:
    velVec = np.random.normal(params.mean_v0, params.sigma_vel, params.shapeA)
  return velVec
