import numpy as np
from typing import NamedTuple
import userParams_RWPT as UserP

input_default = UserP.ParamsIn()


# this class holds all parameters that may be needed by the model during
# initialization or simulation time
# they are defined or calculated using values from the `ParamsIn` object
class InputParams(object):
  def __init__(self, input=input_default):
    # total (max) number of particles
    self.N = input.N
    # time step length [s]
    self.dt = input.dt
    # number of spatial dimensions
    self.dim = input.dim
    # max simulation time
    maxT = input.maxT
    # number of time steps between saving time series data
    self.saveInterval = input.saveInterval
    # particles enter the domain 'emit_num' at a time, for each of the first
    # 'emit_steps' time steps
    self.emit_steps = input.emit_steps
    self.emitterVel = np.array(input.emitterVel)

    # shape of full-sized arrrays
    self.shapeA = (self.N, self.dim)

    # make sure that dt evenly divides maxT and that saveInterval evenly
    # divides the number of time steps (maxT / dt).
    # if not, fix this by extending maxT
    stepsCalc = self.calc_steps(maxT, self.saveInterval, self.dt)
    self.maxT = stepsCalc.maxT
    self.nSteps = stepsCalc.tSteps
    self.nSaveSteps = stepsCalc.nSaveSteps
    # the shape of the array we will save the velocity or position data
    self.shapeSave = (self.N, self.dim, self.nSaveSteps)
    # self.NP_frame = np.zeros(self.nSaveSteps)

    self.emitNum = int(np.ceil(self.N / self.emit_steps))

    # TODO: For now, we assume a [0, 1] hypercube
    self.maxL_omega = input.maxL_omega
    self.Omega = np.zeros((2, self.dim))
    for i in range(self.dim):
      self.Omega[1, i] = self.maxL_omega

    # (initial) velocity statistics
    # mean velocities [m/s]
    self.mean_v0 = input.mean_v0
    self.mean_vel = self.mean_v0 * np.ones(self.shapeA)
    # average boundary-layer zonal wind variances [m^2/s^2]
    self.sigma_vel = input.sigma_v0 * np.ones(self.shapeA)
    self.vel_IC = input.initial_condition_vel
    self.IC_Case = input.IC_Case
    # self.v0 = self.init_velocity(self.mean_vel,
    #                              self.sigma_vel,
    #                              self.vel_IC,
    #                              self.shapeA)

    # packets get emitted from the emitter location, so use initial value
    self.initial_pos = np.array(input.emitterX0)
    # self.X0 = self.init_X(self.shapeA, initial_pos)
    # self.X0 = input.emitterX0

    self.invTL = 1 / input.TL_0

  # calculate various time-stepping-related quantities
  def calc_steps(self, maxT, saveInterval, dt):
    nSteps = NamedTuple('stepsCalc', [('tSteps', int),
                                      ('maxT', float),
                                      ('nSaveSteps', int)])
    nSteps.maxT = maxT
    # remember: `int()` *truncates* (rounds down)
    nSteps.tSteps = int(nSteps.maxT / dt)
    rem = int(np.mod(nSteps.maxT, nSteps.tSteps))
    if rem != 0:
      nSteps.tSteps += 1
      nSteps.maxT = nSteps.tSteps * dt
    nSteps.nSaveSteps = int(float(nSteps.tSteps) / saveInterval)
    # already accounts for first time step, so add 1 for final
    nSteps.nSaveSteps += 1
    rem = int(np.mod(nSteps.tSteps, saveInterval))
    # if nSaveSteps doesn't evenly divide tSteps, add time steps maintain time
    # difference between time series entries, and add 1 more saveStep
    if rem != 0:
      nSteps.nSaveSteps += 1
      nSteps.tSteps += (saveInterval - rem)
    return nSteps

  # # NOTE: the initial velocity needs to be divergence-free, but for now, we
  # # randomly scatter according to the distribution of initial velocity
  # def init_velocity(self, mean_v0, sigma_vel, vel_IC, shapeA):
  #   if vel_IC is self.IC_Case.Point:
  #     velVec = mean_v0 * np.ones(shapeA)
  #   elif vel_IC is self.IC_Case.Gaussian:
  #     velVec = np.random.normal(mean_v0, sigma_vel, shapeA)
  #   return velVec

  # def init_X(self, shapeX, initial_pos):
  #   X0 = np.ndarray(shapeX)
  #   for p in range(shapeX[0]):
  #     X0[p, :] = initial_pos
  #   return X0
