import RWPT_params as PTp
import RWPT_fxns as PTfxn


class Particles:
  def __init__(self):
    self.params = PTp.InputParams()

    self.dim = self.params.dim

    self.velVec = PTfxn.init_velocity(self.params)
    self.XVec = PTfxn.init_X(self.params)

    self.tSeries_vel, \
      self.tSeries_X, \
      self.tSeries_Np = PTfxn.init_tSeries(self.params)

    self.emitNum = self.params.emitNum
    self.emitterVel = self.params.emitterVel
    self.emitLoc = self.params.initial_pos

    self.N_current = self.emitNum
    self.N_total = self.params.N

    self.nSteps = self.params.nSteps
    self.emit_steps = self.params.emit_steps

  def write_data(self, step):
    save_idx = PTfxn.is_saveStep(step, self.params.saveInterval)
    if save_idx >= 0:
      for d in range(self.dim):
        self.tSeries_vel[:, d, save_idx] = self.velVec[:, d]
        self.tSeries_X[:, d, save_idx] = self.XVec[:, d]
      self.tSeries_Np[save_idx] = self.N_current

  def get_xNow(self):
    return self.XVec[0:self.N_current, :]

  def get_xLater(self):
    return self.XVec[self.N_current:, :]

  def get_velNow(self):
    return self.velVec[0:self.N_current, :]

  # FIXME: apply BC
  def update_quantities(self):
    # update the velocity
    self.velVec += PTfxn.dVel_func(self.velVec, self.params)
    # velVec = PTfxn.apply_BC(velVec)
    self.XVec = PTfxn.advect_tracer(self.XVec, self.velVec, self.params)

  def update_quantities_emit(self):
    x = self.get_xNow()
    v = self.get_velNow()
    # update the velocity
    self.velVec += PTfxn.dVel_func(self.velVec, self.params)
    # velVec = PTfxn.apply_BC(velVec)
    self.XVec[0:self.N_current] = PTfxn.advect_tracer(x, v, self.params)

  # NOTE: the intended ordering (used below) is:
  #       - update(_emit)
  #       - write_data
  #       - increment N_current (if applicable)

  def particle_step(self, step):
    self.update_quantities()
    self.write_data(step)

  def particle_step_emit(self, step):
    self.update_quantities_emit()
    self.write_data(step)
    if self.N_current < self.N_total:
      x = self.get_xLater()
      self.XVec[self.N_current:] = PTfxn.move_emitter(x, self.params)
      # add the packet of particles
      self.N_current += self.emitNum
