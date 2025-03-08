# this class holds the user-defined model-level parameters
# these are separated for the sake of keeping all the things a user touches
# in a single place
class ParamsIn(object):
  def __init__(self):
    # total (max) number of particles
    self.N = int(5e2)
    # time step length [s]
    self.dt = 1

    # number of spatial dimensions
    self.dim = 2
    # For now, we assume a [0, 1] hypercube
    self.maxL_omega = 1
    # max simulation time
    self.maxT = 1e2
    # initial mean velocity
    # TODO: for now, isotropic, but can make this more interesting
    self.mean_v0 = [-1, 2]
    # initial position of tracer particles
    # TODO: for now, stationary, but can make this more interesting
    self.initial_pos = 0

    # # tuning parameter [-] (from Pope Sec. 12.4 {p. 504})
    # C_0 = 2.1
    # relaxation time scale [1/s]
    # NOTE: $k/epsilon$ is taken as a ratio to be "MEAN_TS" = TL_0
    # NOTE: T_L = k / (0.75 * C_0 * epsilon)
    # # Approximate value from GMD paper p. 7883
    # TL_0 = 0.6 * 3600
    self.TL_0 = 500

    # number of time steps between saving time series data
    self.saveInterval = int(2e0)
    # particles enter the domain 'emit_num' at a time, for each of the first
    # 'emit_steps' time steps
    self.emit_steps = 25
    # velocity of the tracer emitter (e.g., ship)
    self.emitterVel = [3, 3]
