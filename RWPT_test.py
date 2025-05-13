import pytest
import numpy as np
import RWPT_params as PTp
import RWPT_fxns as PTfxn
import particles as PT


def test_calc_steps():
  params = PTp.InputParams()
  nSteps = params.calc_steps(params.maxT, params.saveInterval, params.dt)
  for i in range(nSteps.nSaveSteps):
    finalStep = i * params.saveInterval
  msg = f'Calculated final step = {finalStep}, ' \
        f'nSteps = {nSteps.tSteps}, ' \
        f'nSteps.tSteps = {nSteps.tSteps}, ' \
        f'nSteps.nSaveSteps = {nSteps.nSaveSteps}, ' \
        f'nSteps.maxT = {nSteps.maxT}, ' \
        f'saveInterval = {params.saveInterval}, dt = {params.dt}'
  assert finalStep == nSteps.tSteps, msg
  # ===========================================================================
  # first do a couple of dumb ones that caused bugs on initial testing
  # ===========================================================================
  maxT = 1024
  saveInterval = 112
  dt = 1
  nSteps = params.calc_steps(maxT, saveInterval, dt)
  for i in range(nSteps.nSaveSteps):
    finalStep = i * saveInterval
  msg = f'Calculated final step = {finalStep}, ' \
        f'nSteps = {nSteps.tSteps}, ' \
        f'nSteps.tSteps = {nSteps.tSteps}, ' \
        f'nSteps.nSaveSteps = {nSteps.nSaveSteps}, ' \
        f'nSteps.maxT = {nSteps.maxT}, ' \
        f'saveInterval = {saveInterval}, dt = {dt}'
  assert finalStep == nSteps.tSteps, msg
  maxT = 2161
  saveInterval = 50
  dt = 5
  nSteps = params.calc_steps(maxT, saveInterval, dt)
  for i in range(nSteps.nSaveSteps):
    finalStep = i * saveInterval
  msg = f'Calculated final step = {finalStep}, ' \
        f'nSteps = {nSteps.tSteps}, ' \
        f'nSteps.tSteps = {nSteps.tSteps}, ' \
        f'nSteps.nSaveSteps = {nSteps.nSaveSteps}, ' \
        f'nSteps.maxT = {nSteps.maxT}, ' \
        f'saveInterval = {saveInterval}, dt = {dt}'
  assert finalStep == nSteps.tSteps, msg
  # ===========================================================================
  # now stress test with some random values
  # ===========================================================================
  for ens in range(10):
    maxT = np.random.randint(1000, 5000)
    saveInterval = np.random.randint(20, 200)
    dt = np.random.choice([1, 2, 5, 10])
    nSteps = params.calc_steps(maxT, saveInterval, dt)
    for i in range(nSteps.nSaveSteps):
      finalStep = i * saveInterval
    msg = f'Calculated final step = {finalStep}, ' \
          f'nSteps = {nSteps.tSteps}, ' \
          f'nSteps.tSteps = {nSteps.tSteps}, ' \
          f'nSteps.nSaveSteps = {nSteps.nSaveSteps}, ' \
          f'nSteps.maxT = {nSteps.maxT}, ' \
          f'saveInterval = {saveInterval}, dt = {dt}'
    assert finalStep == nSteps.tSteps, msg


def urand_inBounds(bbox):
  xloc = np.random.uniform(bbox[0, 0], bbox[1, 0], 1)
  yloc = np.random.uniform(bbox[0, 1], bbox[1, 1], 1)
  zloc = np.random.uniform(bbox[0, 2], bbox[1, 2], 1)
  return np.array([xloc, yloc, zloc]).squeeze()


def test_periodic_BC():
  params = PTp.InputParams()

  up_del1 = [0, 0.42]
  up_del2 = [0, 0.68]
  up_del3 = [0, 7.37]
  upDel = np.array([np.array([up_del1, up_del1, up_del1]).T,
                    np.array([up_del2, up_del2, up_del2]).T,
                    np.array([up_del3, up_del3, up_del3]).T,
                    np.array([up_del1, up_del2, up_del3]).T])

  down_del1 = [-0.72, 0]
  down_del2 = [-0.14, 0]
  down_del3 = [-3.86, 0]
  downDel = np.array([np.array([down_del1, down_del1, down_del1]).T,
                      np.array([down_del2, down_del2, down_del2]).T,
                      np.array([down_del3, down_del3, down_del3]).T,
                      np.array([down_del1, down_del2, down_del3]).T])

  omega1 = np.array([[0, 1],
                     [0, 1],
                     [0, 1]]).T
  omega2 = np.array([[-1, 1],
                     [-1, 1],
                     [-1, 1]]).T
  omega3 = np.array([[-4.2, 7],
                     [-4.2, 7],
                     [-4.2, 7]]).T
  omega4 = np.array([[0, 1],
                     [-1, 1],
                     [-4.2, 7]]).T
  Omega = np.array([omega1, omega2, omega3, omega4])

  oshape = (4, 2, 3)
  msg = f'Omega shape = {np.shape(Omega)}'
  assert oshape == np.shape(Omega), msg

  for i in np.arange(3, 4):
    otmp = Omega[i, :, :]
    tVec1 = Omega[i, 0, :] + downDel[i, 0, :]
    tVec2 = Omega[i, 1, :] + upDel[i, 1, :]
    testVec = np.vstack((urand_inBounds(otmp),
                         urand_inBounds(otmp),
                         tVec1, tVec2))
    for j in range(3):
      params.Omega = otmp[:, 0:j + 1]
      params.dim = j + 1
      resultVec = PTfxn.apply_BC(testVec[:, 0:j + 1], params)
      for k in range(j):
        msg1 = 'BC incorrectly applied--lower bound incorrect. ' \
               f'current dim = {k}, result = {resultVec[:, k]}, ' \
               f'test val = {params.Omega[0, k]}'
        assert not np.any(resultVec[:, k] < params.Omega[0, k]), msg1
        msg2 = 'BC incorrectly applied--upper bound incorrect. ' \
               f'current dim = {k}, result = {resultVec[:, k]}, ' \
               f'test val = {params.Omega[1, k]}'
        assert not np.any(resultVec[:, k] > params.Omega[1, k]), msg2


def test_interval_save():
  p = PT.Particles()
  params = p.params

  p.velVec = np.random.random((10, 2))
  p.XVec = np.random.random((10, 2))
  ansV = p.velVec
  ansX = p.XVec
  p.tSeries_X = np.zeros((10, 2, 5))
  p.tSeries_vel = np.zeros((10, 2, 5))

  # First, test with the object-provided `saveInterval`
  if params.saveInterval <= 1:
    return
  else:
    # write to entry 0
    step = 0
    p.write_data(step)
    assert (p.tSeries_X[:, :, step] == ansX).all()
    assert (p.tSeries_vel[:, :, step] == ansV).all()
    # write to index 2
    smult = 2
    step = smult * params.saveInterval
    p.write_data(step)
    assert (p.tSeries_X[:, :, smult] == ansX).all()
    assert (p.tSeries_vel[:, :, smult] == ansV).all()
    # call for value that won't write
    # i = 4 * params.saveInterval - 1
    smult = 4
    step = smult * params.saveInterval - 1
    p.write_data(step)
    assert (p.tSeries_X[:, :, 0] == ansX).all()
    assert (p.tSeries_X[:, :, 1] == 0).all()
    assert (p.tSeries_X[:, :, 2] == ansX).all()
    assert (p.tSeries_X[:, :, 3] == 0).all()
    assert (p.tSeries_X[:, :, 4] == 0).all()
    assert (p.tSeries_vel[:, :, 0] == ansV).all()
    assert (p.tSeries_vel[:, :, 1] == 0).all()
    assert (p.tSeries_vel[:, :, 2] == ansV).all()
    assert (p.tSeries_vel[:, :, 3] == 0).all()
    assert (p.tSeries_vel[:, :, 4] == 0).all()

  q = PT.Particles()
  params = q.params

  q.velVec = np.random.random((10, 2))
  q.XVec = np.random.random((10, 2))
  ansV = q.velVec
  ansX = q.XVec
  q.tSeries_X = np.zeros((10, 2, 3))
  q.tSeries_vel = np.zeros((10, 2, 3))

  # Now test for a user-provided `saveInterval`
  # write to entry 0
  params.saveInterval = 42
  step = 0
  q.write_data(step)
  assert (q.tSeries_X[:, :, 0] == ansX).all()
  assert (q.tSeries_vel[:, :, 0] == ansV).all()
  # write to entry 1
  step = 42
  q.write_data(step)
  assert (q.tSeries_X[:, :, 1] == ansX).all()
  assert (q.tSeries_vel[:, :, 1] == ansV).all()
  # no write
  step = 21
  q.write_data(step)
  assert (q.tSeries_X[:, :, 0] == ansX).all()
  assert (q.tSeries_X[:, :, 1] == ansX).all()
  assert (q.tSeries_X[:, :, 2] == 0).all()
  assert (q.tSeries_vel[:, :, 0] == ansV).all()
  assert (q.tSeries_vel[:, :, 1] == ansV).all()
  assert (q.tSeries_vel[:, :, 2] == 0).all()

  # now make sure that things are working properly when N_current < N_total
  pq = PT.Particles()
  assert (pq.tSeries_X == 0).all()
  for nn in range(pq.N_current):
    for d in range(pq.params.dim):
      pq.XVec[nn, d] = np.random.rand()
  step = 0
  pq.write_data(step)
  assert (pq.tSeries_X[0:pq.N_current, :, step] != 0).all()
  assert (pq.tSeries_X[pq.N_current:, :, step] == 0).all()


def test_initX():
  params = PTp.InputParams()

  print(params.initial_pos)
  print(params.initial_pos[0] == 0)
  print(params.initial_pos[1] == 0)
  print(np.all(params.initial_pos == 0))
  assert (np.all(params.initial_pos == 0))
  testX = PTfxn.init_X(params)
  print(f'Input Initial X position = {params.initial_pos}')
  assert (np.all(testX == 0))


# TODO: test convergence rate?
def test_advect_tracer():
  params = PTp.InputParams()

  X0 = [[0, 0],
        [1, 1]]
  vel = [0, 1] * np.ones(np.shape(X0))
  params.dt = 1
  testX = PTfxn.advect_tracer(X0, vel, params)
  assert (testX == [[0, 1],
                    [1, 2]]).all()

  X0 = np.array([[0, 0], [1, 1]])
  vel = np.array([[-1, 0.8], [3, -2.4]])
  ans = [[-0.5, 0.4],
         [2.5, -0.2]]
  params.dt = 0.5
  testX = PTfxn.advect_tracer(X0, vel, params)
  assert (testX - ans < 1e-12).all()


# NOTE: slightly vacuous, given that the function is the same as above,
# but that may not always be the case.
def test_move_emitter():
  params = PTp.InputParams()

  X0 = [[0, 0],
        [1, 1]]
  vel = [0, 1] * np.ones(np.shape(X0))
  params.dt = 1
  params.emitterVel = vel
  testX = PTfxn.move_emitter(X0, params)
  assert (testX == [[0, 1],
                    [1, 2]]).all()

  X0 = np.array([[0, 0], [1, 1]])
  vel = np.array([[-1, 0.8], [3, -2.4]])
  ans = [[-0.5, 0.4],
         [2.5, -0.2]]
  params.dt = 0.5
  params.emitterVel = vel
  testX = PTfxn.move_emitter(X0, params)
  assert (testX - ans < 1e-12).all()


def test_enum_test():
  params = PTp.InputParams()
  assert params.vel_IC == params.IC_Case.Point or \
                          params.IC_Case.Gaussian or \
                          params.IC_Case.Uniform


def test_update_particles():
  p = PT.Particles()
  xnow = p.get_xNow()
  assert (xnow == 0).all()

  p.update_quantities_emit()
  xnow = p.get_xNow()
  # assert (p.N_current == (2 * p.emitNum))
  assert (not np.all(p.get_xNow() == 1))

  p.update_quantities()
  # NOTE: this should almost surely be true...
  #       but hitting the lottery is always *possible*
  assert (np.all(p.XVec != 0))


# # FIXME: incorporate the old test_interval_save()
# def test_write_data():
#   pq = PT.Particles()
#   assert (pq.tSeries_X == 0).all()
#   for nn in range(pq.N_current):
#     for d in range(pq.params.dim):
#       pq.XVec[nn, d] = np.random.rand()
#   step = 0
#   pq.write_data(step)
#   assert (pq.tSeries_X[0:pq.N_current, :, step] != 0).all()
#   assert (pq.tSeries_X[pq.N_current:, :, step] == 0).all()


def check_X_update(p, N_prev):
  randices = np.random.randint(N_prev, p.N_total, (20, 1))
  nz_vel = p.emitterVel != 0
  if np.all(nz_vel):
    for r in randices:
      assert (p.XVec[r, :] != 0).all()
  elif np.any(nz_vel):
    for d in p.params.dim:
      if nz_vel[d]:
        for r in randices:
          assert (p.XVec[r, d] != 0).all()
      else:
        for r in randices:
          assert (p.XVec[r, d] == 0).all()
  else:
    for r in randices:
      assert (p.XVec[r, :] == 0).all()


def test_particle_step():
  p = PT.Particles()

  assert (p.N_current == p.emitNum)
  step = 0
  p.particle_step_emit(step)
  assert (p.N_current == (2 * p.emitNum))
  N_prev = p.N_current - p.emitNum
  assert (p.tSeries_X[0:N_prev, :, step] != 0).all()
  # first check default from userParams
  check_X_update(p, N_prev)

  # now check for other choices
  largeNeg = np.random.uniform(-7, -2.1)
  smallNeg = np.random.uniform(-2, 0)
  smallPos = np.random.uniform(0, 3)
  largePos = np.random.uniform(3.1, 11)
  vels = [[largeNeg, smallNeg],
          [smallNeg, smallPos],
          [smallPos, largePos],
          [largeNeg, largePos],
          [smallNeg, largeNeg],
          [smallPos, smallNeg],
          [largePos, smallPos],
          [largePos, largeNeg],
          [largeNeg, 0],
          [smallNeg, 0],
          [smallPos, 0],
          [largePos, 0],
          [0, largeNeg],
          [0, smallNeg],
          [0, smallPos],
          [0, largePos],
          [0, 0]]
  for v in vels:
    q = PT.Particles()
    q.emitterVel = v
    print(f'v = {v}')
    q.particle_step_emit(step)
    N_prev = q.N_current - q.emitNum
    check_X_update(q, N_prev)
