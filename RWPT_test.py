import pytest
import numpy as np
import RWPT_fxns as PTfxn


def test_calc_steps():
  params = PTfxn.get_params()
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
      otst = otmp[:, 0:j + 1]
      resultVec = PTfxn.apply_BC(testVec[:, 0:j + 1], otst, j + 1)
      for k in range(j):
        msg1 = 'BC incorrectly applied--lower bound incorrect. ' \
               f'current dim = {k}, result = {resultVec[:, k]}, ' \
               f'test val = {otst[0, k]}'
        assert not np.any(resultVec[:, k] < otst[0, k]), msg1
        msg2 = 'BC incorrectly applied--upper bound incorrect. ' \
               f'current dim = {k}, result = {resultVec[:, k]}, ' \
               f'test val = {otst[1, k]}'
        assert not np.any(resultVec[:, k] > otst[1, k]), msg2


def test_interval_save():
  params = PTfxn.get_params()
  velVec = np.random.random((10, 2))
  ans = velVec
  testVec = np.zeros((10, 2, 5 * params.saveInterval))

  # First, test with the object-provided `saveInterval`
  if params.saveInterval <= 1:
    return
  else:
    # write to entry 0
    testVec = PTfxn.interval_save(velVec, 0, testVec)
    assert (testVec[:, :, 0] == ans).all()
    # write to index 2
    i = 2 * params.saveInterval
    testVec = PTfxn.interval_save(velVec, i, testVec)
    assert (testVec[:, :, 2] == ans).all()
    # call for value that won't write
    i = 4 * params.saveInterval - 1
    testVec = PTfxn.interval_save(velVec, i, testVec)
    assert (testVec[:, :, 0] == ans).all()
    assert (testVec[:, :, 1] == 0).all()
    assert (testVec[:, :, 2] == ans).all()
    assert (testVec[:, :, 3] == 0).all()
    assert (testVec[:, :, 4] == 0).all()

  testVec = np.zeros((10, 2, 3))
  # Now test for a user-provided `saveInterval`
  # write to entry 0
  testVec = PTfxn.interval_save(velVec, 0, testVec, interval=42)
  assert (testVec[:, :, 0] == ans).all()
  # write to entry 1
  testVec = PTfxn.interval_save(velVec, 42, testVec, interval=42)
  assert (testVec[:, :, 1] == ans).all()
  # no write
  testVec = PTfxn.interval_save(velVec, 21, testVec, interval=42)
  assert (testVec[:, :, 0] == ans).all()
  assert (testVec[:, :, 1] == ans).all()
  assert (testVec[:, :, 2] == 0).all()


# TODO: test convergence rate?
def test_advect_tracer():
  X0 = [[0, 0],
        [1, 1]]
  vel = [0, 1] * np.ones(np.shape(X0))
  ans = [[0, 1],
         [1, 2]]
  testX = PTfxn.advect_tracer(X0, vel, 1)
  assert (testX == ans).all()

  X0 = np.array([[0, 0], [1, 1]])
  vel = np.array([[-1, 0.8], [3, -2.4]])
  ans = [[-0.5, 0.4],
         [2.5, -0.2]]
  testX = PTfxn.advect_tracer(X0, vel, 0.5)
  assert (testX - ans < 1e-12).all()


# NOTE: slightly vacuous, given that the function is the same as above,
# but that may not always be the case.
def test_move_emitter():
  X0 = [[0, 0],
        [1, 1]]
  vel = [0, 1] * np.ones(np.shape(X0))
  testX = PTfxn.move_emitter(X0, vel, 1)
  assert (testX == [[0, 1],
                    [1, 2]]).all()

  X0 = np.array([[0, 0], [1, 1]])
  vel = np.array([[-1, 0.8], [3, -2.4]])
  ans = [[-0.5, 0.4],
         [2.5, -0.2]]
  testX = PTfxn.move_emitter(X0, vel, 0.5)
  assert (testX - ans < 1e-12).all()
