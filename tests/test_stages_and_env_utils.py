import numpy as np

from pld_rc365.envs.rc365_env import DATASET_STATE_ORDER, state_vector16, threshold_discrete
from pld_rc365.envs.stages import StageMachine
from pld_rc365.sft.lerobot_reader import ACTION_TO_ENV


class Flags:
    def __init__(self):
        self.v = {}


def test_stage_machine_is_sequential_and_monotone():
    env = Flags()
    preds = {n: (lambda e, n=n: e.v.get(n, False)) for n in ("a", "b", "c")}
    sm = StageMachine("X", ["a", "b", "c"], preds)
    assert sm.reset(env) == 0
    env.v["b"] = True
    assert sm.update(env) == 0           # b before a does not count
    env.v["a"] = True
    assert sm.update(env) == 2           # a then b in the same step
    env.v["a"] = False                   # e.g. gripper released after the pick stage
    assert sm.update(env) == 2           # latched
    env.v["c"] = True                    # final stage reached while an earlier predicate is false again
    assert sm.update(env) == 3


def test_toasted_bread_success_reaches_last_stage():
    # pick (grasped) is false at success time (bread on plate, gripper away); place must still count
    env = Flags()
    preds = {n: (lambda e, n=n: e.v.get(n, False)) for n in ("press", "wait", "pick", "place")}
    sm = StageMachine("GetToastedBread", ["press", "wait", "pick", "place"], preds)
    sm.reset(env)
    for k in ("press", "wait", "pick"):
        env.v[k] = True
        sm.update(env)
    assert sm.stage == 3
    env.v.update(pick=False, place=True)
    assert sm.update(env) == 4


def test_unknown_task_success_only():
    sm = StageMachine("CloseFridge", ["success"])
    assert sm.num_stages == 1


def test_threshold_and_state_layout():
    a = np.zeros(12)
    a[6], a[11] = 0.49, 0.5
    out = threshold_discrete(a)
    assert out[6] == -1 and out[11] == 1
    st = {k: np.arange(n) + 10 * j for j, (k, n) in enumerate(DATASET_STATE_ORDER)}
    v = state_vector16(st)
    assert v.shape == (16,) and v[0] == 0 and v[3] == 10 and v[14] == 40


def test_lerobot_action_reorder_roundtrip():
    # env order: eef_pos 0:3, eef_rot 3:6, gripper 6, base_motion 7:11, control_mode 11
    env_a = np.arange(12, dtype=np.float64)
    # lerobot order (robocasa reorder_hdf5_action): base_motion 0:4, mode 4, eef_pos 5:8, eef_rot 8:11, gripper 11
    ler = np.concatenate([env_a[7:11], env_a[11:12], env_a[0:3], env_a[3:6], env_a[6:7]])
    np.testing.assert_array_equal(ler[ACTION_TO_ENV], env_a)
