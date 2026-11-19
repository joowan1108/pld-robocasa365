"""Online stage predicates + monotone stage machine for per-stage specialists.

Paper (§4.4, YAM GPU insertion): long-horizon tasks are decomposed into stages, each with a
sparse stage-success reward, coordinated by a state machine. In simulation we replace the learned
reward classifier with predicates on privileged sim state.

Stage index semantics: `stage` = number of completed stages (0..K). Stage k is "active" while
stage == k; k == K means the whole task succeeded. The index is latched (monotone): it is the max
over time of the number of *leading* predicates that hold at the current step. Predicates of later
stages include the conditions the env's `_check_success` needs from earlier stages, so a stage can
only be completed in a state consistent with the final success condition.
"""

from __future__ import annotations

from typing import Callable, Sequence

# NOTE: robocasa imports are deferred into the predicate bodies so this module (and its stage
# machine) can be unit-tested without robocasa/mujoco installed.


def _ou():
    import robocasa.utils.object_utils as OU  # noqa: N812

    return OU


# --------------------------------------------------------------------------- CoffeeSetupMug
def env_success(env) -> bool:
    return bool(env._check_success())


# --------------------------------------------------------------------------- GetToastedBread
def toast_press(env) -> bool:
    # `toaster_on` is latched inside GetToastedBread._check_success (bread in slot AND slot turned on).
    # Our worker calls _check_success every step, so the latch is current.
    return bool(getattr(env, "toaster_on", False))


def toast_wait(env) -> bool:
    # toaster was turned on (latched) and no slot is currently in its 500-step forced-on phase,
    # i.e. the lever popped back up.
    if not toast_press(env):
        return False
    state = env.toaster.get_state(env)
    return not any(bool(s["turned_on"]) for s in state.values())


def toast_pick(env) -> bool:
    if not toast_wait(env):
        return False
    OU = _ou()
    grasped = OU.check_obj_grasped(env, "obj")
    in_slot = bool(env.toaster.check_slot_contact(env, "obj"))
    return bool(grasped and not in_slot)


def toast_place(env) -> bool:
    return bool(env._check_success())


# --------------------------------------------------------------------------- SteamInMicrowave
def steam_veg_in_bowl(env) -> bool:
    return bool(_ou().check_obj_in_receptacle(env, "vegetable", "bowl"))


def steam_bowl_in_microwave(env) -> bool:
    return steam_veg_in_bowl(env) and bool(_ou().obj_inside_of(env, "bowl", env.microwave))


def steam_door_closed(env) -> bool:
    return steam_bowl_in_microwave(env) and bool(env.microwave.is_closed(env=env))


def steam_turned_on(env) -> bool:
    return bool(env._check_success())


PREDICATES: dict[str, dict[str, Callable]] = {
    "CoffeeSetupMug": {"success": env_success},
    "GetToastedBread": {
        "press": toast_press,
        "wait": toast_wait,
        "pick": toast_pick,
        "place": toast_place,
    },
    "SteamInMicrowave": {
        "veg_in_bowl": steam_veg_in_bowl,
        "bowl_in_microwave": steam_bowl_in_microwave,
        "door_closed": steam_door_closed,
        "turned_on": steam_turned_on,
    },
}


class StageMachine:
    """Monotone stage tracker. `update(env)` must be called once per env step (after stepping)."""

    def __init__(self, task: str, stage_names: Sequence[str], predicates: dict[str, Callable] | None = None):
        if predicates is not None:
            preds = predicates
        elif task in PREDICATES:
            preds = PREDICATES[task]
        elif list(stage_names) == ["success"]:
            preds = {"success": env_success}  # any other task (e.g. forgetting-check eval)
        else:
            raise KeyError(f"No stage predicates registered for {task}")
        missing = [s for s in stage_names if s not in preds]
        if missing:
            raise KeyError(f"No predicate for stages {missing} of task {task}")
        self.task = task
        self.stage_names = list(stage_names)
        self._preds = [preds[s] for s in stage_names]
        self.stage = 0

    @property
    def num_stages(self) -> int:
        return len(self.stage_names)

    def reset(self, env=None) -> int:
        self.stage = 0
        if env is not None:
            self.stage = self._leading_true(env)
        return self.stage

    def _leading_true(self, env) -> int:
        n = 0
        for p in self._preds:
            if not p(env):
                break
            n += 1
        return n

    def update(self, env) -> int:
        self.stage = max(self.stage, self._leading_true(env))
        return self.stage
