"""Robotics development tools (headless core of the IDE's Robotics panel).

Everything here works on a :class:`PhysicsEngine` (the IDE's MuJoCo instance) and is safe to
call from the GUI thread while the simulation runs: every access takes the engine lock.

* model inspection: joints, actuators, sensors, bodies, sites with their parameters
* validation: MuJoCo compile warnings plus robotics lint (limits, actuator ranges, masses)
* direct control: set joint positions, stage actuator controls
* physics configuration: timestep, gravity, integrator, solver iterations
* snapshots (exact integration state) and trajectory recording with replay and CSV export
* RL experiment browsing (Tulpa's ``rl.Experiment`` run directories)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

_JNT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
_TRN = {0: "joint", 1: "joint (parent frame)", 2: "slider-crank", 3: "tendon", 4: "site", 5: "body"}
_INTEGRATORS = ["Euler", "RK4", "implicit", "implicitfast"]


def _name(m, obj, i, prefix):
    return mujoco.mj_id2name(m, obj, i) or f"{prefix}{i}"


def inspect_model(engine) -> dict[str, list[dict[str, Any]]]:
    """Structured description of the loaded model (for the model tree)."""
    O = mujoco.mjtObj
    with engine._lock:
        m = engine.model
        if m is None:
            return {}
        joints = []
        for j in range(m.njnt):
            t = int(m.jnt_type[j])
            unit = "rad" if t == 3 else "m" if t == 2 else ""
            joints.append({
                "name": _name(m, O.mjOBJ_JOINT, j, "joint"), "type": _JNT.get(t, "?"), "unit": unit,
                "limited": bool(m.jnt_limited[j]), "range": m.jnt_range[j].tolist(),
                "qposadr": int(m.jnt_qposadr[j]), "dofadr": int(m.jnt_dofadr[j]),
                "body": _name(m, O.mjOBJ_BODY, int(m.jnt_bodyid[j]), "body"),
                "damping": float(m.dof_damping[m.jnt_dofadr[j]]) if t in (2, 3) else None,
            })
        acts = []
        for a in range(m.nu):
            trn = int(m.actuator_trntype[a])
            tid = int(m.actuator_trnid[a, 0])
            target = _name(m, O.mjOBJ_JOINT, tid, "joint") if trn in (0, 1) and tid >= 0 else None
            acts.append({
                "name": _name(m, O.mjOBJ_ACTUATOR, a, "actuator"), "transmission": _TRN.get(trn, "?"),
                "target": target, "limited": bool(m.actuator_ctrllimited[a]),
                "ctrlrange": m.actuator_ctrlrange[a].tolist(), "gear": float(m.actuator_gear[a, 0]),
                "forcerange": m.actuator_forcerange[a].tolist(),
            })
        sensors = [{
            "name": _name(m, O.mjOBJ_SENSOR, s, "sensor"), "type": int(m.sensor_type[s]),
            "dim": int(m.sensor_dim[s]), "adr": int(m.sensor_adr[s]), "noise": float(m.sensor_noise[s]),
        } for s in range(m.nsensor)]
        bodies = [{
            "name": _name(m, O.mjOBJ_BODY, b, "body"), "mass": float(m.body_mass[b]),
            "parent": _name(m, O.mjOBJ_BODY, int(m.body_parentid[b]), "body") if b else None,
        } for b in range(m.nbody)]
        sites = [{"name": _name(m, O.mjOBJ_SITE, s, "site"),
                  "body": _name(m, O.mjOBJ_BODY, int(m.site_bodyid[s]), "body")} for s in range(m.nsite)]
        return {"joints": joints, "actuators": acts, "sensors": sensors, "bodies": bodies, "sites": sites}


def validate(engine) -> list[tuple[str, str]]:
    """(severity, message) findings for the loaded model."""
    out: list[tuple[str, str]] = []
    info = inspect_model(engine)
    if not info:
        return [("error", "no model loaded")]
    with engine._lock:
        m, d = engine.model, engine.data
        mujoco.mj_forward(m, d)
        for w in range(len(d.warning)):
            if d.warning[w].number:
                out.append(("warning", f"MuJoCo warning {mujoco.mjtWarning(w).name}: {d.warning[w].number}x"))
        if m.opt.timestep > 0.01:
            out.append(("warning", f"timestep {m.opt.timestep} s is large; contacts and stiff joints may be unstable"))
    for j in info["joints"]:
        if j["limited"] and j["range"][0] >= j["range"][1]:
            out.append(("error", f"joint '{j['name']}': empty range {j['range']}"))
        if j["type"] in ("hinge", "slide") and not j["limited"]:
            out.append(("info", f"joint '{j['name']}' is unlimited"))
    for a in info["actuators"]:
        if a["limited"] and a["ctrlrange"][0] >= a["ctrlrange"][1]:
            out.append(("error", f"actuator '{a['name']}': empty ctrlrange {a['ctrlrange']}"))
        if not a["limited"]:
            out.append(("warning", f"actuator '{a['name']}' has no ctrlrange: commands are unbounded"))
        if a["gear"] == 0:
            out.append(("error", f"actuator '{a['name']}' has zero gear"))
    for b in info["bodies"][1:]:
        if b["mass"] <= 0:
            out.append(("warning", f"body '{b['name']}' has no mass"))
    dup = [n for kind in ("joints", "actuators", "sensors", "bodies") for n in {x["name"] for x in info[kind]} if
           [x["name"] for x in info[kind]].count(n) > 1]
    for n in dup:
        out.append(("error", f"duplicate name '{n}'"))
    return out


def set_joint(engine, name: str, value: float) -> None:
    """Teleports a hinge/slide joint (kinematic, velocity zeroed); range-checked if limited."""
    with engine._lock:
        m, d = engine.model, engine.data
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise KeyError(f"no joint '{name}'")
        if int(m.jnt_type[j]) not in (2, 3):
            raise ValueError(f"joint '{name}' is a {_JNT[int(m.jnt_type[j])]} joint")
        lo, hi = m.jnt_range[j]
        if m.jnt_limited[j] and not lo <= value <= hi:
            raise ValueError(f"joint '{name}': {value} outside its range [{lo}, {hi}]")
        d.qpos[m.jnt_qposadr[j]] = value
        d.qvel[m.jnt_dofadr[j]] = 0.0
        mujoco.mj_forward(m, d)


def set_control(engine, name: str, value: float) -> float:
    """Stages an actuator control (clamped to ctrlrange when limited). Returns the applied value."""
    with engine._lock:
        m, d = engine.model, engine.data
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a < 0:
            raise KeyError(f"no actuator '{name}'")
        if m.actuator_ctrllimited[a]:
            value = float(np.clip(value, *m.actuator_ctrlrange[a]))
        d.ctrl[a] = value
        return value


def physics(engine) -> dict[str, Any]:
    with engine._lock:
        o = engine.model.opt
        return {"timestep": float(o.timestep), "gravity": o.gravity.tolist(), "integrator": _INTEGRATORS[int(o.integrator)],
                "iterations": int(o.iterations), "tolerance": float(o.tolerance), "density": float(o.density),
                "viscosity": float(o.viscosity), "wind": o.wind.tolist()}


def configure(engine, **kw) -> None:
    """Changes physics options; unknown keys and invalid values are errors."""
    with engine._lock:
        o = engine.model.opt
        for k, v in kw.items():
            if k == "timestep":
                if not v > 0:
                    raise ValueError("timestep must be positive")
                o.timestep = float(v)
            elif k in ("gravity", "wind"):
                if len(v) != 3:
                    raise ValueError(f"{k} needs 3 components")
                getattr(o, k)[:] = v
            elif k == "integrator":
                if v not in _INTEGRATORS:
                    raise ValueError(f"integrator must be one of {_INTEGRATORS}")
                o.integrator = _INTEGRATORS.index(v)
            elif k in ("iterations",):
                o.iterations = int(v)
            elif k in ("tolerance", "density", "viscosity"):
                setattr(o, k, float(v))
            else:
                raise KeyError(f"unknown physics option '{k}'")


def snapshot(engine) -> dict[str, Any]:
    """Exact integration state (time, qpos, qvel, act, ctrl, mocap, ...)."""
    with engine._lock:
        sig = mujoco.mjtState.mjSTATE_INTEGRATION
        s = np.zeros(mujoco.mj_stateSize(engine.model, sig))
        mujoco.mj_getState(engine.model, engine.data, s, sig)
        return {"time": float(engine.data.time), "state": s}


def restore(engine, snap: dict[str, Any]) -> None:
    with engine._lock:
        mujoco.mj_setState(engine.model, engine.data, np.asarray(snap["state"]), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(engine.model, engine.data)


@dataclass
class Recording:
    """A trajectory recorded frame by frame; replay restores exact states."""
    columns: list[str] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)

    def capture(self, engine) -> None:
        with engine._lock:
            m, d = engine.model, engine.data
            if not self.columns:
                O = mujoco.mjtObj
                self.columns = (["time"] + [f"qpos[{i}]" for i in range(m.nq)] + [f"qvel[{i}]" for i in range(m.nv)]
                                + [f"ctrl:{_name(m, O.mjOBJ_ACTUATOR, a, 'actuator')}" for a in range(m.nu)]
                                + [f"sensor:{_name(m, O.mjOBJ_SENSOR, s, 'sensor')}[{k}]"
                                   for s in range(m.nsensor) for k in range(int(m.sensor_dim[s]))])
            row = [float(d.time)] + d.qpos.tolist() + d.qvel.tolist() + d.ctrl.tolist() + d.sensordata.tolist()
        self.frames.append({"row": row, "snap": snapshot(engine)})

    def __len__(self) -> int:
        return len(self.frames)

    def replay(self, engine, i: int) -> float:
        f = self.frames[max(0, min(i, len(self.frames) - 1))]
        restore(engine, f["snap"])
        return f["snap"]["time"]

    def export_csv(self, path: str) -> int:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.columns)
            for f in self.frames:
                w.writerow(f["row"])
        return len(self.frames)


def experiments(root: str) -> list[dict[str, Any]]:
    """Tulpa rl.Experiment runs under `root` (each has config.json and metrics.jsonl)."""
    out = []
    base = Path(root)
    if not base.is_dir():
        return out
    for d in sorted(p for p in base.iterdir() if (p / "config.json").exists()):
        cfg = json.loads((d / "config.json").read_text() or "{}")
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        rows = load_metrics(str(d))
        evals = [r["eval_return"] for r in rows if "eval_return" in r]
        out.append({"name": d.name, "dir": str(d), "algo": cfg.get("algo", summ.get("algo", "?")), "seed": cfg.get("seed"),
                    "steps": summ.get("steps", max((r.get("step", 0) for r in rows), default=0)),
                    "best_eval": max(evals) if evals else None, "last_eval": evals[-1] if evals else None,
                    "finished": bool(summ), "checkpoints": len(list((d / "checkpoints").glob("*.json"))) if (d / "checkpoints").exists() else 0})
    return out


def load_metrics(run_dir: str) -> list[dict[str, Any]]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a run still writing its last line
    return rows


def metric_series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r and isinstance(r[key], (int, float)):
            xs.append(float(r.get("step", len(xs))))
            ys.append(float(r[key]))
    return xs, ys
