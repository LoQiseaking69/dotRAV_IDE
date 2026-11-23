"""Robotics tools: model inspection, validation, controls, physics, snapshots, recording,
experiment browsing, and the Robotics panel in the IDE window."""
import csv
import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from physics_engine import robotics as rb  # noqa: E402
from physics_engine.engine import PhysicsEngine  # noqa: E402

ARM = """<mujoco model="arm">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="upper" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90" limited="true" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".03" mass="1"/>
      <body name="lower" pos="0 0 -.4">
        <joint name="slider" type="slide" axis="0 0 1" range="-0.1 0.1" limited="true"/>
        <geom type="box" size=".03 .03 .1" mass=".5"/>
        <site name="tool" pos="0 0 -.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-5 5" ctrllimited="true"/>
    <motor name="free_motor" joint="slider"/>
  </actuator>
  <sensor><jointpos name="q" joint="shoulder"/><framepos name="tool_pos" objtype="site" objname="tool"/></sensor>
</mujoco>"""


@pytest.fixture
def eng():
    e = PhysicsEngine()
    e.load_model_from_xml(ARM)
    return e


def test_inspect(eng):
    info = rb.inspect_model(eng)
    assert [j["name"] for j in info["joints"]] == ["shoulder", "slider"]
    sh = info["joints"][0]
    assert sh["type"] == "hinge" and sh["unit"] == "rad" and sh["limited"]
    assert sh["range"] == pytest.approx([-np.pi / 2, np.pi / 2])
    assert info["joints"][1]["unit"] == "m"
    a = info["actuators"][0]
    assert a["target"] == "shoulder" and a["ctrlrange"] == [-5, 5]
    assert [s["dim"] for s in info["sensors"]] == [1, 3]
    assert info["sites"] == [{"name": "tool", "body": "lower"}]


def test_validate_reports_robotics_problems(eng):
    res = rb.validate(eng)
    msgs = [m for _, m in res]
    assert any("free_motor" in m and "no ctrlrange" in m for m in msgs)
    assert not any(sev == "error" for sev, _ in res)


def test_controls_are_checked_and_clamped(eng):
    rb.set_joint(eng, "shoulder", 0.5)
    assert eng.data.qpos[0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="outside its range"):
        rb.set_joint(eng, "shoulder", 3.0)
    with pytest.raises(KeyError):
        rb.set_joint(eng, "elbow", 0.0)
    assert rb.set_control(eng, "shoulder_motor", 12.0) == 5.0
    assert rb.set_control(eng, "free_motor", 12.0) == 12.0


def test_physics_configuration(eng):
    rb.configure(eng, timestep=0.001, gravity=[0, 0, -1.62], integrator="RK4", iterations=50)
    p = rb.physics(eng)
    assert p["timestep"] == 0.001 and p["gravity"] == [0, 0, -1.62] and p["integrator"] == "RK4"
    for bad, exc in [({"timestep": 0}, ValueError), ({"integrator": "verlet"}, ValueError), ({"warp": 1}, KeyError)]:
        with pytest.raises(exc):
            rb.configure(eng, **bad)


def test_snapshots_restore_exactly(eng):
    rb.set_control(eng, "shoulder_motor", 2.0)
    for _ in range(50):
        eng.step()
    s = rb.snapshot(eng)
    for _ in range(100):
        eng.step()
    a = eng.data.qpos.copy()
    rb.restore(eng, s)
    assert eng.data.time == pytest.approx(s["time"])
    for _ in range(100):
        eng.step()
    assert np.array_equal(eng.data.qpos, a)


def test_recording_replay_and_export(eng, tmp_path):
    rec = rb.Recording()
    rb.set_control(eng, "shoulder_motor", 3.0)
    for _ in range(20):
        for _ in range(5):
            eng.step()
        rec.capture(eng)
    assert len(rec) == 20
    t = rec.replay(eng, 4)
    assert eng.data.time == pytest.approx(t) == pytest.approx(rec.frames[4]["row"][0])
    n = rec.export_csv(str(tmp_path / "traj.csv"))
    rows = list(csv.reader(open(tmp_path / "traj.csv")))
    assert n == 20 and len(rows) == 21
    assert rows[0][:3] == ["time", "qpos[0]", "qpos[1]"] and "ctrl:shoulder_motor" in rows[0]
    assert "sensor:tool_pos[2]" in rows[0]


def test_experiment_browser(tmp_path):
    run = tmp_path / "runs" / "pend-sac"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"algo": "sac", "seed": 3}))
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 1000, "q_loss": 2.0, "train_return": -900}, {"step": 2000, "eval_return": -700},
        {"step": 4000, "eval_return": -150}]) + "\n{\"step\": 5")  # last line still being written
    (run / "checkpoints" / "step-4000.json").write_text("{}")
    (run / "summary.json").write_text(json.dumps({"algo": "sac", "steps": 4000}))
    ex = rb.experiments(str(tmp_path / "runs"))
    assert ex == [{"name": "pend-sac", "dir": str(run), "algo": "sac", "seed": 3, "steps": 4000,
                   "best_eval": -150, "last_eval": -150, "finished": True, "checkpoints": 1}]
    assert rb.metric_series(rb.load_metrics(str(run)), "eval_return") == ([2000.0, 4000.0], [-700.0, -150.0])


def test_panel_in_ide_window(qtbot, tmp_path):
    from gui.ide.window import IdeWindow
    w = IdeWindow()
    qtbot.addWidget(w)
    w.simulation.engine.load_model_from_xml(ARM)
    w.simulation.model_loaded.emit({})
    p = w.robotics
    tree = {p.tree.topLevelItem(i).text(0) for i in range(p.tree.topLevelItemCount())}
    assert "Joints (2)" in tree and "Actuators (2)" in tree
    assert set(p.joint_sliders) == {"shoulder", "slider"} and set(p.act_sliders) == {"shoulder_motor", "free_motor"}
    p.act_sliders["shoulder_motor"].setValue(1000)
    assert w.simulation.engine.data.ctrl[0] == pytest.approx(5.0)
    res = p.validate()
    assert p.findings.count() == len(res) > 0
    p.dt.setValue(0.001)
    p.apply_physics()
    assert w.simulation.engine.model.opt.timestep == pytest.approx(0.001)
    p.take_snapshot()
    assert p.snap_list.count() == 1
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "config.json").write_text('{"algo": "td3"}')
    p.runs_dir.setText(str(tmp_path / "runs"))
    assert p.load_experiments()[0]["algo"] == "td3" and p.exp_table.rowCount() == 1
    assert any(a.text() == "Validate Model" for m in w.menuBar().actions() if m.menu() for a in m.menu().actions())
