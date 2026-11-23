"""Co-simulation protocol 2: the extended RPCs behind Tulpa's std/mujoco cosim backend."""
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from physics_engine.engine import PhysicsEngine  # noqa: E402
from tulpa_bridge import toolchain  # noqa: E402
from tulpa_bridge.cosim_server import CAPABILITIES, CoSimServer, CoSimSession  # noqa: E402

PENDULUM = """<mujoco model="pendulum">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01" range="-180 180" limited="true"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
      <site name="tip" pos="0 0 -0.5"/>
    </body>
  </worldbody>
  <actuator><motor name="torque" joint="hinge" ctrlrange="-2 2" ctrllimited="true"/></actuator>
  <sensor><jointpos name="angle" joint="hinge"/><jointvel name="rate" joint="hinge"/></sensor>
  <keyframe><key name="up" qpos="3.14159"/><key name="side" qpos="1.5708"/></keyframe>
</mujoco>"""


@pytest.fixture
def session():
    e = PhysicsEngine()
    e.load_model_from_xml(PENDULUM)
    return CoSimSession(e)


def test_capabilities_and_info(session):
    assert session.handle("capabilities", {}) == CAPABILITIES
    info = session.handle("info", {})
    assert info["protocol"] == 2 and info["envs"] == 1


def test_get_set_fields(session):
    session.handle("reset", {"keyframe": "side"})
    assert session.handle("get", {"field": "qpos"}) == pytest.approx([1.5708])
    session.handle("set", {"field": "qpos", "values": [0.3]})
    assert session.handle("get", {"field": "qpos"}) == pytest.approx([0.3])
    assert session.handle("get", {"field": "jnt_type"}) == [3]
    with pytest.raises(ValueError, match="has 1 values"):
        session.handle("set", {"field": "qpos", "values": [1, 2]})
    with pytest.raises(ValueError, match="unknown field"):
        session.handle("get", {"field": "nope"})
    with pytest.raises(ValueError, match="finite"):
        session.handle("set", {"field": "qpos", "values": [float("nan")]})


def test_state_restore_is_exact(session):
    session.handle("reset", {"keyframe": "side"})
    session.handle("step", {"ctrl": [0.5], "substeps": 20})
    snap = session.handle("state", {})
    a = session.handle("step", {"substeps": 100})["qpos"]
    session.handle("restore", {"state": snap})
    b = session.handle("step", {"substeps": 100})["qpos"]
    assert a == b


def test_options(session):
    session.handle("set_option", {"timestep": 0.001, "gravity": [0, 0, -1.62], "integrator": 1})
    o = session.handle("option", {})
    assert o["timestep"] == 0.001 and o["gravity"] == [0, 0, -1.62] and o["integrator"] == 1
    with pytest.raises(ValueError, match="unknown option"):
        session.handle("set_option", {"warp": 2})
    with pytest.raises(ValueError, match="positive"):
        session.handle("set_option", {"timestep": 0})


def test_dynamics_queries(session):
    session.handle("set", {"field": "qpos", "values": [0.0]})
    J = session.handle("jacobian", {"kind": "site", "name": "tip"})
    assert J["jacp"][0][0] == pytest.approx(-0.5) and J["jacr"][1][0] == pytest.approx(1.0)
    M = session.handle("mass_matrix", {})
    assert 0.08 < M[0][0] < 0.09
    scene = session.handle("scene", {})
    assert [g["type"] for g in scene] == ["plane", "capsule"]
    assert session.handle("contacts", {}) == []


NEEDS_TULPA = pytest.mark.skipif(toolchain.find() is None, reason="tulpa toolchain not found (set TULPA_BIN)")

PROGRAM = """import "std/mujoco" as mj
let sim = mj.open(backend: "cosim")
let hinge = sim.joint("hinge")
let motor = sim.actuator("torque")
sim.reset("side")
while sim.time() < 3[s] {
    motor.command(-20.0 * (hinge.pos() - 45[deg]) / 1[rad] - 2.0 * hinge.vel() / 1[rad/s])
    sim.step(5)
}
print(sim.supports("jacobian"), round(hinge.pos().in("deg"), 1), motor.saturations)
let snap = sim.snapshot()
sim.step(100)
sim.restore(snap)
print(sim.time() == snap.time * 1[s], round(sim.mass_matrix()[0][0], 4))
"""


@NEEDS_TULPA
def test_tulpa_std_mujoco_over_cosim(tmp_path):
    """std/mujoco gives the same answer over the co-simulator as the native backend (40.4°)."""
    tc = toolchain.find()
    e = PhysicsEngine()
    e.load_model_from_xml(PENDULUM)
    srv = CoSimServer(CoSimSession(e)).start()
    try:
        prog = tmp_path / "pd.rav"
        prog.write_text(PROGRAM)
        r = tc.run(str(prog), cwd=str(tmp_path), extra_env={"TULPA_COSIM": srv.address}, timeout=120)
    finally:
        srv.stop()
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "true 40.4 65"
    assert lines[1] == "true 0.0857"
