"""Co-simulation server: lets a Tulpa program drive a MuJoCo model in lock-step.

Protocol (one JSON object per line, UTF-8, over TCP):

    request       {"id": 7, "method": "step", "params": {"ctrl": [...], "substeps": 5}}
    response      {"id": 7, "result": {...}}            or  {"id": 7, "error": "message"}
    notification  {"method": "plot", "params": {...}}   (no id, no response)

Methods: info, reset, observe, step, set_ctrl, set_state, apply_force, plus the extended set
used by Tulpa's ``std/mujoco`` (protocol 2): capabilities, get, set, state, restore, option,
set_option, jacobian, mass_matrix, inverse, contacts, scene, time, warnings; plus any method
registered with :meth:`CoSimServer.register`. The Tulpa side is ``cosim::connect()``.
"""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import ssl
import threading
import time
from typing import Any, Callable, Optional

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

_JNT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
_GEOM_TYPES = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"]

PROTOCOL = 2
# Capability names shared with Tulpa's std/mujoco (Sim.supports).
CAPABILITIES = ["step", "observe", "reset", "set_state", "apply_force", "fields", "snapshot",
                "jacobian", "mass_matrix", "contacts", "scene", "options", "inverse_dynamics"]

# Fields readable/writable through get/set (mjData first, then mjModel parameters).
_DATA_FIELDS = {"qpos", "qvel", "qacc", "ctrl", "act", "sensordata", "qfrc_applied", "xfrc_applied",
                "qfrc_actuator", "qfrc_bias", "qfrc_passive", "qfrc_constraint", "qfrc_inverse",
                "actuator_force", "actuator_length", "actuator_velocity", "xpos", "xquat", "xmat",
                "xipos", "subtree_com", "cvel", "cacc", "geom_xpos", "geom_xmat", "site_xpos",
                "site_xmat", "mocap_pos", "mocap_quat", "energy"}
_MODEL_FIELDS = {"qpos0", "body_mass", "body_inertia", "body_pos", "body_quat", "jnt_range",
                 "jnt_stiffness", "dof_damping", "dof_armature", "dof_frictionloss", "geom_size",
                 "geom_friction", "geom_pos", "actuator_ctrlrange", "actuator_forcerange",
                 "actuator_gear", "actuator_gainprm", "actuator_biasprm", "sensor_noise",
                 "key_qpos", "key_ctrl"}
_INT_FIELDS = {"jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "jnt_limited", "body_parentid",
               "body_jntnum", "geom_type", "geom_bodyid", "site_bodyid", "actuator_trntype",
               "actuator_trnid", "actuator_ctrllimited", "actuator_dyntype", "sensor_type",
               "sensor_objid", "sensor_adr", "sensor_dim"}
_OPTS = ["timestep", "density", "viscosity", "impratio", "tolerance", "iterations", "integrator",
         "cone", "solver"]


_NO = object()


def _name(model, objtype, i: int, prefix: str) -> str:
    return mujoco.mj_id2name(model, objtype, i) or f"{prefix}{i}"


class CoSimSession:
    """Dispatches RPCs onto a :class:`physics_engine.engine.PhysicsEngine`."""

    def __init__(self, engine, realtime: float = 0.0,
                 on_frame: Optional[Callable[[dict], None]] = None,
                 on_notify: Optional[Callable[[str, dict], None]] = None,
                 frame_interval: float = 1 / 30) -> None:
        self.engine = engine
        self.realtime = realtime          # 0 = as fast as possible, 1 = real time, 2 = 2× ...
        self.on_frame = on_frame
        self.on_notify = on_notify
        self.frame_interval = frame_interval
        self._last_frame = 0.0
        self._wall0: Optional[float] = None
        self._sim0 = 0.0
        self.steps = 0
        self.extra: dict[str, Callable[[dict], Any]] = {}
        self.paused = threading.Event()
        self.paused.set()                 # set = running
        self.disturbances: list[dict] = []

    # ------------------------------------------------------------------ helpers
    @property
    def m(self):
        return self.engine.model

    @property
    def d(self):
        return self.engine.data

    def _info(self) -> dict:
        m = self.m
        O = mujoco.mjtObj
        acts = []
        for i in range(m.nu):
            j = int(m.actuator_trnid[i, 0])
            acts.append({
                "name": _name(m, O.mjOBJ_ACTUATOR, i, "act"),
                "joint": _name(m, O.mjOBJ_JOINT, j, "joint") if j >= 0 else None,
                "ctrlrange": [float(x) for x in m.actuator_ctrlrange[i]],
                "gear": float(m.actuator_gear[i, 0]),
                "qposadr": int(m.jnt_qposadr[j]) if j >= 0 else None,
                "dofadr": int(m.jnt_dofadr[j]) if j >= 0 else None,
            })
        joints = [{
            "name": _name(m, O.mjOBJ_JOINT, j, "joint"),
            "type": _JNT_TYPES.get(int(m.jnt_type[j]), "?"),
            "range": [float(x) for x in m.jnt_range[j]],
            "qposadr": int(m.jnt_qposadr[j]),
            "dofadr": int(m.jnt_dofadr[j]),
        } for j in range(m.njnt)]
        sensors = [{
            "name": _name(m, O.mjOBJ_SENSOR, s, "sensor"),
            "type": int(m.sensor_type[s]),
            "dim": int(m.sensor_dim[s]),
        } for s in range(m.nsensor)]
        keys = [_name(m, O.mjOBJ_KEY, k, "key") for k in range(m.nkey)]
        return {
            "model": mujoco.mj_id2name(m, O.mjOBJ_MODEL, 0) if hasattr(O, "mjOBJ_MODEL") else "",
            "timestep": float(m.opt.timestep),
            "gravity": [float(x) for x in m.opt.gravity],
            "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
            "total_mass": float(sum(m.body_mass)),
            "actuators": acts, "joints": joints, "sensors": sensors,
            "bodies": [_name(m, O.mjOBJ_BODY, b, "body") for b in range(m.nbody)],
            "keyframes": keys,
            "protocol": PROTOCOL,
            "envs": 1,
        }

    def observe(self) -> dict:
        m, d = self.m, self.d
        O = mujoco.mjtObj
        mujoco.mj_subtreeVel(m, d)
        sensors = {}
        for s in range(m.nsensor):
            a, n = int(m.sensor_adr[s]), int(m.sensor_dim[s])
            sensors[_name(m, O.mjOBJ_SENSOR, s, "sensor")] = [float(x) for x in d.sensordata[a:a + n]]
        # per-body contact forces with the world (normal force + centre of pressure)
        contacts: dict[str, dict] = {}
        f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if b1 != 0 and b2 != 0:
                continue
            body = b2 if b1 == 0 else b1
            mujoco.mj_contactForce(m, d, c, f6)
            fn = abs(float(f6[0]))
            name = _name(m, O.mjOBJ_BODY, body, "body")
            e = contacts.setdefault(name, {"force": 0.0, "cop": [0.0, 0.0, 0.0], "n": 0})
            e["force"] += fn
            for k in range(3):
                e["cop"][k] += fn * float(con.pos[k])
            e["n"] += 1
        for e in contacts.values():
            if e["force"] > 1e-9:
                e["cop"] = [x / e["force"] for x in e["cop"]]
        root = 1 if m.nbody > 1 else 0
        return {
            "time": float(d.time),
            "qpos": d.qpos.tolist(),
            "qvel": d.qvel.tolist(),
            "ctrl": d.ctrl.tolist(),
            "act_force": d.actuator_force.tolist(),
            "sensors": sensors,
            "com": d.subtree_com[root].tolist(),
            "com_vel": d.subtree_linvel[root].tolist(),
            "root_pos": d.xpos[root].tolist(),
            "root_quat": d.xquat[root].tolist(),
            "bodies": {_name(m, O.mjOBJ_BODY, b, "body"): d.xpos[b].tolist() for b in range(1, m.nbody)},
            "contacts": contacts,
            "energy": [float(d.energy[0]), float(d.energy[1])] if m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY else None,
        }

    def _pace(self) -> None:
        if self.realtime <= 0:
            return
        now = time.perf_counter()
        if self._wall0 is None:
            self._wall0, self._sim0 = now, float(self.d.time)
            return
        target = self._wall0 + (float(self.d.time) - self._sim0) / self.realtime
        if target > now:
            time.sleep(min(target - now, 0.25))

    def _maybe_frame(self, force: bool = False) -> None:
        if self.on_frame is None:
            return
        now = time.perf_counter()
        if force or now - self._last_frame >= self.frame_interval:
            self._last_frame = now
            try:
                self.on_frame({"time": float(self.d.time), "steps": self.steps})
            except Exception:  # GUI callbacks must never kill the RPC loop
                logger.debug("frame callback failed", exc_info=True)

    # ------------------------------------------------------------------ dispatch
    def handle(self, method: str, params: Any) -> Any:
        p = params if isinstance(params, dict) else {}
        eng = self.engine
        if method == "info":
            return self._info()
        if method == "reset":
            with eng._lock:
                key = p.get("keyframe")
                if isinstance(key, str):
                    k = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, key)
                    if k < 0:
                        raise ValueError(f"no keyframe named '{key}'")
                    mujoco.mj_resetDataKeyframe(self.m, self.d, k)
                elif isinstance(key, int):
                    mujoco.mj_resetDataKeyframe(self.m, self.d, key)
                elif self.m.nkey > 0:
                    mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
                else:
                    mujoco.mj_resetData(self.m, self.d)
                self.d.xfrc_applied[:] = 0
                mujoco.mj_forward(self.m, self.d)
                self._wall0 = None
            self._maybe_frame(force=True)
            return self.observe()
        if method == "observe":
            with eng._lock:
                return self.observe()
        if method in ("step", "set_ctrl"):
            ctrl = p.get("ctrl")
            with eng._lock:
                if ctrl is not None:
                    if len(ctrl) != self.m.nu:
                        raise ValueError(f"ctrl has {len(ctrl)} values, model has {self.m.nu} actuators")
                    lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
                    c = np.asarray(ctrl, dtype=float)
                    if not np.all(np.isfinite(c)):
                        raise ValueError("ctrl contains NaN/inf")
                    lim = self.m.actuator_ctrllimited.astype(bool)
                    c[lim] = np.clip(c[lim], lo[lim], hi[lim])
                    self.d.ctrl[:] = c
                if method == "set_ctrl":
                    return None
                n = max(1, int(p.get("substeps", 1)))
            for _ in range(n):
                self.paused.wait()
                with eng._lock:
                    self._apply_disturbances()
                    mujoco.mj_step(self.m, self.d)
                self.steps += 1
                self._pace()
            self._maybe_frame()
            with eng._lock:
                return self.observe()
        if method == "set_state":
            with eng._lock:
                if p.get("qpos") is not None:
                    self.d.qpos[:] = p["qpos"]
                if p.get("qvel") is not None:
                    self.d.qvel[:] = p["qvel"]
                mujoco.mj_forward(self.m, self.d)
                return self.observe()
        if method == "apply_force":
            body = p.get("body")
            with eng._lock:
                b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body or "")
                if b < 0:
                    raise ValueError(f"no body named '{body}'")
                self.d.xfrc_applied[b, :3] = p.get("force") or [0, 0, 0]
                self.d.xfrc_applied[b, 3:] = p.get("torque") or [0, 0, 0]
            return None
        ext = self._extended(method, p)
        if ext is not _NO:
            return ext
        if method in self.extra:
            return self.extra[method](p)
        raise ValueError(f"unknown method '{method}'")

    # ------------------------------------------------------------------ protocol 2
    def _obj(self, objtype, ref, n: int, what: str) -> int:
        if isinstance(ref, str):
            i = mujoco.mj_name2id(self.m, objtype, ref)
            if i < 0:
                raise ValueError(f"{what}: no object named '{ref}'")
            return i
        if isinstance(ref, int) and 0 <= ref < n:
            return ref
        raise ValueError(f"{what}: expected a name or an id in 0..{n}")

    def _extended(self, method: str, p: dict) -> Any:
        m, d, O = self.m, self.d, mujoco.mjtObj
        lock = self.engine._lock
        if method == "capabilities":
            return CAPABILITIES
        if method == "time":
            return float(d.time)
        if method == "warnings":
            return int(sum(w.number for w in d.warning))
        if method == "get":
            name = str(p.get("field", ""))
            with lock:
                if name in _DATA_FIELDS:
                    return np.asarray(getattr(d, name), dtype=float).ravel().tolist()
                if name in _MODEL_FIELDS:
                    return np.asarray(getattr(m, name), dtype=float).ravel().tolist()
                if name in _INT_FIELDS:
                    return [int(x) for x in np.asarray(getattr(m, name)).ravel()]
            raise ValueError(f"unknown field '{name}'")
        if method == "set":
            name = str(p.get("field", ""))
            vals = np.asarray(p.get("values") or [], dtype=float)
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"field '{name}': values must be finite")
            with lock:
                if name in _DATA_FIELDS:
                    arr = getattr(d, name)
                elif name in _MODEL_FIELDS:
                    arr = getattr(m, name)
                else:
                    raise ValueError(f"unknown or read-only field '{name}'")
                if vals.size != arr.size:
                    raise ValueError(f"field '{name}' has {arr.size} values, got {vals.size}")
                arr.flat[:] = vals
                if name in _MODEL_FIELDS:
                    mujoco.mj_setConst(m, d)
                mujoco.mj_forward(m, d)
            return None
        if method == "state":
            with lock:
                sig = mujoco.mjtState.mjSTATE_INTEGRATION
                out = np.zeros(mujoco.mj_stateSize(m, sig))
                mujoco.mj_getState(m, d, out, sig)
                return out.tolist()
        if method == "restore":
            st = np.asarray(p.get("state") or [], dtype=float)
            with lock:
                sig = mujoco.mjtState.mjSTATE_INTEGRATION
                if st.size != mujoco.mj_stateSize(m, sig):
                    raise ValueError(f"state has {st.size} values, model needs {mujoco.mj_stateSize(m, sig)}")
                mujoco.mj_setState(m, d, st, sig)
                mujoco.mj_forward(m, d)
            return None
        if method == "option":
            o = m.opt
            return {"timestep": float(o.timestep), "gravity": o.gravity.tolist(), "wind": o.wind.tolist(),
                    "density": float(o.density), "viscosity": float(o.viscosity),
                    "impratio": float(o.impratio), "tolerance": float(o.tolerance),
                    "iterations": int(o.iterations), "integrator": int(o.integrator),
                    "cone": int(o.cone), "solver": int(o.solver)}
        if method == "set_option":
            with lock:
                for k, v in p.items():
                    if k in ("gravity", "wind"):
                        if len(v) != 3:
                            raise ValueError(f"set_option: {k} needs 3 components")
                        getattr(m.opt, k)[:] = v
                    elif k in _OPTS:
                        cur = getattr(m.opt, k)
                        setattr(m.opt, k, type(cur)(v))
                    else:
                        raise ValueError(f"set_option: unknown option '{k}'")
                if m.opt.timestep <= 0:
                    raise ValueError("set_option: timestep must be positive")
            return None
        if method == "jacobian":
            kind = p.get("kind", "body")
            with lock:
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                if kind == "site":
                    mujoco.mj_jacSite(m, d, jp, jr, self._obj(O.mjOBJ_SITE, p.get("name"), m.nsite, "jacobian"))
                elif kind == "body":
                    mujoco.mj_jacBody(m, d, jp, jr, self._obj(O.mjOBJ_BODY, p.get("name"), m.nbody, "jacobian"))
                else:
                    raise ValueError("jacobian: kind must be 'body' or 'site'")
                return {"jacp": jp.tolist(), "jacr": jr.tolist()}
        if method == "mass_matrix":
            with lock:
                M = np.zeros((m.nv, m.nv))
                try:
                    mujoco.mj_fullM(m, d, M)          # MuJoCo >= 3.3: (m, d, dst)
                except TypeError:
                    mujoco.mj_fullM(m, M, d.qM)       # older: (m, dst, M)
                return M.tolist()
        if method == "inverse":
            with lock:
                mujoco.mj_inverse(m, d)
                return d.qfrc_inverse.tolist()
        if method == "contacts":
            out = []
            f6 = np.zeros(6)
            with lock:
                for c in range(d.ncon):
                    con = d.contact[c]
                    mujoco.mj_contactForce(m, d, c, f6)
                    g = [int(x) for x in con.geom]
                    out.append({"geom1": _name(m, O.mjOBJ_GEOM, g[0], "geom") if g[0] >= 0 else None,
                                "geom2": _name(m, O.mjOBJ_GEOM, g[1], "geom") if g[1] >= 0 else None,
                                "dist": float(con.dist), "pos": con.pos.tolist(), "force": f6.tolist()})
            return out
        if method == "scene":
            with lock:
                return [{"name": mujoco.mj_id2name(m, O.mjOBJ_GEOM, g),
                         "type": _GEOM_TYPES[int(m.geom_type[g])] if int(m.geom_type[g]) < len(_GEOM_TYPES) else "?",
                         "body": mujoco.mj_id2name(m, O.mjOBJ_BODY, int(m.geom_bodyid[g])),
                         "size": m.geom_size[g].tolist(), "pos": d.geom_xpos[g].tolist(),
                         "mat": d.geom_xmat[g].tolist(), "rgba": [float(x) for x in m.geom_rgba[g]]}
                        for g in range(m.ngeom)]
        return _NO

    def push(self, body: str, force, duration: float) -> None:
        """Schedule an external push (from the GUI) applied for `duration` seconds."""
        self.disturbances.append({"body": body, "force": list(force), "until": None, "duration": duration})

    def _apply_disturbances(self) -> None:
        if not self.disturbances:
            return
        t = float(self.d.time)
        keep = []
        for dist in self.disturbances:
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, dist["body"])
            if b < 0:
                continue
            if dist["until"] is None:
                dist["until"] = t + dist["duration"]
            if t < dist["until"]:
                self.d.xfrc_applied[b, :3] = dist["force"]
                keep.append(dist)
            else:
                self.d.xfrc_applied[b, :3] = 0
        self.disturbances = keep


class _Handler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        if isinstance(self.request, ssl.SSLSocket):
            self.request.settimeout(10)
            self.request.do_handshake()     # raises on a bad client; socketserver logs and drops it
            self.request.settimeout(None)
        super().setup()

    def handle(self) -> None:  # noqa: D401
        server: CoSimServer = self.server  # type: ignore[assignment]
        server.client_connected()
        sess = server.session
        try:
            for raw in self.rfile:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._send({"id": None, "error": f"bad JSON: {exc}"})
                    continue
                method = msg.get("method", "")
                params = msg.get("params")
                if "id" not in msg:            # notification
                    if method == "bye":
                        break
                    if sess.on_notify:
                        try:
                            sess.on_notify(method, params or {})
                        except Exception:
                            logger.debug("notify callback failed", exc_info=True)
                    continue
                try:
                    result = sess.handle(method, params)
                    self._send({"id": msg["id"], "result": result})
                except Exception as exc:  # report to the Tulpa program as a runtime error
                    self._send({"id": msg["id"], "error": str(exc)})
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            server.client_disconnected()

    def _send(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        self.wfile.flush()


def tls_context(cert: str, key: str, client_ca: Optional[str] = None) -> "ssl.SSLContext":
    """TLS 1.3-only server context; with `client_ca`, clients must present a certificate (mTLS)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(cert, key)
    if client_ca:
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(client_ca)
    return ctx


class CoSimServer(socketserver.ThreadingTCPServer):
    """TCP server hosting one :class:`CoSimSession`. Bind port 0 for an ephemeral port."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, session: CoSimSession, host: str = "127.0.0.1", port: int = 0,
                 tls: Optional["ssl.SSLContext"] = None) -> None:
        if tls is None and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("co-sim server: binding beyond loopback needs TLS (pass tls=tls_context(...))")
        super().__init__((host, port), _Handler)
        self.session = session
        self.tls = tls
        self.clients = 0
        self._thread: Optional[threading.Thread] = None

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        super().server_bind()

    def get_request(self):  # TLS 1.3 handshake per connection when configured
        sock, addr = super().get_request()
        if self.tls is not None:
            # handshake on the handler thread, so a slow or bad client can't stall accept()
            sock = self.tls.wrap_socket(sock, server_side=True, do_handshake_on_connect=False)
        return sock, addr

    @property
    def address(self) -> str:
        host, port = self.server_address[:2]
        return f"{'tls://' if self.tls else ''}{host}:{port}"

    def client_connected(self) -> None:
        self.clients += 1
        logger.info("co-sim client connected (%d active)", self.clients)

    def client_disconnected(self) -> None:
        self.clients -= 1
        logger.info("co-sim client disconnected (%d active)", self.clients)

    def start(self) -> "CoSimServer":
        self._thread = threading.Thread(target=self.serve_forever, name="cosim", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.session.paused.set()
        self.shutdown()
        self.server_close()
