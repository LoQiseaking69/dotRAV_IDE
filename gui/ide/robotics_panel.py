"""Robotics panel: model inspector + validation, joint/actuator controls, physics settings,
snapshots + trajectory recording/replay/export, and Tulpa RL experiment browser.

The logic lives in :mod:`physics_engine.robotics`; this module is the Qt front end.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QPushButton, QSlider, QSpinBox, QTabWidget, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QScrollArea)

from physics_engine import robotics as rb

try:
    import pyqtgraph as pg
except ImportError:  # plots are optional
    pg = None


class RoboticsPanel(QWidget):
    status = Signal(str)

    def __init__(self, simulation, parent=None) -> None:
        super().__init__(parent)
        self.simulation = simulation
        self.engine = simulation.engine
        self.recording = rb.Recording()
        self.snapshots: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self._model_tab()
        self._control_tab()
        self._physics_tab()
        self._record_tab()
        self._experiments_tab()
        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._capture)
        simulation.model_loaded.connect(lambda _info: self.refresh())
        if self.engine.is_initialized:
            self.refresh()

    # ------------------------------------------------------------------ model
    def _model_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Details"])
        self.tree.setColumnWidth(0, 180)
        v.addWidget(self.tree, 3)
        row = QHBoxLayout()
        b = QPushButton("Validate model")
        b.clicked.connect(self.validate)
        row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        self.findings = QListWidget()
        v.addWidget(self.findings, 1)
        self.tabs.addTab(w, "Model")

    def refresh(self) -> None:
        info = rb.inspect_model(self.engine)
        self.tree.clear()
        for kind, fmt in [
            ("joints", lambda j: f"{j['type']}" + (f" [{j['unit']}]" if j["unit"] else "") +
             (f"  range {j['range'][0]:.3g}..{j['range'][1]:.3g}" if j["limited"] else "  unlimited") + f"  on {j['body']}"),
            ("actuators", lambda a: f"{a['transmission']} → {a['target'] or '-'}  gear {a['gear']:g}" +
             (f"  ctrl {a['ctrlrange'][0]:g}..{a['ctrlrange'][1]:g}" if a["limited"] else "  unbounded")),
            ("sensors", lambda s: f"type {s['type']}, dim {s['dim']}" + (f", noise {s['noise']:g}" if s["noise"] else "")),
            ("bodies", lambda b: f"mass {b['mass']:.4g} kg" + (f", parent {b['parent']}" if b["parent"] else "")),
            ("sites", lambda s: f"on {s['body']}"),
        ]:
            top = QTreeWidgetItem([f"{kind.capitalize()} ({len(info.get(kind, []))})", ""])
            for x in info.get(kind, []):
                top.addChild(QTreeWidgetItem([x["name"], fmt(x)]))
            self.tree.addTopLevelItem(top)
            top.setExpanded(kind in ("joints", "actuators"))
        self._build_controls(info)
        self._load_physics()
        self.recording = rb.Recording()
        self.snapshots.clear()
        self.snap_list.clear()
        self._update_replay()

    def validate(self) -> list:
        self.findings.clear()
        res = rb.validate(self.engine)
        for sev, msg in res:
            self.findings.addItem(f"{ {'error': '✖', 'warning': '⚠', 'info': 'ℹ'}[sev]}  {msg}")
        if not res:
            self.findings.addItem("✔  no problems found")
        return res

    # ------------------------------------------------------------------ control
    def _control_tab(self) -> None:
        self.ctrl_area = QScrollArea()
        self.ctrl_area.setWidgetResizable(True)
        self.tabs.addTab(self.ctrl_area, "Control")

    def _build_controls(self, info: dict) -> None:
        w = QWidget()
        form = QFormLayout(w)
        self.joint_sliders: dict[str, QSlider] = {}
        self.act_sliders: dict[str, QSlider] = {}
        form.addRow(QLabel("<b>Joints</b> (teleport; best while paused)"))
        for j in info.get("joints", []):
            if j["type"] not in ("hinge", "slide"):
                continue
            lo, hi = j["range"] if j["limited"] else ((-3.1416, 3.1416) if j["type"] == "hinge" else (-1.0, 1.0))
            s = self._slider(lo, hi, lambda val, n=j["name"]: self._set_joint(n, val))
            self.joint_sliders[j["name"]] = s
            form.addRow(f"{j['name']} [{j['unit']}]", s)
        form.addRow(QLabel("<b>Actuators</b> (control signal)"))
        for a in info.get("actuators", []):
            lo, hi = a["ctrlrange"] if a["limited"] else (-1.0, 1.0)
            s = self._slider(lo, hi, lambda val, n=a["name"]: self._set_ctrl(n, val), centre=True)
            self.act_sliders[a["name"]] = s
            form.addRow(a["name"], s)
        zero = QPushButton("Zero all controls")
        zero.clicked.connect(lambda: [s.setValue(500) for s in self.act_sliders.values()])
        form.addRow(zero)
        self.ctrl_area.setWidget(w)

    def _slider(self, lo: float, hi: float, fn, centre: bool = False) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(0, 1000)
        s.lo, s.hi = lo, hi
        s.setValue(500 if centre or lo < 0 < hi else 0)
        s.valueChanged.connect(lambda v: fn(lo + (hi - lo) * v / 1000.0))
        return s

    def _set_joint(self, name: str, val: float) -> None:
        try:
            rb.set_joint(self.engine, name, val)
            self.status.emit(f"{name} = {val:.4g}")
        except (KeyError, ValueError) as e:
            self.status.emit(str(e))

    def _set_ctrl(self, name: str, val: float) -> None:
        applied = rb.set_control(self.engine, name, val)
        self.status.emit(f"ctrl {name} = {applied:.4g}")

    # ------------------------------------------------------------------ physics
    def _physics_tab(self) -> None:
        w = QWidget()
        f = QFormLayout(w)
        self.dt = QDoubleSpinBox(decimals=5, minimum=1e-5, maximum=0.1, singleStep=0.0005)
        self.grav = [QDoubleSpinBox(decimals=3, minimum=-100, maximum=100) for _ in range(3)]
        self.integ = QComboBox()
        self.integ.addItems(rb._INTEGRATORS)
        self.iters = QSpinBox(minimum=1, maximum=1000)
        f.addRow("Timestep [s]", self.dt)
        g = QHBoxLayout()
        for s in self.grav:
            g.addWidget(s)
        f.addRow("Gravity [m/s²]", g)
        f.addRow("Integrator", self.integ)
        f.addRow("Solver iterations", self.iters)
        apply = QPushButton("Apply")
        apply.clicked.connect(self.apply_physics)
        f.addRow(apply)
        self.tabs.addTab(w, "Physics")

    def _load_physics(self) -> None:
        if not self.engine.is_initialized:
            return
        p = rb.physics(self.engine)
        self.dt.setValue(p["timestep"])
        for s, v in zip(self.grav, p["gravity"]):
            s.setValue(v)
        self.integ.setCurrentText(p["integrator"])
        self.iters.setValue(p["iterations"])

    def apply_physics(self) -> None:
        try:
            rb.configure(self.engine, timestep=self.dt.value(), gravity=[s.value() for s in self.grav],
                         integrator=self.integ.currentText(), iterations=self.iters.value())
            self.status.emit("physics updated")
        except (KeyError, ValueError) as e:
            self.status.emit(str(e))

    # ------------------------------------------------------------------ record / replay
    def _record_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.rec_btn = QPushButton("● Record")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self.set_recording)
        row.addWidget(self.rec_btn)
        exp = QPushButton("Export CSV…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        row.addStretch()
        v.addLayout(row)
        self.replay = QSlider(Qt.Horizontal)
        self.replay.valueChanged.connect(self._replay_to)
        self.replay_label = QLabel("no recording")
        v.addWidget(self.replay)
        v.addWidget(self.replay_label)
        srow = QHBoxLayout()
        snap = QPushButton("Snapshot")
        snap.clicked.connect(self.take_snapshot)
        rest = QPushButton("Restore selected")
        rest.clicked.connect(self.restore_selected)
        srow.addWidget(snap)
        srow.addWidget(rest)
        srow.addStretch()
        v.addLayout(srow)
        self.snap_list = QListWidget()
        v.addWidget(self.snap_list)
        self.tabs.addTab(w, "Record")

    def set_recording(self, on: bool) -> None:
        self.rec_btn.setText("■ Stop" if on else "● Record")
        if on:
            self.recording = rb.Recording()
            self.rec_timer.start(20)
        else:
            self.rec_timer.stop()
            self._update_replay()

    def _capture(self) -> None:
        if self.engine.is_initialized:
            self.recording.capture(self.engine)

    def _update_replay(self) -> None:
        n = len(self.recording)
        self.replay.setRange(0, max(0, n - 1))
        self.replay_label.setText(f"{n} frames recorded" if n else "no recording")

    def _replay_to(self, i: int) -> None:
        if len(self.recording) and not self.rec_timer.isActive():
            t = self.recording.replay(self.engine, i)
            self.replay_label.setText(f"frame {i + 1}/{len(self.recording)}  t = {t:.3f} s")

    def _export(self) -> None:
        if not len(self.recording):
            self.status.emit("nothing recorded yet")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export trajectory", "trajectory.csv", "CSV (*.csv)")
        if path:
            n = self.recording.export_csv(path)
            self.status.emit(f"wrote {n} frames to {path}")

    def take_snapshot(self) -> dict:
        s = rb.snapshot(self.engine)
        self.snapshots.append(s)
        self.snap_list.addItem(f"#{len(self.snapshots)}  t = {s['time']:.3f} s")
        return s

    def restore_selected(self) -> None:
        i = self.snap_list.currentRow()
        if 0 <= i < len(self.snapshots):
            rb.restore(self.engine, self.snapshots[i])
            self.status.emit(f"restored snapshot #{i + 1}")

    # ------------------------------------------------------------------ experiments
    def _experiments_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.runs_dir = QLineEdit("runs")
        browse = QPushButton("…")
        browse.clicked.connect(lambda: self.runs_dir.setText(QFileDialog.getExistingDirectory(self, "Runs folder") or self.runs_dir.text()))
        load = QPushButton("Refresh")
        load.clicked.connect(self.load_experiments)
        row.addWidget(QLabel("Runs:"))
        row.addWidget(self.runs_dir, 1)
        row.addWidget(browse)
        row.addWidget(load)
        v.addLayout(row)
        self.exp_table = QTableWidget(0, 6)
        self.exp_table.setHorizontalHeaderLabels(["Run", "Algo", "Steps", "Best eval", "Last eval", "Checkpoints"])
        self.exp_table.itemSelectionChanged.connect(self._plot_selected)
        v.addWidget(self.exp_table, 1)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Metric:"))
        self.metric = QComboBox()
        self.metric.setEditable(True)
        self.metric.addItems(["eval_return", "train_return", "q_loss", "pi_loss", "alpha", "approx_kl"])
        self.metric.currentTextChanged.connect(lambda _t: self._plot_selected())
        mrow.addWidget(self.metric, 1)
        v.addLayout(mrow)
        if pg is not None:
            self.exp_plot = pg.PlotWidget()
            self.exp_plot.showGrid(x=True, y=True, alpha=0.2)
            v.addWidget(self.exp_plot, 1)
        else:
            self.exp_plot = None
        self.experiments: list[dict] = []
        self.tabs.addTab(w, "Experiments")

    def load_experiments(self) -> list[dict]:
        self.experiments = rb.experiments(self.runs_dir.text())
        t = self.exp_table
        t.setRowCount(len(self.experiments))
        for r, e in enumerate(self.experiments):
            vals = [e["name"] + ("" if e["finished"] else " (running)"), e["algo"], str(e["steps"]),
                    "" if e["best_eval"] is None else f"{e['best_eval']:.2f}",
                    "" if e["last_eval"] is None else f"{e['last_eval']:.2f}", str(e["checkpoints"])]
            for c, val in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(val))
        return self.experiments

    def _plot_selected(self) -> None:
        if self.exp_plot is None:
            return
        self.exp_plot.clear()
        rows = sorted({i.row() for i in self.exp_table.selectedIndexes()})
        for k, r in enumerate(rows):
            xs, ys = rb.metric_series(rb.load_metrics(self.experiments[r]["dir"]), self.metric.currentText())
            if xs:
                self.exp_plot.plot(xs, ys, pen=pg.intColor(k, hues=max(3, len(rows))), name=self.experiments[r]["name"])
