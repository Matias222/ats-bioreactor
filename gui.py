"""
gui.py — Tkinter GUI for the TAS-controlled fed-batch bioreactor.

Provides a digital twin interface with:
  - 4 real-time plots (Biomass, Substrate, DO, Pressure/Volume)
  - Scenario selector (A: hypoxia recovery, B: overflow cascade)
  - Perturbation buttons (inject pressure, sensor fault)
  - Live status panel and scrollable event log
"""

import copy
import io
import sys
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from bioreactor_ode import PARAMS
from tas import AutonomousBioreactor

# ─────────────────────────────────────────────
# Simulation parameters (same as main.py)
# ─────────────────────────────────────────────

SIM_PARAMS = copy.deepcopy(PARAMS)
SIM_PARAMS["kLa_ref"] = 50.0

INITIAL_CONDITIONS = dict(X0=0.5, S0=5.0, DO0=8.0, V0=1.0)
INITIAL_PUMP_RPM   = 12
INITIAL_TEMP       = 37.0
DT_HOURS           = 0.1
MAX_VOLUME         = 20.0
DO_STAR            = SIM_PARAMS["DO_star"]


# ─────────────────────────────────────────────
# stdout capture for event log
# ─────────────────────────────────────────────

class _StdoutCapture:
    """Intercepts stdout and sends lines to a queue for the GUI event log."""

    _EVENT_PREFIXES = ("[WARNING]", "[CRITICAL]", "[WILL]", "[HALT]", "[END]",
                       "[ACTION]")

    def __init__(self, log_queue):
        self._queue = log_queue
        self._orig = sys.stdout
        self._buf = ""

    def __enter__(self):
        sys.stdout = self
        return self

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped and any(stripped.startswith(p) for p in self._EVENT_PREFIXES):
                self._queue.put(("log", stripped))

    def flush(self):
        pass

    def __exit__(self, *args):
        sys.stdout = self._orig


# ─────────────────────────────────────────────
# Pressure model for Scenario B
# ─────────────────────────────────────────────

def _pressure_at(t: float) -> float:
    if t < 7.0:
        return 1.0
    if t < 9.0:
        return 1.0 + (t - 7.0) * 0.55
    return 2.1


# ─────────────────────────────────────────────
# GUI Application
# ─────────────────────────────────────────────

class BioreactorGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bioreactor Autonomo — TAS Digital Twin")
        self.root.geometry("1250x850")
        self.root.minsize(1000, 700)

        # State
        self.reactor = None
        self.sim_thread = None
        self.sim_running = False
        self.sim_paused = False
        self.data_queue = queue.Queue()
        self.log_queue = queue.Queue()

        # Plot data lists
        self._clear_data()

        # Perturbation flags
        self.pressure_override = None
        self.sensor_fault_active = False

        # Build UI
        self._build_plots()
        self._build_controls()
        self._build_log()

        # Start polling
        self._poll()

    # ── Data management ──────────────────────

    def _clear_data(self):
        self.t_data = []
        self.x_data = []
        self.s_data = []
        self.do_data = []
        self.v_data = []
        self.p_data = []

    # ── Build UI ─────────────────────────────

    def _build_plots(self):
        """Create 2x2 matplotlib figure embedded in Tkinter."""
        self.fig = Figure(figsize=(11, 5), dpi=100)
        self.fig.set_facecolor("#f0f0f0")

        self.ax_x  = self.fig.add_subplot(2, 2, 1)
        self.ax_s  = self.fig.add_subplot(2, 2, 2)
        self.ax_do = self.fig.add_subplot(2, 2, 3)
        self.ax_pv = self.fig.add_subplot(2, 2, 4)

        # Biomass
        self.line_x, = self.ax_x.plot([], [], color="#1f77b4", linewidth=1.5)
        self.ax_x.set_title("Biomass X (g/L)", fontsize=10)
        self.ax_x.set_xlabel("Time (h)", fontsize=8)
        self.ax_x.grid(True, alpha=0.3)

        # Substrate
        self.line_s, = self.ax_s.plot([], [], color="#2ca02c", linewidth=1.5)
        self.ax_s.set_title("Substrate S (g/L)", fontsize=10)
        self.ax_s.set_xlabel("Time (h)", fontsize=8)
        self.ax_s.grid(True, alpha=0.3)

        # DO with threshold lines
        self.line_do, = self.ax_do.plot([], [], color="#d62728", linewidth=1.5)
        self.ax_do.axhline(y=DO_STAR * 0.35, color="orange", linestyle="--",
                           linewidth=0.8, label="WARNING 35%")
        self.ax_do.axhline(y=DO_STAR * 0.15, color="red", linestyle="--",
                           linewidth=0.8, label="CRITICAL 15%")
        self.ax_do.set_title("Dissolved Oxygen (mg/L)", fontsize=10)
        self.ax_do.set_xlabel("Time (h)", fontsize=8)
        self.ax_do.legend(fontsize=7, loc="upper right")
        self.ax_do.grid(True, alpha=0.3)

        # Pressure + Volume (dual axis)
        self.line_p, = self.ax_pv.plot([], [], color="#ff7f0e", linewidth=1.5,
                                       label="Pressure (atm)")
        self.ax_v = self.ax_pv.twinx()
        self.line_v, = self.ax_v.plot([], [], color="#17becf", linewidth=1.5,
                                      label="Volume (L)")
        self.ax_pv.set_title("Pressure / Volume", fontsize=10)
        self.ax_pv.set_xlabel("Time (h)", fontsize=8)
        self.ax_pv.set_ylabel("Pressure (atm)", fontsize=8, color="#ff7f0e")
        self.ax_v.set_ylabel("Volume (L)", fontsize=8, color="#17becf")
        self.ax_pv.legend(fontsize=7, loc="upper left")
        self.ax_v.legend(fontsize=7, loc="upper right")
        self.ax_pv.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_controls(self):
        """Build the control panel below the plots."""
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        # ── Column 1: Scenario + Sim controls ──
        col1 = ttk.LabelFrame(ctrl_frame, text="Simulation", padding=5)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.scenario_var = tk.StringVar(value="A")
        ttk.Radiobutton(col1, text="Scenario A (Hypoxia)",
                        variable=self.scenario_var, value="A").pack(anchor=tk.W)
        ttk.Radiobutton(col1, text="Scenario B (Overflow)",
                        variable=self.scenario_var, value="B").pack(anchor=tk.W)

        ttk.Separator(col1, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(col1)
        btn_frame.pack()
        self.btn_start = ttk.Button(btn_frame, text="Start", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self._stop,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        self.btn_reset = ttk.Button(btn_frame, text="Reset", command=self._reset)
        self.btn_reset.pack(side=tk.LEFT, padx=2)

        # ── Column 2: Perturbations ──
        col2 = ttk.LabelFrame(ctrl_frame, text="Perturbations", padding=5)
        col2.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.btn_pressure = ttk.Button(col2, text="Inject Pressure (2 atm)",
                                       command=self._inject_pressure)
        self.btn_pressure.pack(fill=tk.X, pady=2)
        self.btn_sensor = ttk.Button(col2, text="DO Sensor Fault",
                                     command=self._sensor_fault)
        self.btn_sensor.pack(fill=tk.X, pady=2)
        self.btn_reset_pert = ttk.Button(col2, text="Reset Perturbations",
                                         command=self._reset_perturbations)
        self.btn_reset_pert.pack(fill=tk.X, pady=2)

        # ── Column 3: Status ──
        col3 = ttk.LabelFrame(ctrl_frame, text="Status", padding=5)
        col3.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.status_var = tk.StringVar(value="IDLE")
        self.lbl_status = ttk.Label(col3, textvariable=self.status_var,
                                    font=("Courier", 11, "bold"),
                                    foreground="gray")
        self.lbl_status.pack(anchor=tk.W)

        self.pump_var = tk.StringVar(value="pump: — rpm")
        ttk.Label(col3, textvariable=self.pump_var, font=("Courier", 9)).pack(anchor=tk.W)
        self.motor_var = tk.StringVar(value="motor: — rpm")
        ttk.Label(col3, textvariable=self.motor_var, font=("Courier", 9)).pack(anchor=tk.W)
        self.valve_var = tk.StringVar(value="valve: —")
        ttk.Label(col3, textvariable=self.valve_var, font=("Courier", 9)).pack(anchor=tk.W)

        # ── Column 4: Current values ──
        col4 = ttk.LabelFrame(ctrl_frame, text="Current Values", padding=5)
        col4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.values_var = tk.StringVar(value="t=—  X=—  S=—\nDO=—  V=—  P=—")
        ttk.Label(col4, textvariable=self.values_var,
                  font=("Courier", 10), justify=tk.LEFT).pack(anchor=tk.W)

    def _build_log(self):
        """Build the event log area at the bottom."""
        log_frame = ttk.LabelFrame(self.root, text="Event Log", padding=3)
        log_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10), expand=False)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, wrap=tk.WORD,
            font=("Courier", 9), state=tk.DISABLED,
            background="#1e1e1e", foreground="#cccccc",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Tag colors for event types
        self.log_text.tag_config("WARNING", foreground="#ffcc00")
        self.log_text.tag_config("CRITICAL", foreground="#ff4444")
        self.log_text.tag_config("WILL", foreground="#44ff44")
        self.log_text.tag_config("HALT", foreground="#ff4444", underline=True)
        self.log_text.tag_config("END", foreground="#88ccff")
        self.log_text.tag_config("ACTION", foreground="#ff4444")

    # ── Simulation controls ──────────────────

    def _create_reactor(self):
        params = copy.deepcopy(SIM_PARAMS)
        self.reactor = AutonomousBioreactor(
            params=params, ic=INITIAL_CONDITIONS.copy(),
            pump_rpm=INITIAL_PUMP_RPM, temp=INITIAL_TEMP,
            max_volume=MAX_VOLUME,
        )

    def _start(self):
        if self.sim_running and self.sim_paused:
            self.sim_paused = False
            self._set_status("RUNNING", "#00aa00")
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            return

        self._clear_data()
        self._clear_log()
        self._create_reactor()
        self.pressure_override = None
        self.sensor_fault_active = False
        self.sim_running = True
        self.sim_paused = False

        self._set_status("RUNNING", "#00aa00")
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.scenario_var.get()  # read before thread starts

        self.sim_thread = threading.Thread(target=self._simulation_loop,
                                           daemon=True)
        self.sim_thread.start()

    def _stop(self):
        if self.sim_running:
            self.sim_paused = True
            self._set_status("PAUSED", "#cc8800")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def _reset(self):
        self.sim_running = False
        self.sim_paused = False
        self.pressure_override = None
        self.sensor_fault_active = False
        self._clear_data()
        self._update_plots()
        self._clear_log()
        self._set_status("IDLE", "gray")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.values_var.set("t=—  X=—  S=—\nDO=—  V=—  P=—")
        self.pump_var.set("pump: — rpm")
        self.motor_var.set("motor: — rpm")
        self.valve_var.set("valve: —")

    def _set_status(self, text, color):
        self.status_var.set(text)
        self.lbl_status.config(foreground=color)

    # ── Perturbation buttons ─────────────────

    def _inject_pressure(self):
        self.pressure_override = 2.0
        self._append_log("[PERTURBATION] Manual pressure injection: 2.0 atm")

    def _sensor_fault(self):
        self.sensor_fault_active = True
        self._append_log("[PERTURBATION] DO sensor fault injected (reading → 0.0)")

    def _reset_perturbations(self):
        self.pressure_override = None
        self.sensor_fault_active = False
        self._append_log("[PERTURBATION] All perturbations cleared")

    # ── Simulation thread ────────────────────

    def _simulation_loop(self):
        scenario = self.scenario_var.get()
        step_num = 0

        stdout_capture = _StdoutCapture(self.log_queue)

        while self.sim_running and step_num < 300:
            if self.sim_paused:
                time.sleep(0.05)
                continue

            # Apply perturbations
            if scenario == "B" and self.pressure_override is None:
                self.reactor.tank.pressure.value = _pressure_at(
                    self.reactor.process.time)
            if self.pressure_override is not None:
                self.reactor.tank.pressure.value = self.pressure_override
            if self.sensor_fault_active:
                self.reactor.tank.oxygen.value = 0.0

            # Step with stdout capture
            with stdout_capture:
                running, actions = self.reactor.step()

            # Collect data
            p = self.reactor.process
            data = {
                "t": p.time,
                "X": p.X,
                "S": p.S,
                "DO": p.DO,
                "V": p.V,
                "P": self.reactor.tank.pressure.value,
                "pump": self.reactor.substrate_line.pump.speed,
                "motor": self.reactor.tank.motor.rpm,
                "valve": self.reactor.air_line.control_valve.is_open,
                "running": running,
                "actions": actions,
            }
            self.data_queue.put(("data", data))

            if not running:
                self.data_queue.put(("done", None))
                return

            step_num += 1
            time.sleep(0.05)  # pace the simulation for visual effect

        self.data_queue.put(("done", None))

    # ── Polling (main thread) ────────────────

    def _poll(self):
        """Called every 50ms by Tkinter to drain queues and update UI."""
        updated = False

        # Drain data queue
        try:
            while True:
                msg_type, payload = self.data_queue.get_nowait()
                if msg_type == "data":
                    self.t_data.append(payload["t"])
                    self.x_data.append(payload["X"])
                    self.s_data.append(payload["S"])
                    self.do_data.append(payload["DO"])
                    self.v_data.append(payload["V"])
                    self.p_data.append(payload["P"])
                    self._update_values(payload)
                    updated = True
                elif msg_type == "done":
                    self.sim_running = False
                    if self.status_var.get() == "RUNNING":
                        self._set_status("COMPLETE", "#0066cc")
                    self.btn_start.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
        except queue.Empty:
            pass

        # Drain log queue
        try:
            while True:
                msg_type, line = self.log_queue.get_nowait()
                if msg_type == "log":
                    self._append_log(line)
        except queue.Empty:
            pass

        if updated:
            self._update_plots()

        self.root.after(50, self._poll)

    # ── UI updates ───────────────────────────

    def _update_values(self, data):
        self.values_var.set(
            f"t={data['t']:>5.1f}h   X={data['X']:>7.1f} g/L   "
            f"S={data['S']:>7.1f} g/L\n"
            f"DO={data['DO']:>6.2f} mg/L   V={data['V']:>5.2f} L   "
            f"P={data['P']:>5.2f} atm"
        )
        self.pump_var.set(f"pump: {data['pump']} rpm")
        self.motor_var.set(f"motor: {data['motor']:.0f} rpm")
        self.valve_var.set(f"valve: {'OPEN' if data['valve'] else 'CLOSED'}")

        # Update status on halt
        if not data["running"]:
            self._set_status("HALTED", "#cc0000")

    def _update_plots(self):
        t = self.t_data

        self.line_x.set_data(t, self.x_data)
        self.line_s.set_data(t, self.s_data)
        self.line_do.set_data(t, self.do_data)
        self.line_p.set_data(t, self.p_data)
        self.line_v.set_data(t, self.v_data)

        for ax in [self.ax_x, self.ax_s, self.ax_do, self.ax_pv, self.ax_v]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw_idle()

    def _append_log(self, line):
        self.log_text.config(state=tk.NORMAL)

        # Determine tag
        tag = None
        for prefix in ("CRITICAL", "HALT", "ACTION", "WARNING", "WILL", "END"):
            if f"[{prefix}]" in line:
                tag = prefix
                break

        if tag:
            self.log_text.insert(tk.END, line + "\n", tag)
        else:
            self.log_text.insert(tk.END, line + "\n")

        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = BioreactorGUI(root)
    root.mainloop()
