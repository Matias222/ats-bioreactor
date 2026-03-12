"""
simulation.py
─────────────────────────────────────────────────────────────
Demonstration runner for the TAS-controlled fed-batch bioreactor.

Two scenarios, both starting from identical initial conditions
(X₀=0.5 g/L, S₀=5 g/L, DO₀=8 mg/L, V₀=1 L, pump=12 rpm):

  SCENARIO A — Demand-Driven Hypoxia → Recovery
    High biomass growth outpaces oxygen transfer (kLa=50 h⁻¹).
    TAS detects low DO corroborated by rising biomass, issues
    reduce_feed, and the run continues to substrate exhaustion.

  SCENARIO B — Overflow Cascade → Process Halt
    Same trajectory, but CO₂ back-pressure accumulates at peak
    biomass (injected perturbation). Combined CRITICAL low DO
    + CRITICAL pressure triggers the overflow cascade halt rule.

Output: periodic status table during nominal operation;
        full TAS reasoning printed only when anomalies fire.
─────────────────────────────────────────────────────────────
"""

import copy
import sys
import io as _io
from main import Tank, AirLine, GasExit, AntiFoamLine, BaseLine, AcidLine, SubstrateLine

import fault_graph
from Bioreactor_ODE import BioreactorProcess, feed_rate_from_speed, PARAMS
from Constellation_copy import (
    TAS_SensorySystem, PresentationSystem, ReasonSystem,
    DecisionSystem, WillSystem, IntellectSystem,
    is_plateaued, is_volume_limit_reached,
)

# ─────────────────────────────────────────────
# Simulation-wide parameters
# ─────────────────────────────────────────────

SIM_PARAMS = copy.deepcopy(PARAMS)
# kLa is now computed dynamically from motor rpm and valve state.
# To reproduce the low-aeration scenario, we lower N_ref so the
# simulation motor (set to 400 rpm) yields kLa ≈ 50 h⁻¹ at that speed.
# Equivalently we just lower kLa_ref to 50 — same effect, cleaner.
SIM_PARAMS["kLa_ref"] = 50.0   # h⁻¹ at N_ref — low aeration to stress-test DO

INITIAL_CONDITIONS = dict(X0=0.5, S0=5.0, DO0=8.0, V0=1.0)
INITIAL_PUMP_RPM   = 12
INITIAL_TEMP       = 37.0
DT_HOURS           = 0.1       # 6-minute timesteps
STATUS_EVERY       = 10        # print a status row every N nominal steps


# ─────────────────────────────────────────────
# stdout filter: suppress routine lines from
# terminal but dump everything to log.txt
# ─────────────────────────────────────────────

LOG_PATH = "./log.txt"

_SUPPRESS_PREFIXES = (
    "[INFO]",
    "[DECISION] Evaluating",
    "[DECISION] No corrective",
    "[DECISION] Process continuing",
)

class _SuppressInfo:
    """
    Context manager that intercepts stdout and:
      - forwards non-suppressed lines to the real terminal
      - writes ALL lines (including suppressed ones) to LOG_PATH

    Lines matching _SUPPRESS_PREFIXES and blank lines are hidden from
    the terminal but fully preserved in the log for post-run inspection.
    """

    def __init__(self, log_file):
        self._log = log_file   # open file handle, written to but not closed here

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = self
        self._buf  = ""
        return self

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._log.write(line + "\n")   # always log
            suppress = (not line.strip()) or any(line.startswith(p) for p in _SUPPRESS_PREFIXES)
            if not suppress:
                self._orig.write(line + "\n")

    def flush(self):
        if self._buf:
            self._log.write(self._buf)
            if not any(self._buf.startswith(p) for p in _SUPPRESS_PREFIXES):
                self._orig.write(self._buf)
        self._buf = ""

    def __exit__(self, *args):
        self.flush()
        sys.stdout = self._orig


class _NullCtx:
    """No-op context manager."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ─────────────────────────────────────────────
# Reactor wrapper
# ─────────────────────────────────────────────

class SimReactor:
    """
    Wires all TAS subsystems and the ODE process model together.
    Mirrors AutonomousBioreactor but accepts custom params/ICs
    so scenarios can be configured without touching shared modules.
    """

    MAX_VOLUME = 20.0   # L — simulation working volume limit

    def __init__(self, params: dict, ic: dict, pump_rpm: int,
                 log_file=None):
        self.log_file = log_file   # open file handle for _SuppressInfo; None = discard
        self.tank           = Tank()
        self.air_line       = AirLine()
        self.gas_exit       = GasExit()
        self.antifoam_line  = AntiFoamLine()
        self.base_line      = BaseLine()
        self.acid_line      = AcidLine()
        self.substrate_line = SubstrateLine()

        self.process = BioreactorProcess(**ic, params=params)
        self.substrate_line.pump.set_speed(pump_rpm)
        self.tank.motor.set_speed(400)          # N_ref → kLa = kLa_ref at start
        self.air_line.control_valve.open()      # sparger open from t=0
        self.tank.temperature.value = INITIAL_TEMP
        self.process.update_sensors(self.tank)

        fault_graph.initialize_fault_graph()

        self.sensory      = TAS_SensorySystem(self.tank)
        self.presentation = PresentationSystem()
        self.reason       = ReasonSystem()
        self.decision     = DecisionSystem()
        self.will         = WillSystem(self)
        self.intellect    = IntellectSystem()

    # ── sensor proxies expected by fault_graph check functions ──
    @property
    def DO_sensor(self):       return self.tank.oxygen
    @property
    def biomass_sensor(self):  return self.tank.od
    @property
    def pressure_sensor(self): return self.tank.pressure

    def step(self, dt_hours: float = DT_HOURS,
             verbose: bool = False) -> "tuple[bool, list[str]]":
        """
        Advance one timestep.

        Parameters
        ----------
        verbose : if True, let all TAS output through (including [INFO]).

        Returns
        -------
        (running, actions_taken)
        """
        F          = feed_rate_from_speed(self.substrate_line.pump.speed, SIM_PARAMS)
        T          = self.tank.temperature.value
        motor_rpm  = self.tank.motor.rpm
        valve_open = self.air_line.control_valve.is_open
        self.process.step(dt_hours, F, T, motor_rpm, valve_open)
        self.process.update_sensors(self.tank)

        raw_data  = self.sensory.read_environment()
        formatted = self.presentation.format_data(raw_data)
        formatted["time_h"] = self.process.time
        self.intellect.store(formatted)

        if verbose:
            ctx = _NullCtx()
        else:
            ctx = _SuppressInfo(self.log_file) if self.log_file else _NullCtx()
        actions_taken = []

        with ctx:
            self.reason.evaluate(self, self.intellect)
            running, actions = self.decision.decide()
            # consume any remaining buffered lines inside the context

        for action in actions:
            if action == "increase_airflow":
                self.will.increase_airflow()
                actions_taken.append("increase_airflow")
            elif action == "reduce_feed":
                self.will.reduce_feed()
                actions_taken.append("reduce_feed")
            elif action == "flag_sensor":
                print("[WILL] Sensor flagged — continuing with caution.")
                actions_taken.append("flag_sensor")

        if running:
            if is_plateaued(self.intellect):
                print("\n[END] Biomass plateau detected. Run complete.")
                return False, actions_taken
            if self.process.V >= self.MAX_VOLUME * 0.9:
                print("\n[END] Volume limit reached. Run complete.")
                return False, actions_taken

        return running, actions_taken


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

_SEP    = "  " + "─" * 72
_HEADER = (
    f"  {'t (h)':>6}  {'X (g/L)':>9}  {'S (g/L)':>9}  "
    f"{'DO (mg/L)':>10}  {'DO %':>6}  {'V (L)':>6}  {'pump':>5}"
)

def _banner(title: str):
    print("\n")
    print("╔" + "═" * 70 + "╗")
    print(f"║  {title:<68}║")
    print("╚" + "═" * 70 + "╝")

def _nominal_row(r: SimReactor, extra: str = "") -> str:
    p    = r.process
    pump = r.substrate_line.pump.speed
    pct  = p.DO / SIM_PARAMS["DO_star"] * 100
    row  = (
        f"  {p.time:>6.1f}  {p.X:>9.2f}  {p.S:>9.2f}  "
        f"{p.DO:>10.3f}  {pct:>5.1f}%  {p.V:>6.2f}  {pump:>5}"
    )
    return row + (f"  ← {extra}" if extra else "")

def _event_block(r: SimReactor, tag: str, detail: str, pressure: float = None):
    """Highlighted block for WARNING / CRITICAL events."""
    p    = r.process
    pump = r.substrate_line.pump.speed
    pct  = p.DO / SIM_PARAMS["DO_star"] * 100
    p_str = f"  P={pressure:.2f} atm" if pressure is not None else ""
    sep = "!" * 72
    print(f"\n  {sep}")
    print(
        f"  [{tag}]  t={p.time:.1f} h  |  X={p.X:.1f} g/L  |  "
        f"DO={p.DO:.3f} mg/L ({pct:.1f}%){p_str}  |  pump={pump} rpm"
    )
    print(f"  {detail}")
    print(f"  {sep}\n")


# ─────────────────────────────────────────────
# Scenario A — Recovery
# ─────────────────────────────────────────────

def scenario_a():
    _banner("SCENARIO A  ·  Demand-Driven Hypoxia  →  Recovery")
    print(
        "  kLa = 50 h⁻¹.  Pump at 12 rpm.  Biomass enters exponential phase ~t=5 h.\n"
        "  O₂ demand begins to outpace transfer; DO falls through the WARNING threshold\n"
        "  (~35 % sat) around t=9 h.  ReasonSystem correlates low DO with rising biomass\n"
        "  and diagnoses demand-driven hypoxia.  WillSystem halves the feed pump speed.\n"
        "  Substrate is exhausted shortly after; run ends at biomass plateau.\n"
    )
    print(_SEP)
    print(_HEADER)
    print(_SEP)

    with open(LOG_PATH, "a") as log:
        log.write("\n" + "=" * 72 + "\n")
        log.write("SCENARIO A  ·  Demand-Driven Hypoxia  →  Recovery\n")
        log.write("=" * 72 + "\n\n")
        reactor = SimReactor(SIM_PARAMS, INITIAL_CONDITIONS, INITIAL_PUMP_RPM,
                             log_file=log)

        for step in range(1, 301):
            running, actions = reactor.step()
            pct = reactor.process.DO / SIM_PARAMS["DO_star"] * 100

            # Determine event severity
            is_critical = pct < 15.0
            is_warning  = pct < 35.0 or bool(actions)

            if is_critical:
                _event_block(reactor, "CRITICAL",
                             f"Actions issued: {', '.join(actions) or 'none'}")
            elif is_warning:
                action_str = (f"  →  actions: {', '.join(actions)}"
                              if actions else "  (no actions yet)")
                _event_block(reactor, "WARNING",
                             f"DO below threshold ({pct:.1f} % sat){action_str}")
            elif step % STATUS_EVERY == 0:
                print(_nominal_row(reactor))

            if not running:
                break

    print(_SEP)
    r = reactor.process.report()
    print(
        f"\n  Run summary:  "
        f"duration={r['time_h']:.1f} h  |  "
        f"X_final={r['X_g_L']:.1f} g/L  |  "
        f"DO_final={r['DO_mg_L']:.3f} mg/L  |  "
        f"V_final={r['V_L']:.2f} L"
    )


# ─────────────────────────────────────────────
# Scenario B — Overflow Cascade → Halt
# ─────────────────────────────────────────────

def scenario_b():
    _banner("SCENARIO B  ·  Overflow Cascade  →  Process Halt")
    print(
        "  Same initial conditions.  From t=7 h, CO₂ back-pressure accumulates in\n"
        "  the headspace (linear ramp: 1.0 → 2.1 atm over 2 h), modelling high-density\n"
        "  CO₂ evolution during exponential growth.\n"
        "  t≈8 h: pressure WARNING → reduce_feed issued.\n"
        "  t≈9 h: pressure reaches CRITICAL (>2 atm) while DO is also falling.\n"
        "  ReasonSystem matches the overflow cascade signature → halt.\n"
    )

    _hdr = _HEADER + f"  {'P (atm)':>8}"
    print(_SEP)
    print(_hdr)
    print(_SEP)

    with open(LOG_PATH, "a") as log:
        log.write("\n" + "=" * 72 + "\n")
        log.write("SCENARIO B  ·  Overflow Cascade  →  Process Halt\n")
        log.write("=" * 72 + "\n\n")
        reactor = SimReactor(SIM_PARAMS, INITIAL_CONDITIONS, INITIAL_PUMP_RPM,
                             log_file=log)

        def pressure_at(t: float) -> float:
            """CO₂ headspace ramp beginning at peak respiration rate."""
            if t < 7.0: return 1.0
            if t < 9.0: return 1.0 + (t - 7.0) * 0.55
            return 2.1

        for step in range(1, 301):
            P = pressure_at(reactor.process.time)
            reactor.tank.pressure.value = P

            running, actions = reactor.step()

            pct = reactor.process.DO / SIM_PARAMS["DO_star"] * 100
            is_critical = P >= 2.0 or pct < 15.0
            is_warning  = P >= 1.5 or pct < 35.0 or bool(actions)

            if is_critical:
                _event_block(reactor, "CRITICAL",
                             f"P={P:.2f} atm  |  DO={pct:.1f}%  |  "
                             f"actions: {', '.join(actions) or 'none'}",
                             pressure=P)
            elif is_warning:
                action_str = (f"  →  actions: {', '.join(actions)}"
                              if actions else "")
                _event_block(reactor, "WARNING",
                             f"P={P:.2f} atm  |  DO={pct:.1f}%{action_str}",
                             pressure=P)
            elif step % STATUS_EVERY == 0:
                p    = reactor.process
                pump = reactor.substrate_line.pump.speed
                print(
                    f"  {p.time:>6.1f}  {p.X:>9.2f}  {p.S:>9.2f}  "
                    f"{p.DO:>10.3f}  {pct:>5.1f}%  {p.V:>6.2f}  {pump:>5}  {P:>8.2f}"
                )

            if not running:
                break

    print(_SEP)
    r = reactor.process.report()
    print(
        f"\n  Run summary:  "
        f"duration={r['time_h']:.1f} h  |  "
        f"X_final={r['X_g_L']:.1f} g/L  |  "
        f"DO_final={r['DO_mg_L']:.3f} mg/L  |  "
        f"V_final={r['V_L']:.2f} L"
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    open(LOG_PATH, "w").close()   # clear log at start of each run
    scenario_a()
    scenario_b()