"""
tas.py — Thinking Autonomous System

TAS subsystems (Sensory, Presentation, Reason, Decision, Will, Intellect)
and the unified AutonomousBioreactor class that wires them together with
the ODE process model and fault graph.
"""

import sys
from hardware import (
    Tank, SensorySystem, AirLine, GasExit,
    AntiFoamLine, BaseLine, AcidLine, SubstrateLine,
)
import fault_graph
from bioreactor_ode import BioreactorProcess, feed_rate_from_speed, PARAMS


# ─────────────────────────────────────────────
# stdout filter: suppress routine lines from
# terminal but dump everything to log.txt
# ─────────────────────────────────────────────

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
      - writes ALL lines (including suppressed ones) to a log file

    Lines matching _SUPPRESS_PREFIXES and blank lines are hidden from
    the terminal but fully preserved in the log for post-run inspection.
    """

    def __init__(self, log_file):
        self._log = log_file

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = self
        self._buf = ""
        return self

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._log.write(line + "\n")
            suppress = (not line.strip()) or any(
                line.startswith(p) for p in _SUPPRESS_PREFIXES
            )
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
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


# ─────────────────────────────────────────────
# TAS Subsystems
# ─────────────────────────────────────────────

class TAS_SensorySystem:

    def __init__(self, tank):
        self.sensor_system = SensorySystem(tank)

    def read_environment(self):
        return self.sensor_system.read_all_sensors()


class PresentationSystem:
    """
    Converts raw sensor data into a standard format
    understood by all TAS subsystems.
    """
    def format_data(self, data):
        formatted = {
            "DO":          data["oxygen"],
            "biomass":     data["optical_density"],
            "pressure":    data["pressure"],
            "temperature": data["temperature"],
            "level":       data["fluid_level"],
        }
        return formatted


class ReasonSystem:
    """
    Checks the sensory data and updates nodes per sensor based on rules.
    """
    def evaluate(self, system, intellect):
        fault_graph.check_temperature(system)
        fault_graph.check_motor_speed(system)
        fault_graph.check_substrate_flow(system)
        fault_graph.check_level(system)
        fault_graph.check_dissolved_oxygen(system, intellect)
        fault_graph.check_biomass(system, intellect)
        fault_graph.check_chamber_pressure(system)


class DecisionSystem:
    def decide(self) -> tuple[bool, list]:
        return fault_graph.decide()


class WillSystem:
    """
    Executes physical actions in the reactor.
    """
    def __init__(self, reactor):
        self.air_line = reactor.air_line
        self.substrate = reactor.substrate_line
        self.motor = reactor.tank.motor

    def increase_airflow(self):
        print("[WILL] Opening air valve")
        self.air_line.control_valve.open()
        current_rpm = self.motor.rpm
        target_rpm = 400
        new_rpm = min(target_rpm, current_rpm + max(1, target_rpm // 4))
        if new_rpm != current_rpm:
            self.motor.set_speed(new_rpm)
            print(f"[WILL] Agitator speed: {current_rpm} → {new_rpm} rpm")

    def reduce_feed(self):
        current = self.substrate.pump.speed
        reduced = max(1, current // 2)
        print(f"[WILL] Reducing substrate pump speed: {current} → {reduced} rpm")
        self.substrate.pump.set_speed(reduced)


class IntellectSystem:
    """
    Stores historical and operational knowledge.
    """
    def __init__(self):
        self.history = []

    def store(self, data):
        self.history.append(data)

    def rate_of_change(self, variable: str, window: int = 3) -> float | None:
        """
        Returns approximate derivative of variable over last N steps.
        Returns None if insufficient history.
        """
        if len(self.history) < window:
            return None
        recent = [step[variable] for step in self.history[-window:]]
        return recent[-1] - recent[0]

    def trend(self, variable: str, window: int = 3) -> str:
        delta = self.rate_of_change(variable, window)
        if delta is None:
            return "insufficient_data"
        if delta > 0.05:
            return "rising"
        if delta < -0.05:
            return "falling"
        return "stable"


# ─────────────────────────────────────────────
# Healthy halt helpers
# ─────────────────────────────────────────────

_PLATEAU_BIOMASS_MIN = 5.0   # g/L

def is_plateaued(intellect, variable="biomass",
                 window=5, threshold=1.0) -> bool:
    """
    Returns True when biomass rate of change falls below threshold
    for the last N steps, indicating stationary phase.
    Only active once biomass has exceeded the startup threshold.
    """
    if len(intellect.history) < window:
        return False
    current_biomass = intellect.history[-1].get(variable, 0.0)
    if current_biomass < _PLATEAU_BIOMASS_MIN:
        return False
    roc = intellect.rate_of_change(variable, window)
    return roc is not None and abs(roc) < threshold


def is_volume_limit_reached(system, limit=Tank.max_level * 0.9) -> bool:
    """
    Returns True when tank level approaches working volume capacity.
    """
    return system.tank.level.value >= limit


# ─────────────────────────────────────────────
# Unified Autonomous Bioreactor
# ─────────────────────────────────────────────

_DEFAULT_IC = dict(X0=0.1, S0=2.0, DO0=8.0, V0=1.0)


class AutonomousBioreactor:
    """
    Wires all TAS subsystems and the ODE process model together.
    Accepts custom params/ICs so scenarios can be configured.
    """

    def __init__(self, params=None, ic=None, pump_rpm=5, temp=37.0,
                 max_volume=None, log_file=None):
        self.log_file = log_file
        self.max_volume = max_volume if max_volume is not None else Tank.max_level

        # Physical system
        self.tank           = Tank()
        self.air_line       = AirLine()
        self.gas_exit       = GasExit()
        self.antifoam_line  = AntiFoamLine()
        self.base_line      = BaseLine()
        self.acid_line      = AcidLine()
        self.substrate_line = SubstrateLine()

        # Process model
        ic = ic if ic is not None else _DEFAULT_IC.copy()
        self.params = params if params is not None else PARAMS.copy()
        self.process = BioreactorProcess(**ic, params=self.params)

        self.substrate_line.pump.set_speed(pump_rpm)
        self.tank.motor.set_speed(400)
        self.air_line.control_valve.open()
        self.tank.temperature.value = temp
        self.process.update_sensors(self.tank)

        # Initialize fault graph
        fault_graph.initialize_fault_graph()

        # TAS subsystems
        self.sensory      = TAS_SensorySystem(self.tank)
        self.presentation = PresentationSystem()
        self.reason       = ReasonSystem()
        self.decision     = DecisionSystem()
        self.will         = WillSystem(self)
        self.intellect    = IntellectSystem()

    # ── sensor proxies expected by fault_graph check functions ──
    @property
    def DO_sensor(self):
        return self.tank.oxygen
    @property
    def biomass_sensor(self):
        return self.tank.od
    @property
    def pressure_sensor(self):
        return self.tank.pressure

    def step(self, dt_hours: float = 0.1,
             verbose: bool = False) -> tuple[bool, list[str]]:
        """
        Advance one timestep.

        Returns
        -------
        (running, actions_taken)
        """
        F          = feed_rate_from_speed(self.substrate_line.pump.speed, self.params)
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
            if self.process.V >= self.max_volume * 0.9:
                print("\n[END] Volume limit reached. Run complete.")
                return False, actions_taken

        return running, actions_taken
