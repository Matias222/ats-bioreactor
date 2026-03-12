from main import Tank, SensorySystem, AirLine, GasExit, AntiFoamLine, BaseLine, AcidLine, SubstrateLine
import fault_graph
from Bioreactor_ODE import BioreactorProcess, feed_rate_from_speed, PARAMS

# Sensory System
class TAS_SensorySystem:

    def __init__(self, tank):
        self.sensor_system = SensorySystem(tank)

    def read_environment(self):
        return self.sensor_system.read_all_sensors()

# Presentation System
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


# Reason System
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

# Decision System
class DecisionSystem:
    def decide(self) -> tuple[bool, list]:
        return fault_graph.decide()


# Will System
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
        # Also ramp agitator toward reference speed to maximise kLa.
        # We step toward N_ref in 25 % increments so the change is gradual.
        current_rpm = self.motor.rpm
        target_rpm  = 400   # N_ref — produces kLa_ref when valve is open
        new_rpm = min(target_rpm, current_rpm + max(1, target_rpm // 4))
        if new_rpm != current_rpm:
            self.motor.set_speed(new_rpm)
            print(f"[WILL] Agitator speed: {current_rpm} → {new_rpm} rpm")

    def reduce_feed(self):
        current = self.substrate.pump.speed
        reduced = max(1, current // 2)
        print(f"[WILL] Reducing substrate pump speed: {current} → {reduced} rpm")
        self.substrate.pump.set_speed(reduced)


# INTELLECT SYSTEM
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
        return recent[-1] - recent[0]  # delta over window, not per-step

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
# Checked in step() independently of fault graph.
# ─────────────────────────────────────────────

# Minimum biomass before plateau detection is meaningful.
# Prevents false plateau detection during lag/startup phase.
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


# Autonomous Bioreactor System
class AutonomousBioreactor:

    def __init__(self):

        # Physical system
        self.tank = Tank()

        self.air_line = AirLine()
        self.gas_exit = GasExit()
        self.antifoam_line = AntiFoamLine()
        self.base_line = BaseLine()
        self.acid_line = AcidLine()
        self.substrate_line = SubstrateLine()

        # Process model
        self.process = BioreactorProcess(
            X0=0.1,   # g/L  — inoculum
            S0=2.0,   # g/L  — initial substrate
            DO0=8.0,  # mg/L — saturated
            V0=1.0,   # L    — initial volume
        )
        # Set initial pump speed and push process state to sensors
        self.substrate_line.pump.set_speed(5)
        self.tank.temperature.value = 37.0
        self.process.update_sensors(self.tank)

        # Initialize fault graph
        self.tank.motor.set_speed(400)   # start at N_ref → kLa = kLa_ref
        fault_graph.initialize_fault_graph()

        # TAS subsystems
        self.sensory = TAS_SensorySystem(self.tank)
        self.presentation = PresentationSystem()
        self.reason = ReasonSystem()
        self.decision = DecisionSystem()
        self.will = WillSystem(self)
        self.intellect = IntellectSystem()

    # Interface for event system
    @property
    def DO_sensor(self):
        return self.tank.oxygen

    @property
    def biomass_sensor(self):
        return self.tank.od

    @property
    def pressure_sensor(self):
        return self.tank.pressure


    # AUTONOMOUS LOOP
    def step(self, dt_hours: float = 0.1):
        # 0 — Advance process model
        F          = feed_rate_from_speed(self.substrate_line.pump.speed, PARAMS)
        T          = self.tank.temperature.value
        motor_rpm  = self.tank.motor.rpm
        valve_open = self.air_line.control_valve.is_open
        self.process.step(dt_hours, F, T, motor_rpm, valve_open)
        self.process.update_sensors(self.tank)
        print(f"[PROCESS] t={self.process.time:.2f}h | "
              f"X={self.process.X:.2f} g/L | "
              f"S={self.process.S:.2f} g/L | "
              f"DO={self.process.DO:.2f} mg/L | "
              f"V={self.process.V:.2f} L")

        # 1 — Sensory
        raw_data = self.sensory.read_environment()

        # 2 — Presentation
        formatted = self.presentation.format_data(raw_data)
        formatted["time_h"] = self.process.time

        # 3 — Intellect: store before reasoning so history is current
        self.intellect.store(formatted)

        # 4 — Reason
        self.reason.evaluate(self, self.intellect)

        # 5 — Decision
        running, actions = self.decision.decide()

        # 6 — Will: execute physical actions
        for action in actions:
            if action == "increase_airflow":
                self.will.increase_airflow()
            elif action == "reduce_feed":
                self.will.reduce_feed()
            elif action == "flag_sensor":
                print("[WILL] Sensor flagged. Continuing with caution.")

        # 7 — Healthy halt conditions
        if running:
            if is_plateaued(self.intellect):
                print("[END] Biomass plateau detected. Run complete.")
                return False
            if is_volume_limit_reached(self):
                print("[END] Volume limit reached. Run complete.")
                return False

        return running


# SIMULATION

def simulate(n_steps: int = 100, dt_hours: float = 0.1):
    """
    ODE-driven simulation loop.
    n_steps  — number of timesteps to run
    dt_hours — duration of each timestep in hours (default 6 min)
    Total simulated time = n_steps * dt_hours hours
    """
    reactor = AutonomousBioreactor()

    print(f"Starting ODE simulation: {n_steps} steps x {dt_hours}h = "
          f"{n_steps * dt_hours:.1f}h total")

    for i in range(n_steps):
        print(f"\n{'='*55}")
        print(f"STEP {i+1}/{n_steps}")
        print(f"{'='*55}")

        running = reactor.step(dt_hours)

        if not running:
            print("\n[SIMULATION] Process terminated.")
            return

    print("\n[SIMULATION] Run completed normally.")
    print(f"Final state: {reactor.process.report()}")


if __name__ == "__main__":
    simulate()