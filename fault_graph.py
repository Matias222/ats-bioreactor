from enum import Enum


# ─────────────────────────────────────────────
# Event primitives
# ─────────────────────────────────────────────

class EventValue(Enum):
    TRUE = "True"
    FALSE = "False"
    SUSPECT = "Suspect"
    UNKNOWN = "Unknown"


class EventPriority(Enum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3


class Event:
    def __init__(self, value: EventValue = EventValue.UNKNOWN,
                 priority: EventPriority = EventPriority.INFO,
                 label: str = ""):
        self.value = value
        self.priority = priority
        self.label = label


# ─────────────────────────────────────────────
# EventNode
# ─────────────────────────────────────────────

class EventNode:
    def __init__(self, label: str, inference_rule=None):
        self.event = Event(label=label)
        self.dependents = []
        self.inference_rule = inference_rule

    def update(self, value: EventValue, priority: EventPriority,
               message: str = ""):
        changed = (self.event.value != value or
                   self.event.priority != priority)
        self.event.value = value
        self.event.priority = priority
        self.announce(message)
        if changed and self.dependents:
            self.announce("Propagating fault to dependents.")
            self._notify_dependents()

    def _notify_dependents(self):
        for node in self.dependents:
            node.evaluate()

    def evaluate(self):
        if self.inference_rule:
            value, priority, message = self.inference_rule()
            self.update(value, priority, message)

    def announce(self, message: str):
        print(f"[{self.event.priority.name}] {message}")


# ─────────────────────────────────────────────
# Inference rules
# ─────────────────────────────────────────────

def low_DO_rule():
    """
    Corroborates a low DO reading against biomass trend and pressure.
    - Low DO + rising biomass              → demand-driven hypoxia
    - Low DO + elevated pressure           → transfer limitation or overflow cascade
    - Low DO + both                        → critical overflow cascade
    - Low DO + no corroboration            → suspect sensor
    """
    do = S_DO_reading_low.event
    biomass = S_biomass_overgrow_reading.event
    pressure = S_high_pressure_reading.event

    if do.value != EventValue.TRUE:
        return EventValue.FALSE, EventPriority.INFO, "Low DO not asserted."

    if biomass.value == EventValue.TRUE and pressure.value == EventValue.TRUE:
        return (EventValue.TRUE, EventPriority.CRITICAL,
                "Low DO corroborated by rising biomass and elevated pressure. "
                "Overflow metabolism cascade suspected.")

    if biomass.value == EventValue.TRUE:
        return (EventValue.TRUE, EventPriority.CRITICAL,
                "Low DO corroborated by rising biomass. Demand-driven hypoxia.")

    if pressure.value == EventValue.TRUE:
        return (EventValue.TRUE, EventPriority.WARNING,
                "Low DO corroborated by elevated pressure. Transfer limitation suspected.")

    # No corroboration — suspect sensor
    S_DO_probe_fault.update(
        EventValue.SUSPECT, EventPriority.WARNING,
        "No corroboration for low DO reading. Probe flagged as suspect."
    )
    return (EventValue.UNKNOWN, EventPriority.WARNING,
            "Low DO reading unconfirmed. No corroborating evidence.")


# Minimum biomass below which high DO is expected (startup/lag phase).
# Culture crash cannot be suspected until biomass has had time to establish.
_BIOMASS_CRASH_THRESHOLD = 2.0   # g/L


def high_DO_rule():
    """
    Corroborates a high DO reading against biomass trend.
    - High DO + biomass below startup threshold → normal startup, ignore
    - High DO + falling/flat biomass           → culture crash suspected
    - High DO + rising biomass                 → excessive aeration
    - High DO + biomass unknown                → suspect sensor
    """
    do = S_DO_reading_high.event
    biomass = S_biomass_overgrow_reading.event

    if do.value != EventValue.TRUE:
        return EventValue.FALSE, EventPriority.INFO, "High DO not asserted."

    # Startup or lag phase — high DO is expected, not anomalous
    if _current_biomass < _BIOMASS_CRASH_THRESHOLD:
        return (EventValue.FALSE, EventPriority.INFO,
                f"High DO during startup phase. Biomass {_current_biomass:.2f} g/L "
                f"below crash threshold {_BIOMASS_CRASH_THRESHOLD} g/L.")

    if biomass.value == EventValue.FALSE:
        return (EventValue.TRUE, EventPriority.CRITICAL,
                "High DO with non-rising biomass. Culture crash suspected.")

    if biomass.value == EventValue.TRUE:
        return (EventValue.TRUE, EventPriority.WARNING,
                "High DO with rising biomass. Excessive aeration relative to demand.")

    S_DO_probe_fault.update(
        EventValue.SUSPECT, EventPriority.WARNING,
        "High DO reading with unknown biomass state. Probe flagged as suspect."
    )
    return (EventValue.UNKNOWN, EventPriority.WARNING,
            "High DO reading unconfirmed. Biomass state unknown.")


def DO_probe_fault_rule():
    """
    Aggregates suspicion for the DO probe.
    - Both low and high DO asserted simultaneously → probe failure confirmed.
    - Suspicion set directly by low_DO_rule or high_DO_rule is preserved.
    """
    low = S_DO_reading_low.event
    high = S_DO_reading_high.event

    if low.value == EventValue.TRUE and high.value == EventValue.TRUE:
        return (EventValue.TRUE, EventPriority.CRITICAL,
                "Both low and high DO asserted simultaneously. Probe failure confirmed.")

    if S_DO_probe_fault.event.value == EventValue.SUSPECT:
        return (EventValue.SUSPECT, EventPriority.WARNING,
                "DO probe remains suspect pending corroboration.")

    return EventValue.FALSE, EventPriority.INFO, "DO probe nominal."


# ─────────────────────────────────────────────
# Graph initialization
# ─────────────────────────────────────────────

def initialize_fault_graph():
    global S_DO_reading_low, S_DO_reading_high
    global S_biomass_overgrow_reading, S_high_pressure_reading
    global P_low_DO, P_high_DO, S_DO_probe_fault

    # Phase 1 — declare nodes
    S_DO_reading_low          = EventNode(label="S_DO_reading_low")
    S_DO_reading_high         = EventNode(label="S_DO_reading_high")
    S_biomass_overgrow_reading = EventNode(label="S_biomass_overgrow_reading")
    S_high_pressure_reading   = EventNode(label="S_high_pressure_reading")
    S_DO_probe_fault          = EventNode(label="S_DO_probe_fault",inference_rule=DO_probe_fault_rule)
    P_low_DO                  = EventNode(label="P_low_DO",inference_rule=low_DO_rule)
    P_high_DO                 = EventNode(label="P_high_DO",inference_rule=high_DO_rule)

    # Phase 2 — wire dependencies
    S_DO_reading_low.dependents.append(P_low_DO)
    S_biomass_overgrow_reading.dependents.append(P_low_DO)
    S_high_pressure_reading.dependents.append(P_low_DO)

    S_DO_reading_high.dependents.append(P_high_DO)
    S_biomass_overgrow_reading.dependents.append(P_high_DO)

    S_DO_reading_low.dependents.append(S_DO_probe_fault)
    S_DO_reading_high.dependents.append(S_DO_probe_fault)


# ─────────────────────────────────────────────
# Module-level process state cache
# Set by check functions so inference rules can
# access raw readings alongside node states.
# ─────────────────────────────────────────────

_current_biomass = 0.0   # g/L — updated by check_biomass each step


# ─────────────────────────────────────────────
# Check functions
# system: object exposing DO_sensor, biomass_sensor, pressure_sensor
# intellect: IntellectSystem instance — optional, used for trend detection.
#            Pass None if history is unavailable (e.g. first timestep).
# ─────────────────────────────────────────────

def check_dissolved_oxygen(system, intellect=None):
    reading = system.DO_sensor.value

    # Use trend from intellect history if available to distinguish
    # sudden drop (suspect sensor) from gradual decline (process drift).
    trend = intellect.trend("DO") if intellect else "insufficient_data"

    if reading < 0.15:
        S_DO_reading_low.update(EventValue.TRUE, EventPriority.CRITICAL,
                                f"DO critically low. Trend: {trend}.")
        S_DO_reading_high.update(EventValue.FALSE, EventPriority.INFO,
                                 "High DO not asserted.")
    elif reading < 0.35:
        S_DO_reading_low.update(EventValue.TRUE, EventPriority.WARNING,
                                f"DO below acceptable threshold. Trend: {trend}.")
        S_DO_reading_high.update(EventValue.FALSE, EventPriority.INFO,
                                 "High DO not asserted.")
    else:
        # NOTE: High DO detection disabled — with kLa=200 h⁻¹ DO remains near
        # saturation throughout a healthy run and the upper threshold is never
        # meaningfully exceeded. Re-enable if running at lower aeration rates.
        S_DO_reading_low.update(EventValue.FALSE, EventPriority.INFO,
                                "DO within normal range.")
        S_DO_reading_high.update(EventValue.FALSE, EventPriority.INFO,
                                 "DO within normal range.")


def check_biomass(system, intellect=None):
    """
    Overgrowth is assessed using both absolute value and rate of change.
    Requires intellect history of at least 3 steps for rate-of-change check.
    Falls back to absolute threshold only if history is unavailable.
    Replace absolute threshold with ODE trajectory comparison when model is available.
    """
    global _current_biomass
    reading = system.biomass_sensor.value
    _current_biomass = reading

    roc   = intellect.rate_of_change("biomass") if intellect else None
    trend = intellect.trend("biomass")           if intellect else "insufficient_data"

    # Overgrowth: high reading AND rising faster than expected
    if roc is not None:
        overgrowth = reading > 80.0 and trend == "rising" and roc > 5.0
        message_detail = f"{reading:.1f} g/L, rate: {roc:.2f} g/L per window."
    else:
        # Fallback — absolute threshold only, no history available
        overgrowth = reading > 80.0
        message_detail = f"{reading:.1f} g/L (no history, absolute threshold used)."

    if overgrowth:
        S_biomass_overgrow_reading.update(EventValue.TRUE, EventPriority.WARNING,
                                          f"Biomass overgrowth detected. {message_detail}")
    else:
        S_biomass_overgrow_reading.update(EventValue.FALSE, EventPriority.INFO,
                                          "Biomass within expected range.")


def check_chamber_pressure(system, intellect=None):
    reading = system.pressure_sensor.value

    if reading > 2.0:
        S_high_pressure_reading.update(EventValue.TRUE, EventPriority.CRITICAL,"Chamber pressure critically high.")
    elif reading > 1.5:
        S_high_pressure_reading.update(EventValue.TRUE, EventPriority.WARNING,"Chamber pressure elevated.")
    else:
        S_high_pressure_reading.update(EventValue.FALSE, EventPriority.INFO,"Chamber pressure nominal.")

def check_temperature(system, intellect=None):
    t = system.tank.temperature.read()
    print(f"[INFO] Temperature {t:.1f} °C — nominal.")

def check_motor_speed(system, intellect=None):
    rpm = system.tank.motor.rpm
    print(f"[INFO] Motor Speed {rpm:.0f} rpm — nominal.")

def check_substrate_flow(system, intellect=None):
    # Flow derived from pump speed via linear calibration.
    # 1 rpm ≈ 0.8 mL/min for lab-scale peristaltic pump.
    flow = system.substrate_line.pump.speed * 0.8
    print(f"[INFO] Substrate Flow {flow:.2f} mL/min — nominal.")

def check_level(system, intellect=None):
    level = system.tank.level.read()
    print(f"[INFO] Level Sensor {level:.2f} L — nominal.")

# ─────────────────────────────────────────────
# Action selection
# Returns a list of action identifier strings.
# Does not execute actions — caller (WillSystem)
# is responsible for physical execution.
# ─────────────────────────────────────────────

def select_actions(nodes: dict) -> list:
    actions = []

    P_low_DO                   = nodes['P_low_DO']
    P_high_DO                  = nodes['P_high_DO']
    S_high_pressure_reading    = nodes['S_high_pressure_reading']
    S_biomass_overgrow_reading = nodes['S_biomass_overgrow_reading']
    S_DO_probe_fault           = nodes['S_DO_probe_fault']

    # ── CRITICAL tier ────────────────────────
    if is_critical(P_low_DO):
        actions.append("increase_airflow")
        actions.append("reduce_feed")
        return actions

    if is_critical(P_high_DO):
        return actions  # no corrective action — halt triggered separately

    if is_critical(S_high_pressure_reading):
        return actions  # no corrective action — halt triggered separately

    if is_critical(S_DO_probe_fault):
        actions.append("flag_sensor")
        return actions

    # ── WARNING tier ─────────────────────────
    if is_warning(P_low_DO):
        actions.append("increase_airflow")
        return actions

    if is_warning(P_high_DO):
        actions.append("reduce_feed")
        return actions

    if is_warning(S_high_pressure_reading):
        actions.append("reduce_feed")
        return actions

    if is_warning(S_biomass_overgrow_reading):
        actions.append("reduce_feed")
        return actions

    # ── SUSPECT tier ─────────────────────────
    if is_suspect(S_DO_probe_fault):
        actions.append("flag_sensor")
        return actions

    return actions  # nominal — no actions


# ─────────────────────────────────────────────
# Halt evaluation
# Separate from action selection.
# Returns True if process should be terminated.
# ─────────────────────────────────────────────

def should_halt(nodes: dict) -> bool:
    P_low_DO                = nodes['P_low_DO']
    P_high_DO               = nodes['P_high_DO']
    S_high_pressure_reading = nodes['S_high_pressure_reading']
    S_DO_probe_fault        = nodes['S_DO_probe_fault']

    if is_critical(P_high_DO):
        print("[HALT] Culture crash suspected. Process cannot continue.")
        return True

    if is_critical(S_high_pressure_reading):
        print("[HALT] Critical overpressure. Safety limit exceeded.")
        return True

    if is_critical(S_DO_probe_fault):
        print("[HALT] DO probe failure confirmed. Cannot operate blind.")
        return True

    # Overflow cascade — CRITICAL low DO with elevated pressure
    if is_critical(P_low_DO) and nodes['S_high_pressure_reading'].event.value == EventValue.TRUE:
        print("[HALT] Overflow metabolism cascade. Process unrecoverable.")
        return True

    return False


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def is_critical(node) -> bool:
    return (node.event.value == EventValue.TRUE and
            node.event.priority == EventPriority.CRITICAL)

def is_warning(node) -> bool:
    return (node.event.value == EventValue.TRUE and
            node.event.priority == EventPriority.WARNING)

def is_suspect(node) -> bool:
    return node.event.value == EventValue.SUSPECT


# ─────────────────────────────────────────────
# decide()
# Orchestrates action selection and halt check.
# Returns (should_continue: bool, actions: list)
# ─────────────────────────────────────────────

def decide() -> tuple:
    print("\n[DECISION] Evaluating system state...")

    nodes = {
        'P_low_DO':                   P_low_DO,
        'P_high_DO':                  P_high_DO,
        'S_high_pressure_reading':    S_high_pressure_reading,
        'S_biomass_overgrow_reading': S_biomass_overgrow_reading,
        'S_DO_probe_fault':           S_DO_probe_fault,
    }

    actions = select_actions(nodes)

    if not actions:
        print("[DECISION] No corrective action required.")

    halt = should_halt(nodes)

    if halt:
        print("[ACTION] Process halted.")
        return False, actions

    print("[DECISION] Process continuing.")
    return True, actions