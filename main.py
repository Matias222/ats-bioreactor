"""
main.py — Demonstration runner for the TAS-controlled fed-batch bioreactor.
─────────────────────────────────────────────────────────────────────────────
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
─────────────────────────────────────────────────────────────────────────────
"""

import copy
from bioreactor_ode import PARAMS
from tas import AutonomousBioreactor

# ─────────────────────────────────────────────
# Simulation-wide parameters
# ─────────────────────────────────────────────

SIM_PARAMS = copy.deepcopy(PARAMS)
SIM_PARAMS["kLa_ref"] = 50.0   # h⁻¹ at N_ref — low aeration to stress-test DO

INITIAL_CONDITIONS = dict(X0=0.5, S0=5.0, DO0=8.0, V0=1.0)
INITIAL_PUMP_RPM   = 12
INITIAL_TEMP       = 37.0
DT_HOURS           = 0.1       # 6-minute timesteps
STATUS_EVERY       = 10        # print a status row every N nominal steps
MAX_VOLUME         = 20.0      # L — simulation working volume limit

LOG_PATH = "./log.txt"


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

def _nominal_row(r: AutonomousBioreactor, extra: str = "") -> str:
    p    = r.process
    pump = r.substrate_line.pump.speed
    pct  = p.DO / SIM_PARAMS["DO_star"] * 100
    row  = (
        f"  {p.time:>6.1f}  {p.X:>9.2f}  {p.S:>9.2f}  "
        f"{p.DO:>10.3f}  {pct:>5.1f}%  {p.V:>6.2f}  {pump:>5}"
    )
    return row + (f"  ← {extra}" if extra else "")

def _event_block(r: AutonomousBioreactor, tag: str, detail: str, pressure: float = None):
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
        reactor = AutonomousBioreactor(
            params=SIM_PARAMS, ic=INITIAL_CONDITIONS,
            pump_rpm=INITIAL_PUMP_RPM, temp=INITIAL_TEMP,
            max_volume=MAX_VOLUME, log_file=log,
        )

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
        reactor = AutonomousBioreactor(
            params=SIM_PARAMS, ic=INITIAL_CONDITIONS,
            pump_rpm=INITIAL_PUMP_RPM, temp=INITIAL_TEMP,
            max_volume=MAX_VOLUME, log_file=log,
        )

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
