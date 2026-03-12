"""
bioreactor_ode.py

Fed-batch bioreactor process model for E. coli cultivation.
Implements Monod kinetics with dissolved oxygen limitation
and temperature correction via the cardinal temperature model.

State variables:
    X  — biomass concentration       (g/L)
    S  — substrate concentration      (g/L)
    DO — dissolved oxygen             (mg/L)
    V  — culture volume               (L)

Inputs (set externally):
    F  — feed rate                    (L/h)  — set via pump speed
    T  — temperature                  (°C)   — read from tank sensor

The process is integrated forward using scipy solve_ivp (RK45).
After each step, update_sensors() writes true state into tank
sensor objects so the rest of the TAS sees realistic readings.
"""

from scipy.integrate import solve_ivp
import math


# ─────────────────────────────────────────────
# Parameters
# Literature values for E. coli fed-batch at lab scale.
# Adjust to match your simulation targets.
# ─────────────────────────────────────────────

PARAMS = {
    "mu_max":  0.8,     # h⁻¹       maximum specific growth rate
    "Ks":      0.05,    # g/L       substrate half-saturation constant
    "Ko":      0.2,     # mg/L      oxygen half-saturation constant
    "Yxs":     0.5,     # g/g       biomass yield on substrate
    "Yxo":     0.3,     # g/g       biomass yield on oxygen
    # Dynamic kLa model: kLa = kLa_ref * (N / N_ref)^kLa_alpha  (valve open)
    #                      kLa = kLa_surface                      (valve closed)
    # Calibrated so that at N_ref rpm with valve open → kLa_ref h⁻¹.
    # Literature range for lab-scale stirred tank: ~50–400 h⁻¹.
    "kLa_ref":     200.0,  # h⁻¹   kLa at reference agitation speed
    "N_ref":       400.0,  # rpm   reference agitation speed
    "kLa_alpha":   1.8,    #       power-law exponent (literature: 1.5–2.0)
    "kLa_surface": 3.0,    # h⁻¹   surface aeration when sparger valve is closed
    "DO_star": 8.0,     # mg/L      oxygen saturation at 37°C, 1 atm
    "S_feed":  300.0,   # g/L       substrate concentration in feed
    # Cardinal temperature model parameters for E. coli
    "T_min":   10.0,    # °C        minimum growth temperature
    "T_opt":   37.0,    # °C        optimal growth temperature
    "T_max":   44.0,    # °C        maximum growth temperature
    # Pump calibration — must match fault_graph.check_substrate_flow
    "pump_cal": 0.8,    # mL/min per rpm → converted to L/h in feed_rate_from_speed
}


# ─────────────────────────────────────────────
# Temperature correction
# Cardinal temperature model (Rosso et al. 1993).
# Returns a value in [0, 1] scaling µ_max.
# Returns 0 outside viable range.
# ─────────────────────────────────────────────

def temperature_factor(T: float, p: dict) -> float:
    T_min, T_opt, T_max = p["T_min"], p["T_opt"], p["T_max"]
    if T <= T_min or T >= T_max:
        return 0.0
    numerator   = (T - T_max) * (T - T_min) ** 2
    denominator = (T_opt - T_min) * (
        (T_opt - T_min) * (T - T_opt)
        - (T_opt - T_max) * (T_opt + T_min - 2 * T)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ─────────────────────────────────────────────
# Feed rate helper
# Converts pump speed (rpm) to feed rate (L/h).
# Calibration: pump_cal mL/min per rpm.
# ─────────────────────────────────────────────

def feed_rate_from_speed(rpm: float, p: dict) -> float:
    flow_mL_per_min = rpm * p["pump_cal"]
    return flow_mL_per_min * 60 / 1000   # L/h


# ─────────────────────────────────────────────
# Dynamic kLa helper
# ─────────────────────────────────────────────

def compute_kLa(motor_rpm: float, valve_open: bool, p: dict) -> float:
    """
    Compute the volumetric oxygen transfer coefficient for the current step.

    When the sparger valve is open, kLa scales with agitation via a
    normalised power law:

        kLa = kLa_ref * (N / N_ref) ^ kLa_alpha

    Normalising by N_ref ensures the result equals kLa_ref at the
    reference speed regardless of alpha, avoiding the calibration
    problem that arises from using raw rpm with a large exponent.

    When the valve is closed (surface aeration only), kLa_surface is
    returned — a small residual transfer from headspace diffusion and
    impeller-induced surface renewal.

    motor_rpm  — current agitator speed (rpm); clamped to >= 0
    valve_open — whether the sparger control valve is open
    p          — parameter dict (must contain kLa_ref, N_ref, kLa_alpha,
                  kLa_surface)
    """
    if not valve_open:
        return p["kLa_surface"]
    rpm = max(motor_rpm, 0.0)
    if rpm == 0.0:
        return p["kLa_surface"]
    return p["kLa_ref"] * (rpm / p["N_ref"]) ** p["kLa_alpha"]


# ─────────────────────────────────────────────
# ODE system
# ─────────────────────────────────────────────

def _odes(t, y, F, T, kLa, p):
    """
    Right-hand side of the ODE system.

    Arguments:
        t   — time (h), required by solve_ivp but unused explicitly
        y   — state vector [X, S, DO, V]
        F   — feed rate (L/h)
        T   — temperature (°C)
        kLa — oxygen transfer coefficient (h⁻¹), pre-computed for this step
        p   — parameter dict

    Returns:
        dydt — list of derivatives [dX/dt, dS/dt, dDO/dt, dV/dt]
    """
    X, S, DO, V = y

    # Clamp to physically meaningful values to prevent
    # solver from exploring negative concentrations
    X  = max(X,  0.0)
    S  = max(S,  0.0)
    DO = max(DO, 0.0)
    V  = max(V,  1e-6)

    # Growth rate — Monod with DO limitation and temperature correction
    fT  = temperature_factor(T, p)
    mu  = (p["mu_max"]
           * (S  / (p["Ks"] + S))
           * (DO / (p["Ko"] + DO))
           * fT)

    D = F / V   # dilution rate (h⁻¹)

    dX_dt  = mu * X - D * X
    dS_dt  = -(mu / p["Yxs"]) * X + D * (p["S_feed"] - S)
    dDO_dt = kLa * (p["DO_star"] - DO) - (mu / p["Yxo"]) * X
    dV_dt  = F

    return [dX_dt, dS_dt, dDO_dt, dV_dt]


# ─────────────────────────────────────────────
# BioreactorProcess
# ─────────────────────────────────────────────

class BioreactorProcess:
    """
    Owns the true process state and integrates it forward each timestep.

    Usage:
        process = BioreactorProcess()
        process.update_sensors(tank)          # push initial state to sensors

        # each step:
        T = tank.temperature.value
        F = feed_rate_from_speed(pump.speed, PARAMS)
        process.step(dt_hours, F, T)
        process.update_sensors(tank)          # sensors now reflect new state
    """

    def __init__(self, X0=0.1, S0=20.0, DO0=8.0, V0=1.0,
                 params: dict = None):
        """
        Initial conditions:
            X0   — initial biomass      (g/L)   default: inoculum
            S0   — initial substrate    (g/L)   default: batch medium
            DO0  — initial dissolved O₂ (mg/L)  default: saturated
            V0   — initial volume       (L)     default: 1 L
        """
        self.state  = [X0, S0, DO0, V0]
        self.time   = 0.0
        self.params = params if params is not None else PARAMS.copy()

    # ── Accessors ─────────────────────────────

    @property
    def X(self)  -> float: return self.state[0]

    @property
    def S(self)  -> float: return self.state[1]

    @property
    def DO(self) -> float: return self.state[2]

    @property
    def V(self)  -> float: return self.state[3]

    # ── Integration ───────────────────────────

    def step(self, dt_hours: float, F: float, T: float,
             motor_rpm: float = 400.0, valve_open: bool = True):
        """
        Integrate ODEs forward by dt_hours.

        dt_hours   — timestep duration in hours
        F          — feed rate (L/h) for this timestep
        T          — temperature (°C) for this timestep
        motor_rpm  — agitator speed (rpm); used to compute dynamic kLa
        valve_open — sparger valve state; used to compute dynamic kLa
        """
        kLa = compute_kLa(motor_rpm, valve_open, self.params)
        t_span = (self.time, self.time + dt_hours)

        result = solve_ivp(
            fun=_odes,
            t_span=t_span,
            y0=self.state,
            args=(F, T, kLa, self.params),
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
        )

        if result.success:
            self.state = [max(v, 0.0) for v in result.y[:, -1]]
            self.time += dt_hours
        else:
            print(f"[WARNING] [BioreactorProcess] ODE solver failed at t={self.time:.2f}h: "
                  f"{result.message}")

    # ── Sensor interface ───────────────────────

    def update_sensors(self, tank):
        """
        Write true process state into tank sensor objects.
        Called after each step so the TAS reads realistic values.

        Dissolved oxygen converted from mg/L to fraction of saturation
        to match the [0, 1] range expected by check_dissolved_oxygen.
        """
        tank.od.value          = self.X
        tank.oxygen.value      = self.DO / self.params["DO_star"]  # → fraction
        tank.level.value       = self.V
        # Pressure is not modeled in the ODE — left to simulation or manual set
        # Temperature is a controlled input — not written back

    def report(self) -> dict:
        """Returns current true state as a readable dict."""
        return {
            "time_h":  round(self.time, 3),
            "X_g_L":   round(self.X,    4),
            "S_g_L":   round(self.S,    4),
            "DO_mg_L": round(self.DO,   4),
            "V_L":     round(self.V,    4),
        }