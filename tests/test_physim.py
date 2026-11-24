"""Acceptance tests for physim (tolerances per spec: 1%, 0.3% for RLC)."""
import numpy as np
import pytest

from physim import (
    Network, VoltageSource, Resistor, Capacitor, Inductor,
    DCMachine, Inertia, Damper, TorqueSource,
    ThermalMass, ThermalResistor, HeatSource,
)


# ---------------------------------------------------------------- Test 1
def test_rlc_step_response():
    """12 V step into series R=0.5, L=10 mH, C=1 mF; h=20 us, 20 ms."""
    net = Network()
    net.add_node("g", "electrical", ground=True)
    net.add_node("a", "electrical")
    net.add_node("b", "electrical")
    net.add_node("c", "electrical")
    net.add(VoltageSource("src", "a", "g", 12.0))
    net.add(Resistor("R", "a", "b", 0.5))
    net.add(Inductor("L", "b", "c", 10e-3))
    net.add(Capacitor("C", "c", "g", 1e-3))

    h, t_end = 20e-6, 20e-3
    ts, recs = net.run(t_end, h)
    vc = np.array([r[1]["c"] for r in recs])
    t = np.array(ts)

    V, R, L, C = 12.0, 0.5, 10e-3, 1e-3
    alpha = R / (2 * L)
    wd = np.sqrt(1.0 / (L * C) - alpha ** 2)
    vc_exact = V * (1 - np.exp(-alpha * t) * (np.cos(wd * t) + (alpha / wd) * np.sin(wd * t)))

    peak_err = np.max(np.abs(vc - vc_exact))
    assert peak_err < 0.05, f"RLC peak error {peak_err:.4f} V >= 0.05 V"


# ---------------------------------------------------------------- Test 2
def _motor_net(V):
    """400 V-class battery (V or callable) + Rb, DC machine, flywheel, damper, load."""
    net = Network()
    net.add_node("gnd", "electrical", ground=True)
    net.add_node("a", "electrical")      # battery terminal
    net.add_node("b", "electrical")      # machine terminal
    net.add_node("rg", "rotational", ground=True)
    net.add_node("sh", "rotational")     # shaft
    net.add(VoltageSource("batt", "a", "gnd", V))
    net.add(Resistor("Rb", "a", "b", 0.05))
    net.add(DCMachine("m", "b", "gnd", "sh", "rg", k=0.8))
    net.add(Inertia("J", "sh", 1.5))
    net.add(Damper("b", "sh", "rg", 0.02))
    net.add(TorqueSource("load", "sh", lambda t: -15.0 if t >= 1.0 else 0.0))
    return net


def test_motor_drive_steady_state():
    net = _motor_net(400.0)
    h = 2e-3
    ts, recs = net.run(5.0, h)
    last = recs[-1][2]
    I = last["m"]["I"]
    w = last["J"]["w"]
    # Analytic steady state: I = 31.20 A, w = 498 rad/s
    assert abs(I - 31.20) / 31.20 < 0.01, f"I={I:.3f} A"
    assert abs(w - 498.0) / 498.0 < 0.01, f"w={w:.3f} rad/s"


# ---------------------------------------------------------------- Test 3
def test_regenerative_braking():
    V = lambda t: 400.0 if t < 8.0 else (250.0 if t < 12.0 else 0.0)
    net = _motor_net(V)
    h, t_end = 2e-3, 20.0
    ts, recs = net.run(t_end, h)
    t = np.array(ts)
    # Bus current delivered by the battery = current through Rb from a to b
    # (the source's internal I has the opposite sign: positive = current into + terminal).
    I_bus = np.array([r[2]["Rb"]["I"] for r in recs])
    I_mach = np.array([r[2]["m"]["I"] for r in recs])
    V_batt = np.array([r[2]["batt"]["V"] for r in recs])
    P_bus = V_batt * I_bus          # power delivered by battery to the bus (>0 motoring)
    w = np.array([r[2]["J"]["w"] for r in recs])
    J = 1.5

    brake = t >= 8.0
    assert np.any(I_mach[brake] < 0), "expected negative (regenerative) current during braking"
    assert np.any(P_bus[brake] < 0), "expected negative bus power during braking"

    # Energy returned to the bus = -integral of negative bus power
    p_neg = np.minimum(P_bus, 0.0)
    e_returned = -np.trapz(p_neg, t)
    # Flywheel KE lost from start of braking (t=8 s) to end of run
    i8 = int(np.searchsorted(t, 8.0))
    ke_lost = 0.5 * J * (w[i8] ** 2 - w[-1] ** 2)

    assert e_returned > 0.0, "bus must absorb energy during braking"
    assert e_returned <= ke_lost, f"returned {e_returned:.0f} J > KE lost {ke_lost:.0f} J"
    ratio = e_returned / ke_lost
    assert 0.30 < ratio < 0.60, f"recovery ratio {ratio:.2f} (reference ~0.45)"


# ---------------------------------------------------------------- Test 4
def test_thermal_step():
    net = Network()
    net.add_node("amb", "thermal", ground=True)
    net.add_node("T1", "thermal")
    net.add(HeatSource("q", "T1", 10.0))
    net.add(ThermalMass("C", "T1", 800.0))
    net.add(ThermalResistor("Rth", "T1", "amb", 0.8))

    h, t_end = 0.5, 3000.0
    ts, recs = net.run(t_end, h)
    t = np.array(ts)
    T = np.array([r[1]["T1"] for r in recs])
    P, Rth = 10.0, 0.8
    T_exact = P * Rth * (1 - np.exp(-t / (800.0 * Rth)))
    peak_err = np.max(np.abs(T - T_exact))
    assert peak_err < 0.01, f"thermal peak error {peak_err:.5f} K >= 0.01 K"


# ---------------------------------------------------------------- Test 5
def test_cross_domain_coupling():
    """100 V bus -> Ra -> DC machine -> flywheel+friction; Ra loss heats a thermal mass."""
    net = Network()
    net.add_node("gnd", "electrical", ground=True)
    net.add_node("a", "electrical")
    net.add_node("b", "electrical")
    net.add_node("rg", "rotational", ground=True)
    net.add_node("sh", "rotational")
    net.add_node("amb", "thermal", ground=True)
    net.add_node("T1", "thermal")
    net.add(VoltageSource("bus", "a", "gnd", 100.0))
    net.add(Resistor("Ra", "a", "b", 0.2))
    net.add(DCMachine("m", "b", "gnd", "sh", "rg", k=0.5))
    net.add(Inertia("J", "sh", 0.05))
    net.add(Damper("fric", "sh", "rg", 0.002))
    net.add(ThermalMass("Cth", "T1", 800.0))
    net.add(ThermalResistor("Rth", "T1", "amb", 0.8))
    net.add(HeatSource("loss", "T1", lambda t, ctx: ctx.get("P_loss::Ra", 0.0)))

    h, t_end = 1e-3, 5.0
    ts, recs = net.run(t_end, h)

    I_ra = np.array([r[2]["Ra"]["I"] for r in recs])
    q_heat = np.array([r[2]["loss"]["Q"] for r in recs])
    Ra = 0.2
    # One-step delay: heat injected at step j equals I^2*Ra of step j-1
    for j in range(1, len(recs)):
        expect = Ra * I_ra[j - 1] ** 2
        assert abs(q_heat[j] - expect) <= 1e-6 * max(1.0, expect), f"delay mismatch at step {j}"
    # Steady-state loss ~ 0.13 W: exact analytic I = b*w/k = 0.8 A -> I^2*Ra = 0.128 W
    p_ss = recs[-1][2]["Ra"]["P_loss"]
    assert abs(p_ss - 0.128) / 0.128 < 0.01, f"steady loss {p_ss:.4f} W"
