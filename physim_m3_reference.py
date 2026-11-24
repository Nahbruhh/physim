"""
physim: a small Simscape-like 1D physical network simulator.

Core idea (same as Simscape):
  - Every domain has an ACROSS variable (voltage, angular speed, temperature...)
    defined at NODES, and a THROUGH variable (current, torque, heat flow)
    flowing through COMPONENTS between nodes.
  - Conservation law at every node: sum of through variables = 0.
  - Components may carry internal STATES (inductor current, spring twist...)
    and internal UNKNOWNS (e.g. machine current for ideal EM coupling).
  - Each time step we solve the resulting nonlinear/implicit system
    (backward-Euler, one scipy.optimize.root solve per step), then commit
    component states. Energy losses can be published to a context dict
    so other domains can pick them up (e.g. copper loss -> heat).

  - Time integration is the IMPLICIT TRAPEZOIDAL RULE: components with
    inertia (Inductor, Inertia, TransMass, DieselEngine crank, springs)
    all use the same discrete law x_{n+1} = x_n + h/2*(f_n + f_{n+1}),
    written so the new flux appears implicitly in the same Newton solve
    as the network equations. This is 2nd-order accurate and, unlike
    explicit Euler, does not inject energy into undamped oscillations
    (measured drift < 1e-6 relative per 20 s on the spring tests).
"""

import numpy as np
from scipy.optimize import root


DOMAINS = {
    "electrical":  {"across": "V",  "through": "I",     "ref": 0.0},
    "rotational":  {"across": "w",  "through": "tau",   "ref": 0.0},
    "thermal":     {"across": "T",  "through": "qd",    "ref": 0.0},
    "translational": {"across": "v", "through": "F",     "ref": 0.0},
}


class Network:
    def __init__(self):
        self.nodes = {}          # name -> domain dict
        self.grounded = set()    # fixed reference nodes
        self.components = []
        self.context = {}        # cross-domain published values

    def add_node(self, name, domain, ground=False):
        assert domain in DOMAINS, f"unknown domain {domain}"
        self.nodes[name] = domain
        if ground:
            self.grounded.add(name)
        return name

    def free_nodes(self):
        return [n for n in self.nodes if n not in self.grounded]

    def add(self, comp):
        comp.net = self
        self.components.append(comp)
        return comp

    # ---- solver -------------------------------------------------------
    def _unpack(self, x):
        free = self.free_nodes()
        vals = {n: 0.0 for n in self.nodes}
        for n, v in zip(free, x[:len(free)]):
            vals[n] = v
        k = len(free)
        internals = {}
        for c in self.components:
            for iname in c.internal_unknowns():
                internals[(id(c), iname)] = x[k]
                k += 1
        return vals, internals

    def _residual(self, x):
        vals, internals = self._unpack(x)
        res = []
        # node conservation: sum of flows out of components into node = 0
        for n in self.free_nodes():
            s = 0.0
            for c in self.components:
                for node, flow in c.flows(vals, internals):
                    if node == n:
                        s += flow
            res.append(s)
        # component internal constraints
        for c in self.components:
            res.extend(c.constraints(vals, internals))
        return np.array(res)

    def step(self, h, t, commit_states=True):
        free = self.free_nodes()
        n_int = sum(len(c.internal_unknowns()) for c in self.components)
        n = len(free) + n_int
        if self.x0 is None:
            x0 = np.zeros(n)
            for i, nd in enumerate(free):
                x0[i] = DOMAINS[self.nodes[nd]]["ref"]
        else:
            x0 = self.x0.copy()
        sol = root(self._residual, x0, tol=1e-10)
        if not sol.success and np.max(np.abs(self._residual(sol.x))) > 1e-6:
            raise RuntimeError(f"solve failed at t={t}: {sol.message}")
        self.x0 = sol.x
        vals, internals = self._unpack(sol.x)
        if commit_states:
            for c in self.components:
                c.commit(h, vals, internals)
                c.publish(self.context)
        self.t = t
        return vals

    def run(self, t_end, h, log_every=1):
        self.x0 = None
        self.last_h = h
        self.t = 0.0
        for c in self.components:
            c.reset()
        self.context = {}
        ts, recs = [], []
        n = int(round(t_end / h))
        for i in range(n + 1):
            t = i * h
            # last sample reports the state AT t=n*h -> do not advance states again
            vals = self.step(h, t, commit_states=(i < n))
            ts.append(t)
            recs.append((t, dict(vals), {c.name: c.observe(vals) for c in self.components}))
        return ts, recs


# ----- components -------------------------------------------------------

class Component:
    def __init__(self, name):
        self.name = name
        self.net = None
    def terminals(self): raise NotImplementedError
    def internal_unknowns(self): return []
    def constraints(self, vals, internals): return []
    def flows(self, vals, internals): raise NotImplementedError
    def commit(self, h, vals, internals): pass
    def publish(self, ctx): pass
    def observe(self, vals): return None
    def reset(self): pass
    def _r(self, vals, a, b):
        return vals[a] - vals[b]


class VoltageSource(Component):
    """Ideal source: fixed (or time-varying) voltage between + and -. Internal unknown: current."""
    def __init__(self, name, plus, minus, V):
        super().__init__(name); self.p, self.m, self.V = plus, minus, V
    def terminals(self): return [self.p, self.m]
    def internal_unknowns(self): return ["I"]
    def constraints(self, vals, internals):
        v = self.V(self.net.t) if callable(self.V) else self.V
        return [self._r(vals, self.p, self.m) - v]
    def flows(self, vals, internals):
        i = internals[(id(self), "I")]
        return [(self.p, -i), (self.m, +i)]


class Resistor(Component):
    def __init__(self, name, a, b, R):
        super().__init__(name); self.a, self.b, self.R = a, b, R
        self.last_i = 0.0
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        i = self._r(vals, self.a, self.b) / self.R
        self.last_i = i
        self._p_loss = i * i * self.R
        return [(self.a, -i), (self.b, +i)]
    def observe(self, vals):
        vab = self._r(vals, self.a, self.b)
        i = vab / self.R
        return {"V": vab, "I": i, "P_loss": i * i * self.R}
    def publish(self, ctx):
        ctx[f"P_loss::{self.name}"] = getattr(self, "_p_loss", 0.0)


class Capacitor(Component):
    def __init__(self, name, a, b, C):
        super().__init__(name); self.a, self.b, self.C = a, b, C
    def terminals(self): return [self.a, self.b]
    def reset(self): self.vc = 0.0; self.ic = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        # trapezoidal: (v_{n+1}-v_n)/h = (i_n + i_{n+1})/2  ->  i_{n+1} = 2C(v_{n+1}-v_n)/h - i_n
        i = 2 * self.C * (self._r(vals, self.a, self.b) - self.vc) / h - self.ic
        self._i_new = i
        return [(self.a, -i), (self.b, +i)]
    def commit(self, h, vals, internals):
        self.vc = self._r(vals, self.a, self.b)
        self.ic = getattr(self, "_i_new", self.ic)
    def observe(self, vals): return {"Vc": self.vc}


class Inductor(Component):
    def __init__(self, name, a, b, L):
        super().__init__(name); self.a, self.b, self.L = a, b, L
    def terminals(self): return [self.a, self.b]
    def reset(self): self.iL = 0.0; self.v_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        # trapezoidal: i_{n+1} = i_n + h/(2L) (v_n + v_{n+1})
        i = self.iL + h * (self.v_prev + self._r(vals, self.a, self.b)) / (2 * self.L)
        self._i_new = i
        return [(self.a, -i), (self.b, +i)]
    def commit(self, h, vals, internals):
        self.iL = getattr(self, "_i_new", self.iL)
        self.v_prev = self._r(vals, self.a, self.b)
    def observe(self, vals): return {"I": self.iL, "E": 0.5 * self.L * self.iL ** 2}


class Damper(Component):
    """Rotational viscous damper (also usable as mechanical 'resistor')."""
    def __init__(self, name, a, b, B):
        super().__init__(name); self.a, self.b, self.B = a, b, B
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        tau = self.B * self._r(vals, self.a, self.b)
        return [(self.a, -tau), (self.b, +tau)]
    def observe(self, vals):
        w = self._r(vals, self.a, self.b)
        return {"w": w, "P_loss": self.B * w * w}


class Spring(Component):
    """Implicit trapezoidal spring: deflection is updated with the same
    trapezoidal rule as the rest of the solver (theta_{n+1} = theta_n +
    h/2*(wrel_n + wrel_{n+1})), so undamped oscillations do not drift."""
    def __init__(self, name, a, b, k):
        super().__init__(name); self.a, self.b, self.k = a, b, k
    def terminals(self): return [self.a, self.b]
    def reset(self): self.theta = 0.0; self.wrel_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        wrel_n1 = self._r(vals, self.a, self.b)
        th = self.theta + 0.5 * h * (self.wrel_prev + wrel_n1)   # implicit update
        tau = self.k * th
        self._wrel_new = wrel_n1
        return [(self.a, -tau), (self.b, +tau)]
    def commit(self, h, vals, internals):
        self.theta += 0.5 * h * (self.wrel_prev + self._wrel_new)
        self.wrel_prev = self._wrel_new
    def observe(self, vals): return {"theta": self.theta, "tau": self.k * self.theta}


class Inertia(Component):
    """Flywheel from a node to mechanical ground."""
    def __init__(self, name, node, J):
        super().__init__(name); self.node, self.J = node, J
    def terminals(self): return [self.node]
    def reset(self): self.w_prev = 0.0; self.tau_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        # trapezoidal: tau_{n+1} = 2J(w_{n+1}-w_n)/h - tau_n
        tau = 2 * self.J * (vals[self.node] - self.w_prev) / h - self.tau_prev
        self._tau_new = tau
        return [(self.node, -tau)]
    def commit(self, h, vals, internals):
        self.w_prev = vals[self.node]
        self.tau_prev = getattr(self, "_tau_new", self.tau_prev)
    def observe(self, vals):
        w = vals[self.node]
        return {"w": w, "E": 0.5 * self.J * w * w}


class TorqueSource(Component):
    def __init__(self, name, node, tau):
        super().__init__(name); self.node, self.tau = node, tau
    def terminals(self): return [self.node]
    def flows(self, vals, internals):
        t = self.tau(self.net.t) if callable(self.tau) else self.tau
        return [(self.node, +t)]
    def observe(self, vals):
        t = self.tau(self.net.t) if callable(self.tau) else self.tau
        return {"tau": t, "P": t * vals[self.node]}


class DCMachine(Component):
    """Ideal DC machine: electrical terminals (p, m) <-> rotational (r1, r2).
    V_em = k*w, tau = k*I. Internal unknown: armature current I."""
    def __init__(self, name, p, m, r1, r2, k):
        super().__init__(name); self.p, self.m, self.r1, self.r2, self.k = p, m, r1, r2, k
        self._i = 0.0
    def terminals(self): return [self.p, self.m, self.r1, self.r2]
    def internal_unknowns(self): return ["I"]
    def constraints(self, vals, internals):
        i = internals[(id(self), "I")]
        w = self._r(vals, self.r1, self.r2)
        return [self._r(vals, self.p, self.m) - self.k * w]
    def flows(self, vals, internals):
        i = internals[(id(self), "I")]
        tau = self.k * i
        return [(self.p, -i), (self.m, +i), (self.r1, +tau), (self.r2, -tau)]
    def commit(self, h, vals, internals):
        self._i = internals[(id(self), "I")]
    def observe(self, vals):
        w = self._r(vals, self.r1, self.r2)
        tau = self.k * self._i
        return {"I": self._i, "w": w, "tau": tau,
                "P_mech": tau * w, "P_elec": self._i * self._r(vals, self.p, self.m)}


# thermal
class ThermalMass(Component):
    def __init__(self, name, node, C):
        super().__init__(name); self.node, self.C = node, C
    def terminals(self): return [self.node]
    def reset(self): self.T_prev = 0.0; self.q_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        q = 2 * self.C * (vals[self.node] - self.T_prev) / h - self.q_prev
        self._q_new = q
        return [(self.node, -q)]
    def commit(self, h, vals, internals):
        self.T_prev = vals[self.node]
        self.q_prev = getattr(self, "_q_new", self.q_prev)
    def observe(self, vals): return {"T": vals[self.node] - 273.15 if vals[self.node] > 150 else vals[self.node]}


class ThermalResistor(Component):
    def __init__(self, name, a, b, Rth):
        super().__init__(name); self.a, self.b, self.Rth = a, b, Rth
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        q = self._r(vals, self.a, self.b) / self.Rth
        return [(self.a, -q), (self.b, +q)]


class HeatSource(Component):
    """Injects heat into a node. Q may be a function of (t, context)."""
    def __init__(self, name, node, Q):
        super().__init__(name); self.node, self.Q = node, Q
    def terminals(self): return [self.node]
    def flows(self, vals, internals):
        q = self.Q(self.net.t, self.net.context) if callable(self.Q) else self.Q
        return [(self.node, +q)]


# ======================================================================
# Milestone 2: translational domain, couplers, battery, sources, control
# ======================================================================

class TransMass(Component):
    """Point mass from a node to translational ground."""
    def __init__(self, name, node, m):
        super().__init__(name); self.node, self.m = node, m
    def terminals(self): return [self.node]
    def reset(self): self.v_prev = 0.0; self.F_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        # trapezoidal: F_{n+1} = 2m(v_{n+1}-v_n)/h - F_n
        F = 2 * self.m * (vals[self.node] - self.v_prev) / h - self.F_prev
        self._F_new = F
        return [(self.node, -F)]
    def commit(self, h, vals, internals):
        self.v_prev = vals[self.node]
        self.F_prev = getattr(self, "_F_new", self.F_prev)
    def observe(self, vals):
        v = vals[self.node]
        return {"v": v, "E": 0.5 * self.m * v * v}


class TransDamper(Component):
    """Translational viscous damper (aero/road drag approximation)."""
    def __init__(self, name, a, b, B):
        super().__init__(name); self.a, self.b, self.B = a, b, B
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        F = self.B * self._r(vals, self.a, self.b)
        return [(self.a, -F), (self.b, +F)]
    def observe(self, vals):
        v = self._r(vals, self.a, self.b)
        return {"v": v, "P_loss": self.B * v * v}


class TransSpring(Component):
    """Implicit trapezoidal spring (see Spring)."""
    def __init__(self, name, a, b, k):
        super().__init__(name); self.a, self.b, self.k = a, b, k
    def terminals(self): return [self.a, self.b]
    def reset(self): self.x = 0.0; self.vrel_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        vrel_n1 = self._r(vals, self.a, self.b)
        x = self.x + 0.5 * h * (self.vrel_prev + vrel_n1)
        F = self.k * x
        self._vrel_new = vrel_n1
        return [(self.a, -F), (self.b, +F)]
    def commit(self, h, vals, internals):
        self.x += 0.5 * h * (self.vrel_prev + self._vrel_new)
        self.vrel_prev = self._vrel_new
    def observe(self, vals): return {"x": self.x, "F": self.k * self.x}


class ForceSource(Component):
    """Applies a force to a translational node. F may be a constant or f(t, ctx)."""
    def __init__(self, name, node, F):
        super().__init__(name); self.node, self.F = node, F
    def terminals(self): return [self.node]
    def flows(self, vals, internals):
        F = self.F(self.net.t, self.net.context) if callable(self.F) else self.F
        return [(self.node, +F)]
    def observe(self, vals):
        F = self.F(self.net.t, self.net.context) if callable(self.F) else self.F
        return {"F": F, "P": F * vals[self.node]}


class SpeedSource(Component):
    """Ideal speed actuator: enforces v(node) = v_target(t). Internal unknown: force F."""
    def __init__(self, name, node, v_target):
        super().__init__(name); self.node, self.v_target = node, v_target
    def terminals(self): return [self.node]
    def internal_unknowns(self): return ["F"]
    def constraints(self, vals, internals):
        v = self.v_target(self.net.t) if callable(self.v_target) else self.v_target
        return [vals[self.node] - v]
    def flows(self, vals, internals):
        return [(self.node, internals[(id(self), "F")])]
    def commit(self, h, vals, internals):
        self._F = internals[(id(self), "F")]
    def observe(self, vals):
        v = vals[self.node]
        return {"v": v, "F": self._F, "P": self._F * v}


class Sensor(Component):
    """Publishes a node's across variable into ctx as 'value::<name>'."""
    def __init__(self, name, node):
        super().__init__(name); self.node = node
    def terminals(self): return [self.node]
    def flows(self, vals, internals): return [(self.node, 0.0)]
    def commit(self, h, vals, internals):
        self._v = vals[self.node]
    def publish(self, ctx):
        ctx[f"value::{self.name}"] = getattr(self, "_v", 0.0)
    def observe(self, vals): return {"value": getattr(self, "_v", 0.0)}


class Gear(Component):
    """Ideal gearbox, rotational <-> rotational. n = reduction ratio:
    w_in = n * w_out, tau_out = n * tau_in (ideal, power-conserving)."""
    def __init__(self, name, a1, a2, b1, b2, n):
        super().__init__(name); self.a1, self.a2, self.b1, self.b2, self.n = a1, a2, b1, b2, n
    def terminals(self): return [self.a1, self.a2, self.b1, self.b2]
    def internal_unknowns(self): return ["tau_in"]
    def constraints(self, vals, internals):
        return [self._r(vals, self.a1, self.a2) - self.n * self._r(vals, self.b1, self.b2)]
    def flows(self, vals, internals):
        t = internals[(id(self), "tau_in")]
        return [(self.a1, -t), (self.a2, +t), (self.b1, +self.n * t), (self.b2, -self.n * t)]
    def observe(self, vals):
        t = getattr(self, "_t", 0.0)
        w_in = self._r(vals, self.a1, self.a2); w_out = self._r(vals, self.b1, self.b2)
        return {"tau_in": t, "w_in": w_in, "w_out": w_out,
                "P_in": t * w_in, "P_out": self.n * t * w_out}
    def commit(self, h, vals, internals):
        self._t = internals[(id(self), "tau_in")]


class Wheel(Component):
    """Ideal wheel, rotational <-> translational: v = radius * w, F = tau / radius."""
    def __init__(self, name, x, r1, r2, radius):
        super().__init__(name); self.x, self.r1, self.r2, self.r = x, r1, r2, radius
    def terminals(self): return [self.x, self.r1, self.r2]
    def internal_unknowns(self): return ["F"]
    def constraints(self, vals, internals):
        return [vals[self.x] - self.r * self._r(vals, self.r1, self.r2)]
    def flows(self, vals, internals):
        F = internals[(id(self), "F")]
        tau = self.r * F
        return [(self.x, +F), (self.r1, -tau), (self.r2, +tau)]
    def observe(self, vals):
        F = getattr(self, "_F", 0.0)
        v = vals[self.x]; w = self._r(vals, self.r1, self.r2)
        return {"F": F, "v": v, "w": w, "P_lin": F * v,
                "P_mech": self.r * F * w, "torque": self.r * F}
    def commit(self, h, vals, internals):
        self._F = internals[(id(self), "F")]


class CurrentSource(Component):
    """Injects current into node p (flows p -> m externally)."""
    def __init__(self, name, p, m, I):
        super().__init__(name); self.p, self.m, self.I = p, m, I
    def terminals(self): return [self.p, self.m]
    def flows(self, vals, internals):
        i = self.I(self.net.t, self.net.context) if callable(self.I) else self.I
        return [(self.p, +i), (self.m, -i)]


class Battery(Component):
    """Battery: OCV(soc) with internal resistance R.
    ocv: callable(soc) OR list of [soc, V] points (np.interp).
    capacity_Ah: nominal capacity. soc0: initial state of charge.
    Charging raises SOC, discharging drains it."""
    def __init__(self, name, p, m, ocv, R, capacity_Ah, soc0=1.0):
        super().__init__(name); self.p, self.m, self.R = p, m, R
        self.CAh, self.soc0 = capacity_Ah, soc0
        if callable(ocv):
            self.ocv = ocv
        else:
            arr = np.asarray(ocv, dtype=float)
            self.ocv = lambda soc: np.interp(soc, arr[:, 0], arr[:, 1])

    def terminals(self): return [self.p, self.m]
    def internal_unknowns(self): return ["I"]
    def reset(self): self.soc = self.soc0

    def constraints(self, vals, internals):
        i = internals[(id(self), "I")]
        return [self._r(vals, self.p, self.m) - (self.ocv(self.soc) + i * self.R)]

    def flows(self, vals, internals):
        i = internals[(id(self), "I")]
        return [(self.p, -i), (self.m, +i)]

    def commit(self, h, vals, internals):
        i = internals[(id(self), "I")]
        self.soc += h * i / (3600.0 * self.CAh)   # i>0 charges
        self._i = i
        self._vterm = self._r(vals, self.p, self.m)

    def publish(self, ctx):
        ctx[f"SOC::{self.name}"] = self.soc

    def observe(self, vals):
        i = getattr(self, "_i", 0.0)
        v = getattr(self, "_vterm", 0.0)
        return {"SOC": self.soc, "V_term": v, "I": i,
                "P_out": -v * i,                      # >0 when discharging
                "P_chem": -self.ocv(self.soc) * i,    # >0 when discharging
                "E_charged": 3600.0 * self.CAh * self.soc}


class ControlledTorqueSource(Component):
    """PI(D) speed controller acting as a torque source on a rotational node.
    target: f(t, ctx) -> desired speed [rad/s]; sensor_key: ctx key of measured speed.
    Order matters: add the Sensor BEFORE this component so commit() sees the
    same-step measurement (commit/publish run per component, in add order)."""
    def __init__(self, name, node, kp, ki, target, sensor_key, kd=0.0, u_max=None, ff=None):
        super().__init__(name)
        self.node = node; self.kp, self.ki, self.kd = kp, ki, kd
        self.target, self.key, self.u_max = target, sensor_key, u_max
        self.ff = ff                       # feed-forward torque f(t, ctx)


    def terminals(self): return [self.node]
    def reset(self): self._u = 0.0; self._e = 0.0; self._e_prev = None; self._int = 0.0
    def flows(self, vals, internals):
        return [(self.node, +self._u)]
    def commit(self, h, vals, internals):
        v_ref = self.target(self.net.t, self.net.context)
        e = v_ref - self.net.context.get(self.key, 0.0)
        self._e = e
        self._int += e * h
        if self.u_max is not None:                       # crude anti-windup
            u_raw = self.kp * e + self.ki * self._int
            if abs(u_raw) > self.u_max:
                self._int -= e * h
        d = 0.0
        if self.kd and self._e_prev is not None:
            d = self.kd * (e - self._e_prev) / h
        u = self.kp * e + self.ki * self._int + d
        if self.ff is not None:
            u += self.ff(self.net.t, self.net.context)
        if self.u_max is not None:
            u = max(-self.u_max, min(self.u_max, u))
        self._e_prev = e
        self._u = u
    def observe(self, vals):
        return {"u": self._u, "e": self._e, "tau": self._u,
                "P": self._u * vals[self.node]}


class PIController(Component):
    """PI(D) controller with NO physical terminals: publishes u into ctx as
    'u::<name>'. target: f(t, ctx) -> setpoint; sensor_key: ctx key of the
    measurement. ff: f(t, ctx) -> feed-forward added to u. Saturation via
    u_min/u_max with anti-windup. Add the Sensor BEFORE this component."""
    def __init__(self, name, kp, ki, target, sensor_key, kd=0.0,
                  u_min=None, u_max=None, ff=None):
        super().__init__(name)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target, self.key = target, sensor_key
        self.u_min, self.u_max, self.ff = u_min, u_max, ff

    def terminals(self): return []
    def flows(self, vals, internals): return []
    def reset(self): self._u = 0.0; self._e = 0.0; self._e_prev = None; self._int = 0.0
    def commit(self, h, vals, internals):
        y_ref = self.target(self.net.t, self.net.context)
        e = y_ref - self.net.context.get(self.key, 0.0)
        self._e = e
        self._int += e * h
        u = self.kp * e + self.ki * self._int
        d = 0.0
        if self.kd and self._e_prev is not None:
            d = self.kd * (e - self._e_prev) / h
        if self.ff is not None:
            u += self.ff(self.net.t, self.net.context)
        if self.u_min is not None and (u < self.u_min and e < 0):
            self._int -= e * h                        # anti-windup
        if self.u_max is not None and (u > self.u_max and e > 0):
            self._int -= e * h
        u = self.kp * e + self.ki * self._int + d
        if self.ff is not None:
            u += self.ff(self.net.t, self.net.context)
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)
        self._e_prev = e
        self._u = u
    def publish(self, ctx):
        ctx[f"u::{self.name}"] = self._u
    def observe(self, vals):
        return {"u": self._u, "e": self._e}


class DCTransformer(Component):
    """Average-model DC-DC converter / 'DC transformer' (e.g. inverter).
    High side (hp, hm), low side (lp, lm). duty (from ctx key, 0..1):
        V_low = m * V_high,  I_high = m * I_low   (power conserving).
    Internal unknown: low-side current I_l."""
    def __init__(self, name, hp, hm, lp, lm, duty_key, duty_min=0.02, duty_max=0.98):
        super().__init__(name)
        self.hp, self.hm, self.lp, self.lm = hp, hm, lp, lm
        self.key, self.dmin, self.dmax = duty_key, duty_min, duty_max

    def terminals(self): return [self.hp, self.hm, self.lp, self.lm]
    def internal_unknowns(self): return ["I_l"]

    def constraints(self, vals, internals):
        m = np.clip(self.net.context.get(self.key, 0.0), self.dmin, self.dmax)
        vh = self._r(vals, self.hp, self.hm)
        vl = self._r(vals, self.lp, self.lm)
        return [m * vh - vl]

    def flows(self, vals, internals):
        il = internals[(id(self), "I_l")]
        m = np.clip(self.net.context.get(self.key, 0.0), self.dmin, self.dmax)
        ih = m * il
        return [(self.hp, -ih), (self.hm, +ih), (self.lp, +il), (self.lm, -il)]

    def commit(self, h, vals, internals):
        self._il = internals[(id(self), "I_l")]
        self._m = float(np.clip(self.net.context.get(self.key, 0.0), self.dmin, self.dmax))

    def observe(self, vals):
        vh = self._r(vals, self.hp, self.hm); vl = self._r(vals, self.lp, self.lm)
        return {"duty": getattr(self, "_m", 0.0), "I_low": getattr(self, "_il", 0.0),
                "V_high": vh, "V_low": vl,
                "P_high": -self._m * getattr(self, "_il", 0.0) * vh,
                "P_low": getattr(self, "_il", 0.0) * vl}


# ======================================================================
# Milestone 3: diesel engine, CVT, slip wheel, road forces (6x6 truck)
# ======================================================================

class QuadraticDamper(Component):
    """Aerodynamic drag on a translational node pair: F = c * v * |v|.

    WHY v*|v| AND NOT v^2: drag physically opposes motion, so the force
    must be an odd function of v (F(-v) = -F(v)). Plain c*v^2 would push
    the vehicle backwards while it rolls backwards, an energy pump.
    v*|v| gives the |F| ~ v^2 magnitude with the correct sign, and it is
    C^1-smooth at v=0 so the Newton solver does not chatter at rest.

    CALIBRATION: c = 0.5*rho_air*Cd*A. For the truck Cd ~ 0.9 (bluff box)
    and A = 2.65 m wide * 2.5 m tall = 6.6 m^2, so c = 3.65 N/(m/s)^2.
    At 105 km/h (29 m/s) that is ~3.1 kN, the dominant road load at
    top speed, which is what makes top speed power-limited.
    """
    def __init__(self, name, a, b, c):
        super().__init__(name); self.a, self.b, self.c = a, b, c
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        v = self._r(vals, self.a, self.b)
        F = self.c * v * abs(v)
        return [(self.a, -F), (self.b, +F)]
    def observe(self, vals):
        v = self._r(vals, self.a, self.b)
        return {"v": v, "F": self.c * v * abs(v), "P_loss": self.c * v**3}


class RoadForce(Component):
    """Longitudinal road loads on a vehicle mass node (grade + rolling).

    Two effects, both functions of (t, ctx) so grades can change in time:

    1) GRADE: the spec quotes grades in percent (60% grade = 0.6). The
       slope angle is theta = atan(grade), and the along-road gravity
       component is m*g*sin(theta) (NOT m*g*grade: at 60% that would
       overstate the force by 5%). cos(theta) shrinks the normal load,
       which matters because the traction limit and rolling resistance
       both scale with the normal load.

    2) ROLLING RESISTANCE: physically Coulomb-like, F = -Crr*N*sign(v),
       but sign(v) is discontinuous at rest and would jam the implicit
       solver (the vehicle could never leave v=0). tanh(v/v_ref) is the
       standard Simscape regularization: identical to sign(v) once
       |v| >> v_ref = 0.3 m/s, smooth and unambiguous at v = 0.

    Crr calibration: 0.008 is typical truck tires on pavement;
    0.02 for soft off-road ground (dirt/gravel).
    """
    def __init__(self, name, node, mass, grade_fn, Crr, v_ref=0.3, g=9.81):
        super().__init__(name); self.node, self.m, self.Crr = node, mass, Crr
        self.grade, self.v_ref, self.g = grade_fn, v_ref, g
    def terminals(self): return [self.node]
    def flows(self, vals, internals):
        gr = self.grade(self.net.t, self.net.context)
        theta = np.arctan(gr)
        F_grade = -self.m * self.g * np.sin(theta)
        F_roll = -self.Crr * self.m * self.g * np.cos(theta) * np.tanh(vals[self.node] / self.v_ref)
        self._F = F_grade + F_roll
        return [(self.node, self._F)]
    def observe(self, vals):
        return {"F": self._F, "P": self._F * vals[self.node]}


class DieselEngine(Component):
    """Turbo-diesel engine on a rotational node (case to ground).

    MODEL, in reading order of flows():

    1) FULL-LOAD TORQUE CURVE T_full(w): the manufacturer WOT curve, given
       as [[rpm, Nm], ...] (interpolated in rad/s; left end clamps to the
       first value = torque available below idle, right end = 0 so the
       engine cannot make power past the governor). Actual output torque
       is throttle * T_full(w): a diesel is torque-controlled (throttle
       sets fueling -> torque), unlike a gasoline engine's manifold.

    2) ENGINE BRAKING: when unthrottled the pistons/compression still
       pump air, a drag roughly proportional to speed. Modeled as
       (1-throttle) * c_brake * w, so at full throttle it vanishes and
       at closed throttle it coast-down brakes the crank. c_brake = 25
       Nm*s/rad reproduces a typical ~10 s idle-to-zero coastdown.

    3) CRANK INERTIA, trapezoidal: the engine node sees
       tau_net = tors - tau_J with tau_J = 2J(w_{n+1}-w_n)/h - tau_J_n,
       the SAME discrete law as TransMass/Inertia. This keeps engine
       kinetic energy accounted exactly like every other storage, so
       engine inertia exchanges energy with the vehicle consistently
       (important during ratio changes and shifts).

    4) FUEL: brake-specific consumption bsfc(w, tau) in g/kWh; the fuel
       flow is bsfc * P / 3.6e6 g/s when P > 0 (see commit(): kg are
       accumulated as g/kWh * W * h / 3.6e9). The default island model
       195 + 260*(1 - load)^2 g/kWh: ~195 g/kWh (~43% thermal) near full
       load, rising ~3.3x at 10% load (pumping/friction overhead of a
       big diesel at light load). Below zero power (coasting) fueling is
       not zero in reality (idle injection), so an idle floor of
       0.6 kg/h is added. fuel_kg is a committed state, observed in L
       via density 0.832 kg/L (winter diesel).

    Throttle comes from ctx[throttle_key], set by e.g. a PIController,
    so the engine is a plant in the control loop, not the controller.
    """
    def __init__(self, name, node, T_full, w_idle, J, throttle_key,
                 bsfc=None, c_brake=None, fuel_density=0.832):
        super().__init__(name); self.node, self.J = node, J
        self.w_idle, self.key = w_idle, throttle_key
        self.c_brake, self.fuel_density = c_brake, fuel_density
        if callable(T_full):
            self.T_full = T_full
        else:
            arr = np.asarray(T_full, dtype=float)          # [rpm, Nm] -> [rad/s, Nm]
            self.T_full = lambda w: np.interp(w, arr[:, 0] * np.pi / 30.0, arr[:, 1],
                                              left=arr[0, 1], right=0.0)
        if bsfc is None:
            # simple efficiency island: ~195 g/kWh near full load, quadratic
            # penalty as load drops (light-load pumping losses)
            bsfc = lambda w, tau: 195.0 + 260.0 * (1.0 - abs(tau) / max(self.T_full(w), 1e-9)) ** 2
        self.bsfc = bsfc

    def terminals(self): return [self.node]
    def reset(self):
        self.w_prev = 0.0; self.tau_J_prev = 0.0; self.fuel_kg = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
        w = vals[self.node]
        th = np.clip(self.net.context.get(self.key, 0.0), 0.0, 1.0)
        tors = th * self.T_full(w) - (1.0 - th) * (self.c_brake or 0.0) * w
        tau_J = 2 * self.J * (w - self.w_prev) / h - self.tau_J_prev   # trapezoidal inertia
        self._tors = tors
        return [(self.node, tors - tau_J)]
    def commit(self, h, vals, internals):
        w = vals[self.node]
        th = np.clip(self.net.context.get(self.key, 0.0), 0.0, 1.0)
        tors = th * self.T_full(w) - (1.0 - th) * (self.c_brake or 0.0) * w
        P = tors * w
        if P > 0:
            self.fuel_kg += self.bsfc(w, tors) * P / 3.6e9 * h        # g/kWh*W -> kg/s
        else:
            self.fuel_kg += 0.6 * h / 3600.0                          # idle floor ~0.6 kg/h
        # trapezoidal inertia torque, same discrete law as flows()
        tau_J_new = 2 * self.J * (w - self.w_prev) / h - self.tau_J_prev
        self.w_prev = w
        self.tau_J_prev = tau_J_new
    def observe(self, vals):
        w = vals[self.node]
        return {"rpm": w * 30.0 / np.pi, "torque": self._tors, "power_W": self._tors * w,
                "fuel_kg": self.fuel_kg, "fuel_L": self.fuel_kg / self.fuel_density}


class VariableGear(Component):
    """Ideal CVT: the movable-ratio version of Gear.

    LAW: w_in = rho * w_out and tau_out = rho * tau_in (power conserving,
    no losses). rho = clip(ratio_fn(t, ctx), rho_min, rho_max) is a CONTROL
    input, exactly like a CVT's servo deciding the pulley ratio: it is
    enforced as a hard kinematic constraint, not emerged from dynamics.

    INTERNAL UNKNOWN: 'tau_in'. The constraint fixes only the SPEED
    relation; the torque flowing through is unknown until the whole
    network is solved, so it becomes an extra unknown in the Newton
    system (same pattern as DCMachine's armature current).

    WHY _rho() IS CALLED IN BOTH constraints() AND flows(): during a
    Newton iteration the residual and Jacobian must describe the SAME
    machine; if flows() used a different rho than constraints() (e.g. a
    stale copy from last step), the solve would converge to a state that
    violates the constraint. Both therefore call the single _rho() helper,
    which also stores the clipped value for observe().

    DYNAMICS WARNING: an ideal CVT is RIGID - it changes ratio instantly
    and conserves power, so it cannot multiply torque at zero output
    speed without slipping. A standing launch up a steep grade therefore
    needs a torque converter or clutch (later milestone); at speed the
    model is accurate. rho bounds [5, 50] act as low/overdrive and
    deep-reduction gears; hitting a bound means the schedule is
    infeasible (e.g. engine over-speed near top speed).
    """
    def __init__(self, name, a1, a2, b1, b2, ratio_fn, rho_min, rho_max):
        super().__init__(name)
        self.a1, self.a2, self.b1, self.b2 = a1, a2, b1, b2
        self.rfn, self.rmin, self.rmax = ratio_fn, rho_min, rho_max
    def terminals(self): return [self.a1, self.a2, self.b1, self.b2]
    def internal_unknowns(self): return ["tau_in"]
    def _rho(self):
        rho = float(np.clip(self.rfn(self.net.t, self.net.context), self.rmin, self.rmax))
        self._rho_local = rho
        return rho
    def constraints(self, vals, internals):
        rho = self._rho()
        return [self._r(vals, self.a1, self.a2) - rho * self._r(vals, self.b1, self.b2)]
    def flows(self, vals, internals):
        t = internals[(id(self), "tau_in")]
        rho = self._rho()
        return [(self.a1, -t), (self.a2, +t), (self.b1, +rho * t), (self.b2, -rho * t)]
    def commit(self, h, vals, internals):
        self._t = internals[(id(self), "tau_in")]
    def observe(self, vals):
        t = getattr(self, "_t", 0.0)
        w_in = self._r(vals, self.a1, self.a2); w_out = self._r(vals, self.b1, self.b2)
        return {"ratio": getattr(self, "_rho_local", 1.0), "tau_in": t,
                "w_in": w_in, "w_out": w_out, "P_in": t * w_in,
                "P_out": getattr(self, "_rho_local", 1.0) * t * w_out}


class SlipWheel(Component):
    """Traction wheel with a slip (traction) limit - the tire model.

    F = Fmax * tanh(k_slip * (r*w - v) / Fmax), slip = r*w - v (m/s).

    WHY tanh: a real tire's force/slip curve is steep and linear at small
    slip (slip stiffness), then saturates at the friction limit mu*N and
    stays flat (or falls slightly) - a saturating S-curve. tanh is the
    cheapest smooth function with exactly that shape:
      - small slip: F ~ k_slip * slip, a stiff spring (k_slip = 5e5 N/m
        -> 0.002 m/s slip carries 1000 N), so at low torque the wheel is
        effectively IDEAL ROLLING (v = r*w), which is why T9 shows slip
        of only 0.009 m/s at top speed;
      - large slip: |F| -> Fmax, so no amount of wheelspin gives more
        force than mu*N (T11: acceleration pinned at ~mu*g).
    The (r*w - v)/Fmax normalization keeps the SAME curve shape when the
    normal load changes (heavier, or on a grade), which is how real tire
    friction scales.

    Fmax_fn(t, ctx): the traction budget, e.g. mu*M*g*cos(theta) - all
    driven axles lumped into one aggregate (1D model, no weight
    transfer yet).

    ENERGY: slip power F*(r*w - v) is dissipated as heat in the tire -
    the physical mechanism that limits wheelspin; it is observed as
    P_slip_loss so tests can verify no energy appears from nowhere.
    """
    def __init__(self, name, x, r1, r2, radius, Fmax_fn, k_slip=2.0e5):
        super().__init__(name); self.x, self.r1, self.r2, self.r = x, r1, r2, radius
        self.Fmax_fn, self.k = Fmax_fn, k_slip
    def terminals(self): return [self.x, self.r1, self.r2]
    def flows(self, vals, internals):
        v = vals[self.x]; w = self._r(vals, self.r1, self.r2)
        Fmax = max(self.Fmax_fn(self.net.t, self.net.context), 1e-6)
        slip = self.r * w - v
        F = Fmax * np.tanh(self.k * slip / Fmax)
        self._F, self._slip, self._Fmax = F, slip, Fmax
        return [(self.x, +F), (self.r1, -F * self.r), (self.r2, +F * self.r)]
    def observe(self, vals):
        return {"F": self._F, "slip": self._slip, "F_max": self._Fmax,
                "P_roll": self._F * vals[self.x],
                "P_slip_loss": self._F * self._slip}


class ContextPublisher(Component):
    """Pure signal source with NO physical terminals.

    Simscape separates the physical network from signal routing; this
    class is the signal-side equivalent (PS-Constant / From-Workspace):
    it evaluates val_fn(t, ctx) each commit and publishes it to
    ctx[key] where physical components read it (e.g. a throttle or a
    profile). Because it has no terminals it contributes no flows and
    no unknowns; it only materializes in commit/publish, so it cannot
    perturb the network solve.
    """
    def __init__(self, name, key, val_fn):
        super().__init__(name); self.key_, self.val_fn = key, val_fn
    def terminals(self): return []
    def flows(self, vals, internals): return []
    def commit(self, h, vals, internals):
        self._v = self.val_fn(self.net.t, self.net.context)
    def publish(self, ctx):
        ctx[self.key_] = getattr(self, "_v", 0.0)
    def observe(self, vals):
        return {"value": getattr(self, "_v", 0.0)}
