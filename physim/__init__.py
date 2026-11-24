"""
physim: a small Simscape-like 1D physical network simulator.

Every domain has an ACROSS variable (voltage, angular speed, temperature...)
defined at NODES, and a THROUGH variable (current, torque, heat flow)
flowing through COMPONENTS between nodes. Conservation at every node:
sum of through variables = 0. Each time step the implicit system
(trapezoidal integration) is solved with scipy.optimize.root, then
component states are committed. Components may publish values (e.g.
copper loss) into a shared context dict that other domains can read.

Sign conventions
----------------
* Positive through-variable flows from terminal ``a`` to terminal ``b``
  through the component (e.g. current enters at ``a`` when ``V_a > V_b``).
* A positive TorqueSource drives the shaft; use a negative value for a load.
* Grounded nodes are fixed at the domain reference (0).

Solver
------
Fixed step ``h``, trapezoidal (index-1 semi-implicit DAE): each step solves
the residual system (node flow balances + component constraints) with
``scipy.optimize.root`` (Newton, warm-started from the previous step's
solution), then commits component internal states.
"""

import numpy as np
from scipy.optimize import root

__all__ = [
    "DOMAINS",
    "Network",
    "Component",
    "VoltageSource",
    "Resistor",
    "Capacitor",
    "Inductor",
    "Damper",
    "Spring",
    "Inertia",
    "TorqueSource",
    "DCMachine",
    "ThermalMass",
    "ThermalResistor",
    "HeatSource",
]

__version__ = "0.1.0"

DOMAINS = {
    "electrical":  {"across": "V",  "through": "I",    "ref": 0.0},
    "rotational":  {"across": "w",  "through": "tau",  "ref": 0.0},
    "thermal":     {"across": "T",  "through": "qd",   "ref": 0.0},
}


class Network:
    def __init__(self):
        self.nodes = {}          # name -> domain
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
        for n in self.free_nodes():
            s = 0.0
            for c in self.components:
                for node, flow in c.flows(vals, internals):
                    if node == n:
                        s += flow
            res.append(s)
        for c in self.components:
            res.extend(c.constraints(vals, internals))
        return np.array(res)

    def step(self, h, t):
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
            vals = self.step(h, t)
            ts.append(t)
            recs.append((t, dict(vals), {c.name: c.observe(vals) for c in self.components}))
        return ts, recs


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
    def _v(self):
        return self.V(self.net.t) if callable(self.V) else self.V
    def constraints(self, vals, internals):
        return [self._r(vals, self.p, self.m) - self._v()]
    def flows(self, vals, internals):
        i = internals[(id(self), "I")]
        self._i = i
        return [(self.p, -i), (self.m, +i)]
    def observe(self, vals):
        v = self._v(); i = getattr(self, "_i", 0.0)
        # I is positive when current enters the + terminal, so P = V*I > 0
        # means the source ABSORBS power (being charged); P < 0 means it delivers.
        return {"V": v, "I": i, "P": v * i}


class Resistor(Component):
    def __init__(self, name, a, b, R):
        super().__init__(name); self.a, self.b, self.R = a, b, R
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        i = self._r(vals, self.a, self.b) / self.R
        self._p_loss = i * i * self.R
        return [(self.a, -i), (self.b, +i)]
    def observe(self, vals):
        vab = self._r(vals, self.a, self.b); i = vab / self.R
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
        i = self.iL + h * (self.v_prev + self._r(vals, self.a, self.b)) / (2 * self.L)
        self._i_new = i
        return [(self.a, -i), (self.b, +i)]
    def commit(self, h, vals, internals):
        self.iL = getattr(self, "_i_new", self.iL)
        self.v_prev = self._r(vals, self.a, self.b)
    def observe(self, vals): return {"I": self.iL, "E": 0.5 * self.L * self.iL ** 2}


class Damper(Component):
    """Rotational viscous damper."""
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
    def __init__(self, name, a, b, k):
        super().__init__(name); self.a, self.b, self.k = a, b, k
    def terminals(self): return [self.a, self.b]
    def reset(self): self.theta = 0.0
    def flows(self, vals, internals):
        tau = self.k * self.theta
        return [(self.a, -tau), (self.b, +tau)]
    def commit(self, h, vals, internals):
        self.theta += h * self._r(vals, self.a, self.b)
    def observe(self, vals): return {"theta": self.theta, "tau": self.k * self.theta}


class Inertia(Component):
    """Flywheel from a node to mechanical ground."""
    def __init__(self, name, node, J):
        super().__init__(name); self.node, self.J = node, J
    def terminals(self): return [self.node]
    def reset(self): self.w_prev = 0.0; self.tau_prev = 0.0
    def flows(self, vals, internals):
        h = self.net.last_h
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
    """Positive tau drives the node (use negative for an opposing load)."""
    def __init__(self, name, node, tau):
        super().__init__(name); self.node, self.tau = node, tau
    def terminals(self): return [self.node]
    def _t(self):
        return self.tau(self.net.t) if callable(self.tau) else self.tau
    def flows(self, vals, internals):
        return [(self.node, +self._t())]
    def observe(self, vals):
        t = self._t()
        return {"tau": t, "P": t * vals[self.node]}


class DCMachine(Component):
    """Ideal DC machine: electrical (p,m) <-> rotational (r1,r2).
    V_em = k*w, tau = k*I. Internal unknown: armature current I."""
    def __init__(self, name, p, m, r1, r2, k):
        super().__init__(name); self.p, self.m, self.r1, self.r2, self.k = p, m, r1, r2, k
        self._i = 0.0
    def terminals(self): return [self.p, self.m, self.r1, self.r2]
    def internal_unknowns(self): return ["I"]
    def constraints(self, vals, internals):
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
    def observe(self, vals): return {"T": vals[self.node]}


class ThermalResistor(Component):
    def __init__(self, name, a, b, Rth):
        super().__init__(name); self.a, self.b, self.Rth = a, b, Rth
    def terminals(self): return [self.a, self.b]
    def flows(self, vals, internals):
        q = self._r(vals, self.a, self.b) / self.Rth
        return [(self.a, -q), (self.b, +q)]
    def observe(self, vals):
        q = self._r(vals, self.a, self.b) / self.Rth
        return {"Q": q, "dT": self._r(vals, self.a, self.b)}


class HeatSource(Component):
    """Injects heat into a node. Q may be a function of (t, context)."""
    def __init__(self, name, node, Q):
        super().__init__(name); self.node, self.Q = node, Q
    def terminals(self): return [self.node]
    def _q(self):
        return self.Q(self.net.t, self.net.context) if callable(self.Q) else self.Q
    def flows(self, vals, internals):
        q = self._q()
        self._q_last = q
        return [(self.node, +q)]
    def observe(self, vals): return {"Q": getattr(self, "_q_last", 0.0)}
