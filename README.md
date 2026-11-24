# physim — a mini Simscape-style 1D physical network simulator

`physim` mimics MATLAB Simulink/Simscape for 1D physical network simulation across
**electrical**, **rotational**, and **thermal** domains, with an implicit DAE solver
and a notebook-friendly API. Pure Python: `numpy`, `scipy`, `matplotlib` only.

```python
from physim import Network, VoltageSource, Resistor, Capacitor
net = Network()
net.add_node("g", "electrical", ground=True)      # reference node, fixed at 0
net.add_node("a", "electrical")
net.add(VoltageSource("src", "a", "g", 5.0))       # 5 V between a and g
net.add(Resistor("R", "a", "g", 100.0))
net.add(Capacitor("C", "a", "g", 1e-6))
ts, recs = net.run(1e-3, 1e-6)                     # fixed step h = 1 µs, t_end = 1 ms
# recs[i] = (t, {node: across_value}, {component_name: observe_dict})
```

That's the whole "build your own system in 6 lines" example — run it and inspect
`recs[-1][1]["a"]` for the capacitor voltage at the end.

## Concepts: across / through variables

Every domain defines an **across** variable measured *at nodes* and a **through**
variable *flowing through components* between nodes:

| Domain       | Across (at node) | Through (through component) | SI unit |
|--------------|------------------|-----------------------------|---------|
| `electrical` | voltage `V`      | current `I`                 | V, A    |
| `rotational` | angular speed `ω`| torque `τ`                  | rad/s, N·m |
| `thermal`    | temperature `T`  | heat flow `q̇`               | K, W    |

* **Grounded nodes** (`ground=True`) are fixed at the domain reference (0).
* **Conservation law**: at every free node, the sum of through-variables flowing
  into the node from all connected components is zero.
* The **unknown vector** solved each step = free-node across values (fixed order)
  + all component *internal unknowns* (e.g. the armature current of an ideal machine).
* The **residual** = node flow balances + component *constraint* equations.

## Sign conventions

* **Positive through-variable flows from terminal `a` to terminal `b` through the
  component** — e.g. a resistor with `V_a > V_b` carries positive current out of `a`.
* **`flows(vals, internals)` returns `(node, flow)` pairs giving the flow *into* each
  node from the component**; the solver sums these to zero at every free node.
* **A positive `TorqueSource` drives the shaft; use a negative value for an opposing load.**
* `VoltageSource.observe()["P"]` is `V·I` with `I` positive when current enters the `+`
  terminal — so `P > 0` means the source *absorbs* power (charging), `P < 0` means it delivers.

## Time stepping / solver

Fixed step `h`, **trapezoidal integration**, implicit. Each step solves the residual
system with `scipy.optimize.root` (Newton, method `hybr`), **warm-started from the
previous step's solution**, then commits component internal states. This is an
index-1 semi-implicit DAE scheme: every component's constitutive law is written so the
state update is trapezoidal, e.g.

* Capacitor: `i_{n+1} = 2C(v_{n+1} − v_n)/h − i_n`
* Inductor:  `i_{n+1} = i_n + h(v_n + v_{n+1})/(2L)`
* Inertia:   `τ_{n+1} = 2J(ω_{n+1} − ω_n)/h − τ_n`
* Thermal mass: `q_{n+1} = 2C(T_{n+1} − T_n)/h − q_n`

Because the step is fixed and the system is index-1, each Newton solve is cheap and
warm-starting makes the whole run fast. (Adaptive stepping is a possible extension.)

## Cross-domain coupling via the shared `context`

After each step, every component may `publish(ctx)` values into `net.context`.
Other components read them on the **next** step (one-step delay), which is how
cross-domain energy coupling is modeled — e.g. a `Resistor` publishes `P_loss::<name>`
and a `HeatSource` in the thermal domain injects it:

```python
net.add(Resistor("Ra", "a", "b", 0.2))                                   # publishes P_loss::Ra
net.add(HeatSource("loss", "T1", lambda t, ctx: ctx.get("P_loss::Ra", 0.0)))
```

## Components

| Component | Terminals | Behavior |
|---|---|---|
| `VoltageSource(name, plus, minus, V)` | 2 (electrical) | ideal source; `V` may be a constant or `f(t)`; internal unknown: current |
| `Resistor(name, a, b, R)` | 2 (electrical) | `I = (V_a−V_b)/R`; publishes `P_loss::name` |
| `Capacitor(name, a, b, C)` | 2 (electrical) | trapezoidal state: `vc`, `ic` |
| `Inductor(name, a, b, L)` | 2 (electrical) | trapezoidal state: `iL`, `v_prev` |
| `DCMachine(name, p, m, r1, r2, k)` | 2 electrical + 2 rotational | ideal machine `V = kω`, `τ = kI`; internal unknown: armature current |
| `TorqueSource(name, node, tau)` | 1 (rotational) | `tau` may be `f(t)`; negative = load |
| `Inertia(name, node, J)` | 1 (rotational, to ground) | flywheel; trapezoidal state |
| `Damper(name, a, b, B)` | 2 (rotational) | viscous `τ = B·Δω` |
| `Spring(name, a, b, k)` | 2 (rotational) | `τ = k·θ`, `θ̇ = Δω` |
| `ThermalMass(name, node, C)` | 1 (thermal, to ground) | `q̇ = C·dT/dt`, trapezoidal |
| `ThermalResistor(name, a, b, Rth)` | 2 (thermal) | `q̇ = ΔT/Rth` |
| `HeatSource(name, node, Q)` | 1 (thermal) | `Q` may be `f(t, context)` |

## API

```python
net = Network()
net.add_node(name, domain, ground=False)   # domain in {"electrical","rotational","thermal"}
comp = net.add(Component(...))             # returns the component
ts, recs = net.run(t_end, h)               # recs[i] = (t, {node: across}, {comp: observe})
```

## Adding your own component

Subclass `Component` and implement:

* `terminals()` — node names it connects to;
* `flows(vals, internals)` — return `[(node, flow_into_node), ...]` (the through variable);
* optionally `internal_unknowns()` (names solved together with node values),
  `constraints(vals, internals)` (extra residual equations),
  `commit(h, vals, internals)` (advance internal states after the solve),
  `publish(ctx)` / `observe(vals)` / `reset()`.

Adding a new **domain** = one entry in `DOMAINS`, e.g.
`"translational": {"across": "v", "through": "F", "ref": 0.0}` — after which components
in that domain plug into the same conservation law unchanged.

## Examples

Open and run **`physim_demo.ipynb`** (executes top-to-bottom, with plots):

1. **RLC step response** — 12 V into R=0.5 Ω, L=10 mH, C=1 mF; peak error vs the exact
   underdamped response = 0.034 V (acceptance < 0.05 V).
2. **DC motor drive** — 400 V battery (Rb=0.05 Ω), k=0.8 machine, J=1.5, b=0.02, −15 N·m
   load from t=1 s; at t=5 s: I = 31.20 A, ω = 498 rad/s (analytic, within 1%).
3. **Regenerative braking** — bus voltage 400→250→0 V; during braking the current is
   negative and the bus recovers ~83 kJ of the ~186 kJ flywheel KE lost (≈ 45%).
4. **Thermal step + cross-domain coupling** — 10 W into C=800 J/K with Rth=0.8 K/W matches
   `T = PRth(1−e^{−t/RC})` to 0.003 K; armature copper loss feeds a thermal mass through
   the shared context (one-step delay), steady loss ≈ 0.128 W.

## Tests

```bash
python -m pytest tests/ -v
```

Five acceptance tests (RLC step response, motor steady state, regenerative braking,
thermal step, cross-domain coupling) with 1% tolerance (RLC: peak error < 0.05 V).

## Requirements & layout

Python ≥ 3.9; only `numpy`, `scipy`, `matplotlib` (plus `pytest` for tests).

```
physim/__init__.py     # the whole engine (Network, Component, all components)
tests/test_physim.py   # acceptance tests
physim_demo.ipynb      # runnable demos with plots
README.md
```

## Possible extensions (not implemented, by design)

Translational domain (`v`, `F`), hydraulic domain (pressure, flow), battery OCV+R model,
gearbox, adaptive step size.
