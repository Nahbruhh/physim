"""Generate truck_demo.png: 4-panel validation figure for the 6x6 diesel truck.

Panels: (a) top-speed run with engine rpm, (b) 60% grade climb with wheel
force, (c) low-mu launch with slip and F/F_max, (d) 80 km/h cruise fuel/range.
Run:  python truck_demo.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "tests")
from test_truck import build_truck, M_TRUCK, G, TANK_L

H = 5e-3

# (a) top speed: unreachable target -> full throttle, level road
net = build_truck(lambda t, ctx: 0.0, 0.9, 0.008, lambda t, ctx: 40.0)
ts, recs = net.run(150.0, H)
t = np.array(ts)
v = np.array([r[1]["veh"] for r in recs])
rpm = np.array([r[2]["eng"]["rpm"] for r in recs])
v_top = v[-1] * 3.6

# (b) 60% grade met at t=8 s with speed (rigid-CVT standing launch is the
# documented limitation -> torque converter in the 6-speed milestone)
def grade60(t, ctx):
    return 0.0 if t < 8.0 else 0.6
net = build_truck(grade60, 0.8, 0.02, lambda t, ctx: 12.0, kp=0.15, ki=0.05)
ts2, recs2 = net.run(120.0, H)
t2 = np.array(ts2)
v2 = np.array([r[1]["veh"] for r in recs2])
F2 = np.array([r[2]["wheels"]["F"] for r in recs2])
F_req2 = M_TRUCK * G * np.sin(np.arctan(0.6))
v_ss = v2[-1]

# (c) low-mu launch (wet clay, mu=0.45)
net = build_truck(lambda t, ctx: 0.0, 0.45, 0.02, lambda t, ctx: 30.0,
                  kp=0.08, ki=0.03)
ts3, recs3 = net.run(40.0, H)
t3 = np.array(ts3)
v3 = np.array([r[1]["veh"] for r in recs3])
slip3 = np.array([r[2]["wheels"]["slip"] for r in recs3])
F3 = np.array([r[2]["wheels"]["F"] for r in recs3])
Fmax3 = np.array([r[2]["wheels"]["F_max"] for r in recs3])

# (d) cruise 80 km/h -> fuel rate -> range on the 320 L tank
net = build_truck(lambda t, ctx: 0.0, 0.9, 0.008, lambda t, ctx: 80.0 / 3.6)
ts4, recs4 = net.run(200.0, H)
t4 = np.array(ts4)
v4 = np.array([r[1]["veh"] for r in recs4])
fuel4 = np.array([r[2]["eng"]["fuel_L"] for r in recs4])
v_cruise = np.mean(v4[-4000:])
kmpl = (v_cruise * (200.0 - t4[-4000]) / 1000.0) / (fuel4[-1] - fuel4[-4000])
rng = kmpl * TANK_L

fig, ax = plt.subplots(2, 2, figsize=(13, 8))

a = ax[0, 0]
a.plot(t, v * 3.6, label="speed")
a.set(xlabel="t [s]", ylabel="v [km/h]", title=f"(a) top speed: {v_top:.1f} km/h (spec 105-110)")
a2 = a.twinx(); a2.plot(t, rpm, "r--", lw=1, label="engine rpm"); a2.set_ylabel("rpm [1/min]", color="r")
a.grid(alpha=0.3); a.legend(loc="lower right")

a = ax[0, 1]
a.plot(t2, v2 * 3.6, label="speed")
a.axvline(8.0, color="k", ls=":", lw=1)
a.set(xlabel="t [s]", ylabel="v [km/h]",
      title=f"(b) 60% grade from t=8 s: crawl {v_ss*3.6:.1f} km/h")
a2 = a.twinx(); a2.plot(t2, F2 / 1000, "r--", lw=1, label="wheel force")
a2.axhline(F_req2 / 1000, color="g", ls=":", lw=1, label=f"gravity req {F_req2/1000:.0f} kN")
a2.set_ylabel("wheel F [kN]", color="r"); a.grid(alpha=0.3)
a.legend(loc="upper right"); a2.legend(loc="lower right")

a = ax[1, 0]
a.plot(t3, v3 * 3.6, label="speed")
a.set(xlabel="t [s]", ylabel="v [km/h]", title="(c) low-mu launch (mu=0.45): traction-limited")
a2 = a.twinx()
a2.plot(t3, slip3, "r--", lw=1, label="slip [m/s]")
a2.plot(t3, F3 / Fmax3, "g-.", lw=1, label="F/F_max")
a2.set_ylabel("slip [m/s], F/F_max [-]"); a2.set_ylim(-0.1, 1.2)
a.grid(alpha=0.3); a.legend(loc="lower right"); a2.legend(loc="upper right")

a = ax[1, 1]
a.plot(t4, v4 * 3.6, label="speed")
a.set(xlabel="t [s]", ylabel="v [km/h]", title=f"(d) 80 km/h cruise: range {rng:.0f} km on {TANK_L:.0f} L ({100/kmpl:.1f} L/100km)")
a2 = a.twinx(); a2.plot(t4, fuel4, "r--", lw=1, label="fuel used [L]")
a2.set_ylabel("fuel [L]", color="r"); a.grid(alpha=0.3)
a.legend(loc="lower right"); a2.legend(loc="upper left")

fig.suptitle("physim Milestone 3 - 6x6 diesel utility truck, 20 t GVW, CVT stage (h = 5 ms)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("truck_demo.png", dpi=130)
print(f"saved truck_demo.png  (top {v_top:.1f} km/h, grade crawl {v_ss*3.6:.1f} km/h, range {rng:.0f} km)")
