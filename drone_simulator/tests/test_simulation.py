"""Тесты физики, автопилота и игровой логики (без pygame)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_sim import config as cfg  # noqa: E402
from drone_sim.autopilot import PID, FlightMode, PilotInput  # noqa: E402
from drone_sim.physics import Drone  # noqa: E402
from drone_sim.world import Building, GameState, PadKind, Simulation  # noqa: E402

DT = 1 / 120


def fly(drone: Drone, seconds: float, throttle: float, tilt: float = 0.0, wind=(0.0, 0.0)):
    drone.landed = False
    for _ in range(int(seconds / DT)):
        drone.step(DT, throttle, tilt, wind)


# ----------------------------------------------------------------------------- физика
def test_hover_throttle_keeps_altitude():
    d = Drone(0, 50)
    d.throttle = d.hover_throttle()   # моторы уже раскручены
    fly(d, 3, d.hover_throttle())
    assert abs(d.vy) < 0.2
    assert abs(d.y - 50) < 1.0


def test_full_throttle_climbs_and_zero_throttle_falls():
    up = Drone(0, 50)
    fly(up, 2, 1.0)
    assert up.vy > 3 and up.y > 52

    down = Drone(0, 50)
    fly(down, 2, 0.0)
    assert down.vy < -5 and down.y < 45


def test_tilt_right_moves_right():
    d = Drone(0, 50)
    fly(d, 3, 0.6, tilt=1.0)
    assert d.angle == pytest.approx(cfg.DRONE_MAX_TILT)
    assert d.vx > 3 and d.x > 3


def test_wind_pushes_drone():
    d = Drone(0, 50)
    for _ in range(int(5 / DT)):
        d.landed = False
        d.step(DT, d.hover_throttle(), 0.0, wind=(-6.0, 0.0))
    assert d.vx < -2


def test_payload_drains_battery_faster():
    light, heavy = Drone(0, 50), Drone(0, 50, payload=1.2)
    fly(light, 20, light.hover_throttle())
    fly(heavy, 20, heavy.hover_throttle())
    assert heavy.battery < light.battery < 100


def test_empty_battery_cuts_motors():
    d = Drone(0, 50, battery=0.0)
    fly(d, 1, 1.0)
    assert d.throttle == 0 and d.vy < 0


def test_landed_drone_stays_on_ground_with_low_throttle():
    d = Drone(0, 0.2)
    d.landed = True
    for _ in range(120):
        d.step(DT, 0.1, 0.0)
    assert d.landed and d.vy == 0 and d.y == pytest.approx(0.2)


# ----------------------------------------------------------------------------- PID / режимы
def test_pid_output_is_clamped():
    pid = PID(10, 0, 0, out_limit=0.5)
    assert pid.update(100, 0.1) == 0.5
    assert pid.update(-100, 0.1) == -0.5


def test_assist_mode_holds_altitude():
    sim = Simulation(seed=1)
    d = sim.drone
    d.x, d.y, d.landed = 30.0, 60.0, False
    sim.controller.mode = FlightMode.ASSIST
    for _ in range(int(6 / DT)):
        thr, tilt = sim.controller.command(d, PilotInput(), DT)
        d.step(DT, thr, tilt)
    assert abs(d.vy) < 0.3


def test_manual_mode_changes_throttle_gradually():
    sim = Simulation(seed=1)
    sim.toggle_manual()
    assert sim.controller.mode is FlightMode.MANUAL
    thr, _ = sim.controller.command(sim.drone, PilotInput(vertical=1), 1.0)
    assert 0.6 < thr < 0.8


# ----------------------------------------------------------------------------- мир
def make_flat_sim() -> Simulation:
    """Симуляция с одним зданием, чтобы проверять столкновения детерминированно."""
    sim = Simulation(seed=3)
    sim.buildings = [Building(100, 20, 30)]
    sim.birds = []
    sim.no_fly = []
    sim._update_birds = lambda dt: None
    return sim


def test_soft_landing_no_damage():
    sim = make_flat_sim()
    d = sim.drone
    d.x, d.y, d.vy, d.landed = 200.0, 0.2, -1.5, False
    sim._collide(prev_bottom=0.1)
    assert d.landed and d.hull == 100


def test_hard_landing_damages():
    sim = make_flat_sim()
    d = sim.drone
    d.x, d.y, d.vy, d.landed = 200.0, 0.1, -6.0, False
    sim._collide(prev_bottom=0.4)
    assert d.landed and 0 < d.hull < 100


def test_fatal_impact_destroys_drone():
    sim = make_flat_sim()
    d = sim.drone
    d.x, d.y, d.vy, d.landed = 200.0, 0.1, -15.0, False
    sim._collide(prev_bottom=0.4)
    assert not d.alive


def test_landing_on_roof():
    sim = make_flat_sim()
    d = sim.drone
    d.x, d.y, d.vy, d.landed = 110.0, 30.1, -1.0, False
    sim._collide(prev_bottom=30.2)
    assert d.landed and d.bottom == pytest.approx(30.0)


def test_side_hit_pushes_out_and_damages():
    sim = make_flat_sim()
    d = sim.drone
    d.x, d.y, d.vx, d.landed = 100.3, 15.0, 8.0, False
    sim._collide(prev_bottom=d.bottom)
    assert d.x < 100 and d.vx < 0 and d.hull < 100


def test_drone_falls_off_roof_edge():
    sim = make_flat_sim()
    d = sim.drone
    d.x, d.y, d.landed = 125.0, 30.0 + d.spec.height / 2, True
    sim._collide(prev_bottom=d.bottom)
    assert not d.landed


def test_safe_speed_blocks_flying_into_wall():
    sim = make_flat_sim()
    assert sim.safe_speed(97.0, 300.0, bottom=5.0) < 0.5
    assert sim.safe_speed(97.0, 300.0, bottom=40.0) == 15.0
    assert sim.safe_speed(20.0, 300.0, bottom=5.0) > 1.0


def test_city_is_generated_with_required_pads():
    sim = Simulation(seed=123)
    kinds = {p.kind for p in sim.pads}
    assert {PadKind.BASE, PadKind.DEPOT, PadKind.CLIENT} <= kinds
    assert sim.pads[0].kind is PadKind.BASE
    assert all(0 <= b.x and b.right <= cfg.WORLD_WIDTH for b in sim.buildings)


def test_same_seed_same_city():
    a, b = Simulation(seed=5), Simulation(seed=5)
    assert [(x.x, x.height) for x in a.buildings] == [(x.x, x.height) for x in b.buildings]


# ----------------------------------------------------------------------------- миссии
def land_on(sim: Simulation, pad) -> None:
    d = sim.drone
    d.x, d.y = pad.x, pad.y + d.spec.height / 2
    d.vx = d.vy = 0.0
    d.landed = True


def test_pickup_and_delivery():
    sim = Simulation(seed=8)
    m = sim.mission
    land_on(sim, m.pickup)
    for _ in range(int(2 / DT)):
        sim.step(DT, PilotInput())
    assert m.picked_up and sim.drone.payload == m.cargo_mass

    land_on(sim, m.dropoff)
    for _ in range(int(2 / DT)):
        sim.step(DT, PilotInput())
    assert sim.deliveries == 1
    assert sim.score > 0
    assert sim.drone.payload == 0
    assert sim.mission is not m


def test_base_charges_battery_and_pauses_timer():
    sim = Simulation(seed=2)
    d = sim.drone
    d.battery, d.hull = 40.0, 50.0
    elapsed = sim.mission.elapsed
    for _ in range(int(2 / DT)):
        sim.step(DT, PilotInput())
    assert d.battery > 50 and d.hull > 50
    assert sim.mission.elapsed == elapsed


def test_mission_timeout_penalty():
    sim = Simulation(seed=4)
    d = sim.drone
    d.x, d.y, d.landed = 300.0, 150.0, False   # висим в воздухе, не на базе
    sim.mission.elapsed = sim.mission.time_limit - 0.001
    old = sim.mission
    sim.step(DT, PilotInput(vertical=0))
    assert sim.mission is not old
    assert sim.score <= -cfg.MISSION_FAIL_PENALTY


def test_no_fly_zone_costs_points():
    sim = Simulation(seed=6)
    z = sim.no_fly[0]
    d = sim.drone
    d.x, d.y, d.landed = (z.x1 + z.x2) / 2, z.y1 + 10, False
    for _ in range(int(1 / DT)):
        sim.step(DT, PilotInput())
    assert sim.score < -20


def test_autopilot_penalty_halves_reward():
    sim = Simulation(seed=8)
    sim.toggle_autopilot()
    assert sim.mission.used_autopilot
    sim.toggle_autopilot()
    assert sim.controller.mode is FlightMode.ASSIST


@pytest.mark.parametrize("seed", [0, 2, 6])
def test_autopilot_delivers_without_crashing(seed):
    sim = Simulation(seed=seed)
    sim.toggle_autopilot()
    t = 0.0
    while t < 400 and sim.deliveries == 0:
        if sim.controller.mode is not FlightMode.AUTO:
            sim.toggle_autopilot()
        sim.step(1 / 60, PilotInput())
        t += 1 / 60
    assert sim.state is GameState.RUNNING
    assert sim.deliveries >= 1


def test_wind_grows_with_altitude():
    sim = Simulation(seed=9)
    sim.wind.base = 3.0
    sim.wind.waves = []
    low, _ = sim.wind.at(0, 5)
    high, _ = sim.wind.at(0, 100)
    assert abs(high) > abs(low)
    assert math.isfinite(high)
