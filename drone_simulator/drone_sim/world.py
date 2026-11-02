"""Игровой мир и симуляция: город, ветер, птицы, бесполётные зоны и миссии доставки.

Модуль не зависит от pygame, поэтому всю логику можно тестировать без окна.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from . import config as cfg
from .autopilot import FlightController, FlightMode, PilotInput
from .physics import Drone, clamp


# =============================================================================
# Объекты мира
# =============================================================================
@dataclass
class Building:
    x: float      # левый край
    width: float
    height: float
    seed: int = 0  # для отрисовки окон

    @property
    def right(self) -> float:
        return self.x + self.width


class PadKind(Enum):
    BASE = "База"
    DEPOT = "Склад"
    CLIENT = "Клиент"


@dataclass
class Pad:
    name: str
    kind: PadKind
    x: float      # центр
    y: float      # высота поверхности площадки
    width: float = 6.0

    def contains(self, x: float) -> bool:
        return abs(x - self.x) <= self.width / 2


@dataclass
class Bird:
    x: float
    y: float
    vx: float
    phase: float
    amplitude: float = 3.0
    base_y: float = 0.0
    alive: bool = True

    def step(self, dt: float) -> None:
        self.phase += dt * 2.2
        self.x += self.vx * dt
        self.y = self.base_y + math.sin(self.phase) * self.amplitude


@dataclass
class NoFlyZone:
    x1: float
    x2: float
    y1: float
    y2: float
    label: str = "АЭРОПОРТ"

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class Wind:
    """Ветер = постоянная составляющая + порывы (сумма синусоид) + усиление с высотой."""

    def __init__(self, rng: random.Random, strength: float) -> None:
        self.strength = strength
        self.base = rng.uniform(-1, 1) * strength
        self.waves = [(rng.uniform(0.05, 0.4), rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 1.0))
                      for _ in range(4)]

    def at(self, t: float, y: float) -> tuple[float, float]:
        gust = sum(a * math.sin(f * t + p) for f, p, a in self.waves) / 2.0
        altitude_factor = 0.4 + min(y, cfg.CEILING) / 70.0
        wx = (self.base + gust * self.strength) * altitude_factor
        wy = 0.25 * gust * self.strength * math.sin(0.7 * t)
        return wx, wy


# =============================================================================
# Миссии
# =============================================================================
@dataclass
class Mission:
    pickup: Pad
    dropoff: Pad
    cargo_name: str
    cargo_mass: float
    time_limit: float
    reward: int
    fragile: bool = False
    picked_up: bool = False
    elapsed: float = 0.0
    used_autopilot: bool = False
    cargo_health: float = 100.0

    @property
    def target(self) -> Pad:
        return self.dropoff if self.picked_up else self.pickup

    @property
    def time_left(self) -> float:
        return self.time_limit - self.elapsed


CARGO = [
    ("Лекарства", 0.5, True),
    ("Пицца", 0.8, False),
    ("Запчасти", 1.3, False),
    ("Документы", 0.3, False),
    ("Торт", 1.0, True),
    ("Финики", 0.9, False),
    ("Ноутбук", 1.1, True),
    ("Цветы", 0.4, True),
]


class GameState(Enum):
    RUNNING = 1
    CRASHED = 2
    OUT_OF_POWER = 3


@dataclass
class Event:
    text: str
    color: tuple[int, int, int] = (255, 255, 255)
    age: float = 0.0


# =============================================================================
# Симуляция
# =============================================================================
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]
    size: float = 2.0


class Simulation:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed if seed is not None else random.randrange(1_000_000)
        self.rng = random.Random(self.seed)
        self.time = 0.0
        self.level = 1
        self.score = 0
        self.deliveries = 0
        self.state = GameState.RUNNING
        self.events: list[Event] = []
        self.particles: list[Particle] = []
        self.on_pad_timer = 0.0
        self.in_no_fly = False
        self.returning_home = False   # автопилот летит заряжаться
        self.ap_committed_to: Pad | None = None

        self.buildings: list[Building] = []
        self.pads: list[Pad] = []
        self.no_fly: list[NoFlyZone] = []
        self.birds: list[Bird] = []
        self._generate_city()

        base = self.pads[0]
        self.drone = Drone(x=base.x, y=base.y + cfg.DRONE_HEIGHT / 2)
        self.controller = FlightController()
        self.wind = Wind(self.rng, strength=1.5)
        self.mission = self._new_mission()

    # ----------------------------------------------------------------- генерация
    def _generate_city(self) -> None:
        rng = self.rng
        # База — на земле в начале карты.
        self.pads.append(Pad("БАЗА", PadKind.BASE, 40.0, 0.0, width=10.0))

        x = 70.0
        idx = 0
        while x < cfg.WORLD_WIDTH - 60:
            width = rng.uniform(14, 34)
            # Чем дальше от базы, тем выше небоскрёбы.
            tallness = 0.4 + 0.6 * x / cfg.WORLD_WIDTH
            height = rng.uniform(10, 25) + rng.random() ** 2 * 85 * tallness
            b = Building(x, width, height, seed=rng.randrange(10_000))
            self.buildings.append(b)
            if rng.random() < 0.45:
                kind = PadKind.DEPOT if rng.random() < 0.35 else PadKind.CLIENT
                self.pads.append(Pad(f"{kind.value} {idx + 1}", kind, x + width / 2, height,
                                     width=min(7.0, width - 3)))
                idx += 1
            x += width + rng.uniform(8, 30)

        # Пара наземных площадок между домами.
        for _ in range(2):
            gx = self._find_ground_gap()
            if gx is not None:
                self.pads.append(Pad(f"Клиент {idx + 1}", PadKind.CLIENT, gx, 0.0))
                idx += 1

        # Если складов не оказалось, назначаем.
        if not any(p.kind is PadKind.DEPOT for p in self.pads):
            self.pads[1].kind = PadKind.DEPOT
            self.pads[1].name = "Склад 1"

        # Бесполётная зона над случайным участком, не задевая площадки.
        zx = rng.uniform(cfg.WORLD_WIDTH * 0.35, cfg.WORLD_WIDTH * 0.7)
        self.no_fly.append(NoFlyZone(zx, zx + 90, 85, cfg.CEILING + 40))

    def _find_ground_gap(self) -> float | None:
        for _ in range(40):
            x = self.rng.uniform(150, cfg.WORLD_WIDTH - 50)
            if self.surface_height(x - 6) == 0 and self.surface_height(x + 6) == 0 \
                    and all(abs(p.x - x) > 15 for p in self.pads):
                return x
        return None

    def _new_mission(self) -> Mission:
        rng = self.rng
        depots = [p for p in self.pads if p.kind is PadKind.DEPOT]
        clients = [p for p in self.pads if p.kind is PadKind.CLIENT]
        depots.sort(key=lambda p: abs(p.x - self.drone.x))
        pickup = rng.choice(depots[:2])
        # Чем выше уровень, тем дальше клиенты.
        clients.sort(key=lambda p: abs(p.x - pickup.x))
        span = max(1, min(len(clients), 2 + self.level))
        dropoff = rng.choice(clients[:span])
        name, mass, fragile = rng.choice(CARGO)
        # Миссия должна быть выполнима с полной батареей: база → склад → клиент → база.
        for _ in range(30):
            route = (self.leg_cost(self.base.x, pickup.x, 0.0)
                     + self.leg_cost(pickup.x, dropoff.x, mass)
                     + self.leg_cost(dropoff.x, self.base.x, 0.0))
            if route < 85:
                break
            pickup = rng.choice(depots)
            dropoff = rng.choice(clients)
            name, mass, fragile = rng.choice(CARGO)
        d = self.drone
        dist = abs(d.x - pickup.x) + abs(pickup.x - dropoff.x)
        need = (4.0 + self.leg_cost(d.x, pickup.x, d.payload)
                + self.leg_cost(pickup.x, dropoff.x, mass)
                + self.leg_cost(dropoff.x, self.base.x, 0.0))
        if d.battery < need:
            # Придётся заскочить на базу — честно добавляем время на крюк.
            dist = abs(d.x - self.base.x) + abs(self.base.x - pickup.x) + abs(pickup.x - dropoff.x)
        time_limit = 60 + dist / 4.0
        reward = int(cfg.BASE_REWARD + dist * 0.6 + self.level * 40 + (80 if fragile else 0))
        return Mission(pickup, dropoff, name, mass, time_limit, reward, fragile)

    # ----------------------------------------------------------------- геометрия
    def surface_height(self, x: float) -> float:
        for b in self.buildings:
            if b.x <= x <= b.right:
                return b.height
        return 0.0

    def cruise_altitude(self, x1: float, x2: float) -> float:
        lo, hi = min(x1, x2) - 10, max(x1, x2) + 10
        tallest = max((b.height for b in self.buildings if b.right >= lo and b.x <= hi), default=0.0)
        return tallest + 14.0

    def safe_speed(self, x: float, target_x: float, bottom: float, climb_rate: float = 5.0) -> float:
        """Максимальная горизонтальная скорость, при которой дрон успеет
        набрать высоту над каждым зданием на пути к цели."""
        direction = 1.0 if target_x >= x else -1.0
        half_w = self.drone.spec.width / 2
        limit = 15.0
        for b in self.buildings:
            near_edge = b.x if direction > 0 else b.right
            gap = (near_edge - x) * direction - half_w
            if gap < -1.0 or gap > 160:
                continue  # здание позади или очень далеко
            # Здание за целью не мешает.
            if (b.x - target_x) * direction > 0 and not (b.x <= target_x <= b.right):
                continue
            deficit = b.height + 3.0 - bottom
            if deficit <= 0:
                continue
            time_to_climb = deficit / climb_rate
            limit = min(limit, max(0.0, gap - 2.0) / time_to_climb * 0.7)
        return limit

    def pad_under_drone(self) -> Pad | None:
        d = self.drone
        for p in self.pads:
            if p.contains(d.x) and abs(d.bottom - p.y) < 0.3:
                return p
        return None

    # ----------------------------------------------------------------- события
    def log(self, text: str, color: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.events.append(Event(text, color))
        self.events = self.events[-6:]

    def burst(self, x: float, y: float, color: tuple[int, int, int], count: int = 25,
              speed: float = 8.0) -> None:
        for _ in range(count):
            a = self.rng.uniform(0, 2 * math.pi)
            s = self.rng.uniform(0.2, 1.0) * speed
            self.particles.append(Particle(x, y, math.cos(a) * s, math.sin(a) * s,
                                           self.rng.uniform(0.5, 1.4), color,
                                           self.rng.uniform(1.5, 3.5)))

    # ----------------------------------------------------------------- шаг
    @property
    def base(self) -> Pad:
        return self.pads[0]

    def leg_cost(self, x1: float, x2: float, payload: float) -> float:
        """Оценка расхода батареи (%) на перелёт между точками с заданным грузом."""
        d = self.drone
        hover = clamp((d.spec.mass + payload) * cfg.GRAVITY / d.spec.max_thrust, 0.0, 1.0)
        flight_time = abs(x2 - x1) / 13.0 + 14.0     # крейсер + набор высоты и посадка
        wind_factor = 1.0 + 0.05 * abs(self.wind.base)  # против ветра моторы работают сильнее
        return flight_time * d.drain_rate(hover) * wind_factor * 1.1

    def battery_needed(self) -> float:
        """Сколько заряда нужно, чтобы долететь до цели миссии, а потом домой."""
        d, m = self.drone, self.mission
        target = m.target
        payload_after = m.cargo_mass if not m.picked_up else 0.0
        return (4.0 + self.leg_cost(d.x, target.x, d.payload)
                + self.leg_cost(target.x, self.base.x, payload_after))

    def autopilot_target(self) -> Pad:
        d = self.drone
        if self.controller.mode is not FlightMode.AUTO:
            self.returning_home = False
            return self.mission.target

        at_base = d.landed and self.pad_under_drone() is self.base
        if self.returning_home:
            if at_base and d.battery >= 99 and d.hull >= 99:
                self.returning_home = False
                # С полной батареей лучше уже не будет — летим до конца участка.
                self.ap_committed_to = self.mission.target
                self.log("Автопилот: заряжен, продолжаю миссию", (180, 220, 255))
            return self.base

        if at_base and d.battery >= 99 and d.hull >= 99:
            self.ap_committed_to = self.mission.target
        elif self.ap_committed_to is not self.mission.target:
            self.ap_committed_to = None
        emergency = d.battery < 3.0 + self.leg_cost(d.x, self.base.x, d.payload)
        planned = self.ap_committed_to is None and d.battery < self.battery_needed()
        if not at_base and (emergency or planned):
            self.returning_home = True
            self.ap_committed_to = None
            self.log("Автопилот: не хватит заряда, лечу на базу", (255, 210, 120))
            return self.base
        return self.mission.target

    def toggle_autopilot(self) -> None:
        c = self.controller
        if c.mode is FlightMode.AUTO:
            c.set_mode(FlightMode.ASSIST, self.drone)
            self.log("Автопилот выключен", (180, 220, 255))
        else:
            c.set_mode(FlightMode.AUTO, self.drone)
            self.mission.used_autopilot = True
            self.log("Автопилот включён: награда x0.5", (255, 210, 120))

    def toggle_manual(self) -> None:
        c = self.controller
        new = FlightMode.ASSIST if c.mode is FlightMode.MANUAL else FlightMode.MANUAL
        c.set_mode(new, self.drone)
        self.log(f"Режим: {new.value}", (180, 220, 255))

    def step(self, dt: float, pilot: PilotInput) -> None:
        if self.state is not GameState.RUNNING:
            self._step_particles(dt)
            return

        self.time += dt
        d = self.drone
        target = self.autopilot_target()
        cruise = self.cruise_altitude(d.x, target.x)
        max_speed = self.safe_speed(d.x, target.x, d.bottom)
        throttle, tilt = self.controller.command(d, pilot, dt, (target.x, target.y),
                                                 cruise, max_speed)
        wind = self.wind.at(self.time, d.y)
        prev_bottom = d.bottom

        d.step(dt, throttle, tilt, wind)
        self._collide(prev_bottom)
        self._update_birds(dt)
        self._check_zones(dt)
        self._update_mission(dt)
        self._downwash(dt)
        self._step_particles(dt)

        for e in self.events:
            e.age += dt

        if not d.alive:
            self.state = GameState.CRASHED
            self.burst(d.x, d.y, (255, 140, 40), 60, 14)
            self.burst(d.x, d.y, (90, 90, 90), 40, 6)
            self.log("Дрон уничтожен!", (255, 80, 80))
        elif d.battery <= 0 and d.landed:
            self.state = GameState.OUT_OF_POWER
            self.log("Батарея разряжена", (255, 80, 80))

    # ----------------------------------------------------------------- столкновения
    def _collide(self, prev_bottom: float) -> None:
        d = self.drone
        half_w = d.spec.width / 2
        half_h = d.spec.height / 2

        # Границы карты и потолок.
        if d.x < half_w:
            d.x, d.vx = half_w, abs(d.vx) * 0.3
        if d.x > cfg.WORLD_WIDTH - half_w:
            d.x, d.vx = cfg.WORLD_WIDTH - half_w, -abs(d.vx) * 0.3
        if d.y > cfg.CEILING:
            d.vy = min(d.vy, 0.0) - 0.5

        # Боковые удары о стены зданий.
        for b in self.buildings:
            overlap_x = d.x + half_w > b.x and d.x - half_w < b.right
            if not overlap_x or d.bottom >= b.height:
                continue
            if prev_bottom >= b.height - 0.05:
                continue  # это посадка сверху, ниже
            impact = abs(d.vx)
            if d.x < b.x + b.width / 2:
                d.x = b.x - half_w - 0.01
                d.vx = -abs(d.vx) * 0.35
            else:
                d.x = b.right + half_w + 0.01
                d.vx = abs(d.vx) * 0.35
            if impact > 1.0:
                dmg = impact * 4.5
                d.damage(dmg)
                self._hurt_cargo(dmg * 1.5)
                self.burst(d.x, d.y, (255, 220, 120), 12, impact)
                self.log(f"Удар о здание! -{dmg:.0f}% корпуса", (255, 120, 80))

        # Контакт с поверхностью сверху (крыша или земля).
        surface = max(self.surface_height(d.x - half_w), self.surface_height(d.x + half_w))
        if d.landed and d.bottom > surface + 0.05:
            d.landed = False  # съехали с края крыши — падаем
        if d.bottom <= surface:
            impact = -d.vy
            d.y = surface + half_h
            was_landed = d.landed
            d.vy = 0.0
            d.vx *= 0.6
            d.landed = True
            if not was_landed and impact > 0.3:
                self._landing_impact(impact)

    def _landing_impact(self, impact: float) -> None:
        d = self.drone
        tilt = abs(d.angle)
        if impact >= cfg.FATAL_IMPACT:
            d.damage(100)
            return
        bad = 0.0
        if impact > cfg.SAFE_LANDING_VY:
            bad += (impact - cfg.SAFE_LANDING_VY) * 9
        if tilt > cfg.SAFE_LANDING_TILT:
            bad += math.degrees(tilt - cfg.SAFE_LANDING_TILT) * 1.2
        if abs(d.vx) > cfg.SAFE_LANDING_VX:
            bad += (abs(d.vx) - cfg.SAFE_LANDING_VX) * 5
        if bad > 0:
            d.damage(bad)
            self._hurt_cargo(bad * 2)
            self.burst(d.x, d.bottom, (200, 200, 200), 15, 4)
            self.log(f"Жёсткая посадка! -{bad:.0f}% корпуса", (255, 160, 80))
        elif impact > 0.5:
            self.score += 10
            self.log("Мягкая посадка +10", (140, 255, 160))

    def _hurt_cargo(self, amount: float) -> None:
        m = self.mission
        if m.picked_up:
            factor = 2.0 if m.fragile else 1.0
            m.cargo_health = max(0.0, m.cargo_health - amount * factor)

    # ----------------------------------------------------------------- птицы и зоны
    def _update_birds(self, dt: float) -> None:
        d = self.drone
        max_birds = min(1 + self.level, 9)
        if len(self.birds) < max_birds and self.rng.random() < dt * 0.4:
            side = self.rng.choice((-1, 1))
            x = d.x + side * 120
            y = self.rng.uniform(20, 110)
            self.birds.append(Bird(x, y, -side * self.rng.uniform(5, 10), self.rng.uniform(0, 6),
                                   base_y=y, amplitude=self.rng.uniform(1, 4)))
        for bird in self.birds:
            bird.step(dt)
            if abs(bird.x - d.x) < 1.3 and abs(bird.y - d.y) < 1.0:
                bird.alive = False
                d.damage(15)
                self._hurt_cargo(15)
                d.vx += bird.vx * 0.3
                self.burst(bird.x, bird.y, (240, 240, 240), 18, 5)
                self.log("Столкновение с птицей! -15%", (255, 120, 80))
        self.birds = [b for b in self.birds if b.alive and abs(b.x - d.x) < 220]

    def _check_zones(self, dt: float) -> None:
        d = self.drone
        inside = any(z.contains(d.x, d.y) for z in self.no_fly)
        if inside and not self.in_no_fly:
            self.log("Бесполётная зона! Штраф каждую секунду", (255, 80, 80))
        self.in_no_fly = inside
        if inside:
            self.score -= int(40 * dt + self.rng.random())

    # ----------------------------------------------------------------- миссия
    def _update_mission(self, dt: float) -> None:
        d = self.drone
        m = self.mission
        pad = self.pad_under_drone() if d.landed else None
        charging = pad is not None and pad.kind is PadKind.BASE
        if not charging:
            m.elapsed += dt   # на зарядке таймер миссии стоит
        if self.controller.mode is FlightMode.AUTO:
            m.used_autopilot = True

        if pad is not None and pad.kind is PadKind.BASE:
            if d.battery < 100 or d.hull < 100:
                d.battery = min(100.0, d.battery + cfg.BATTERY_CHARGE_RATE * dt)
                d.hull = min(100.0, d.hull + cfg.HULL_REPAIR_RATE * dt)

        if pad is m.target and d.speed < 0.5:
            self.on_pad_timer += dt
            if self.on_pad_timer >= cfg.LOAD_TIME:
                self.on_pad_timer = 0.0
                if not m.picked_up:
                    m.picked_up = True
                    d.payload = m.cargo_mass
                    self.log(f"Груз «{m.cargo_name}» на борту ({m.cargo_mass:.1f} кг)",
                             (140, 220, 255))
                else:
                    self._complete_mission()
        else:
            self.on_pad_timer = 0.0

        if self.mission is m and m.time_left <= 0:
            self.score -= cfg.MISSION_FAIL_PENALTY
            self.log(f"Время вышло! -{cfg.MISSION_FAIL_PENALTY}", (255, 80, 80))
            d.payload = 0.0
            self.mission = self._new_mission()

    def _complete_mission(self) -> None:
        m = self.mission
        d = self.drone
        time_bonus = max(0.0, m.time_left) * 4
        reward = (m.reward + time_bonus) * (m.cargo_health / 100)
        if m.used_autopilot:
            reward *= cfg.AUTOPILOT_PENALTY
        reward = int(reward)
        self.score += reward
        self.deliveries += 1
        d.payload = 0.0
        self.burst(d.x, d.y + 2, (120, 255, 140), 40, 9)
        self.log(f"Доставлено! +{reward}", (120, 255, 140))
        if self.deliveries % 2 == 0:
            self.level += 1
            self.wind = Wind(self.rng, strength=1.5 + self.level * 1.1)
            self.log(f"Уровень {self.level}: ветер усиливается", (255, 220, 120))
        if self.controller.mode is FlightMode.AUTO:
            self.controller.set_mode(FlightMode.ASSIST, d)
        self.mission = self._new_mission()

    # ----------------------------------------------------------------- частицы
    def _downwash(self, dt: float) -> None:
        d = self.drone
        if d.throttle < 0.2 or not d.alive:
            return
        height = d.bottom - self.surface_height(d.x)
        if height < 8 and self.rng.random() < d.throttle * 0.9:
            for side in (-1, 1):
                self.particles.append(Particle(
                    d.x + self.rng.uniform(-0.5, 0.5), d.y - height + 0.1,
                    side * self.rng.uniform(3, 8) * d.throttle, self.rng.uniform(0, 1.0),
                    self.rng.uniform(0.3, 0.7), (190, 180, 150), 2))

    def _step_particles(self, dt: float) -> None:
        for p in self.particles:
            p.life -= dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy -= 6 * dt
        self.particles = [p for p in self.particles if p.life > 0][-400:]
