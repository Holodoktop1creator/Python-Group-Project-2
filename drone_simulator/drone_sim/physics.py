"""Физическая модель квадрокоптера (вид сбоку, 2D).

Модель упрощённая, но честная:
  * тяга направлена вдоль оси дрона и пропорциональна газу;
  * у моторов есть инерция (газ догоняет команду с задержкой);
  * дрон наклоняется с ограниченной угловой скоростью (режим «angle mode»);
  * сопротивление воздуха считается относительно ветра, поэтому ветер сносит дрон;
  * груз увеличивает массу, а значит и расход батареи.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config as cfg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class DroneSpec:
    mass: float = cfg.DRONE_MASS
    max_thrust: float = cfg.DRONE_MAX_THRUST
    max_tilt: float = cfg.DRONE_MAX_TILT
    tilt_rate: float = cfg.DRONE_TILT_RATE
    motor_lag: float = cfg.DRONE_MOTOR_LAG
    drag: float = cfg.DRONE_DRAG
    width: float = cfg.DRONE_WIDTH
    height: float = cfg.DRONE_HEIGHT


@dataclass
class Drone:
    x: float
    y: float
    spec: DroneSpec = field(default_factory=DroneSpec)
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0          # рад, положительный — наклон вправо
    throttle: float = 0.0       # фактический газ 0..1
    battery: float = 100.0      # %
    hull: float = 100.0         # % целостности корпуса
    payload: float = 0.0        # кг груза
    landed: bool = True
    rotor_phase: float = 0.0    # для анимации винтов

    # ------------------------------------------------------------------ свойства
    @property
    def total_mass(self) -> float:
        return self.spec.mass + self.payload

    @property
    def alive(self) -> bool:
        return self.hull > 0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def bottom(self) -> float:
        return self.y - self.spec.height / 2

    def hover_throttle(self) -> float:
        """Газ, при котором дрон висит на месте при текущем наклоне."""
        cos_a = max(math.cos(self.angle), 0.3)
        return clamp(self.total_mass * cfg.GRAVITY / (self.spec.max_thrust * cos_a), 0.0, 1.0)

    @staticmethod
    def drain_rate(throttle: float) -> float:
        """Расход батареи в %/с: электроника + моторы (нелинейно от газа)."""
        return cfg.BATTERY_IDLE_DRAIN + cfg.BATTERY_THRUST_DRAIN * max(throttle, 0.0) ** 1.5

    # ------------------------------------------------------------------ шаг
    def step(self, dt: float, throttle_cmd: float, tilt_cmd: float,
             wind: tuple[float, float] = (0.0, 0.0)) -> None:
        """Продвинуть физику на dt секунд.

        throttle_cmd — желаемый газ 0..1, tilt_cmd — желаемый наклон -1..1.
        Столкновения с миром обрабатывает Simulation, не дрон.
        """
        if not self.alive:
            throttle_cmd, tilt_cmd = 0.0, 0.0
        if self.battery <= 0:
            self.battery = 0.0
            throttle_cmd = 0.0

        throttle_cmd = clamp(throttle_cmd, 0.0, 1.0)
        tilt_cmd = clamp(tilt_cmd, -1.0, 1.0)

        # Инерция моторов: экспоненциальное приближение к команде.
        k = 1.0 - math.exp(-dt / self.spec.motor_lag)
        self.throttle += (throttle_cmd - self.throttle) * k

        # Наклон с ограниченной скоростью поворота.
        target_angle = tilt_cmd * self.spec.max_tilt
        max_delta = self.spec.tilt_rate * dt
        self.angle += clamp(target_angle - self.angle, -max_delta, max_delta)

        thrust = self.throttle * self.spec.max_thrust
        fx = thrust * math.sin(self.angle)
        fy = thrust * math.cos(self.angle) - self.total_mass * cfg.GRAVITY

        # Сопротивление воздуха относительно ветра.
        wind_x, wind_y = wind
        fx += self.spec.drag * (wind_x - self.vx)
        fy += self.spec.drag * (wind_y - self.vy)

        if self.landed and fy <= 0:
            # Стоим на поверхности: опора гасит вертикальную силу.
            fy = 0.0
            self.vy = 0.0
        else:
            self.landed = False

        ax = fx / self.total_mass
        ay = fy / self.total_mass
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Батарея: тяжёлый дрон держит больший газ, а значит и тратит больше.
        self.battery = max(0.0, self.battery - self.drain_rate(self.throttle) * dt)

        self.rotor_phase = (self.rotor_phase + (5 + 60 * self.throttle) * dt) % (2 * math.pi)

    def damage(self, amount: float) -> None:
        self.hull = max(0.0, self.hull - amount)
