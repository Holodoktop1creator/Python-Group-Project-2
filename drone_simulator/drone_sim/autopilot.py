"""Полётный контроллер: режимы управления и автопилот на PID-регуляторах."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .physics import Drone, clamp


class FlightMode(Enum):
    MANUAL = "РУЧНОЙ"        # W/S напрямую меняют газ
    ASSIST = "СТАБИЛИЗАЦИЯ"  # W/S задают вертикальную скорость, газ считает PID
    AUTO = "АВТОПИЛОТ"       # дрон сам летит к цели


@dataclass
class PilotInput:
    """Сигналы с «пульта»: -1..1 по каждой оси."""
    vertical: float = 0.0
    horizontal: float = 0.0


class PID:
    def __init__(self, kp: float, ki: float, kd: float, out_limit: float = 1.0,
                 i_limit: float = 1.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.integral = 0.0
        self.prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        self.integral = clamp(self.integral + error * dt, -self.i_limit, self.i_limit)
        derivative = 0.0 if self.prev_error is None or dt <= 0 else (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(out, -self.out_limit, self.out_limit)


@dataclass
class FlightController:
    mode: FlightMode = FlightMode.ASSIST
    manual_throttle: float = 0.0
    max_climb: float = 5.0       # м/с
    max_descent: float = 4.0     # м/с
    vs_pid: PID = field(default_factory=lambda: PID(0.16, 0.08, 0.0, out_limit=0.6, i_limit=2.0))
    vel_pid: PID = field(default_factory=lambda: PID(0.22, 0.02, 0.0, out_limit=1.0, i_limit=3.0))

    def set_mode(self, mode: FlightMode, drone: Drone) -> None:
        if mode is self.mode:
            return
        # Переключение без рывка: ручной газ стартует с текущего значения.
        self.manual_throttle = drone.throttle
        self.vs_pid.reset()
        self.vel_pid.reset()
        self.mode = mode

    # --------------------------------------------------------------- низкий уровень
    def _hold_vertical_speed(self, drone: Drone, target_vy: float, dt: float) -> float:
        hover = drone.hover_throttle()
        correction = self.vs_pid.update(target_vy - drone.vy, dt)
        return clamp(hover + correction, 0.0, 1.0)

    def _hold_horizontal_speed(self, drone: Drone, target_vx: float, dt: float) -> float:
        return self.vel_pid.update(target_vx - drone.vx, dt)

    # --------------------------------------------------------------- команды
    def command(self, drone: Drone, pilot: PilotInput, dt: float,
                target: tuple[float, float] | None = None,
                cruise_alt: float = 40.0, max_speed: float = 15.0) -> tuple[float, float]:
        """Вернуть (газ, наклон) для дрона с учётом режима.

        cruise_alt — безопасная высота над всем маршрутом,
        max_speed — допустимая горизонтальная скорость (чтобы успеть набрать высоту).
        """
        if self.mode is FlightMode.MANUAL:
            self.manual_throttle = clamp(self.manual_throttle + pilot.vertical * 0.7 * dt, 0.0, 1.0)
            return self.manual_throttle, pilot.horizontal

        if self.mode is FlightMode.ASSIST or target is None:
            if pilot.vertical >= 0:
                target_vy = pilot.vertical * self.max_climb
            else:
                target_vy = pilot.vertical * self.max_descent
            if drone.landed and pilot.vertical <= 0:
                self.vs_pid.reset()
                return 0.0, pilot.horizontal
            return self._hold_vertical_speed(drone, target_vy, dt), pilot.horizontal

        return self._autopilot(drone, target, cruise_alt, max_speed, dt)

    def _autopilot(self, drone: Drone, target: tuple[float, float], cruise_alt: float,
                   max_speed: float, dt: float) -> tuple[float, float]:
        tx, ty = target
        dx = tx - drone.x
        dist = abs(dx)

        # Горизонталь: желаемая скорость пропорциональна расстоянию, у цели — тормозим.
        # Скорость ограничена так, чтобы успеть подняться над зданиями впереди.
        limit = min(15.0, max_speed)
        target_vx = clamp(dx * 0.35, -limit, limit)
        tilt = self._hold_horizontal_speed(drone, target_vx, dt)

        # Вертикаль: крейсерская высота, над целью — снижение (быстрое, потом мягкое).
        if dist > 3.0 or abs(drone.vx) > 1.2:
            desired_alt = max(cruise_alt, ty + 6)
            target_vy = clamp((desired_alt - drone.y) * 0.6, -6.0, 7.0)
        else:
            height_above = drone.bottom - ty
            target_vy = -clamp(height_above * 0.45, 0.6, 6.0)

        if drone.landed and dist <= 3.0:
            return 0.0, 0.0
        return self._hold_vertical_speed(drone, target_vy, dt), tilt
