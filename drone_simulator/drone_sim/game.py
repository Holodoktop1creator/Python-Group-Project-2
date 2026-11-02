"""Отрисовка и игровой цикл на pygame."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pygame

from . import config as cfg
from .autopilot import FlightMode, PilotInput
from .world import GameState, PadKind, Simulation

WIDTH, HEIGHT = 1280, 720
PPM = 7.0                     # пикселей на метр
PHYSICS_DT = 1 / 120          # фиксированный шаг физики
HIGHSCORE_FILE = Path(__file__).resolve().parent.parent / "highscore.json"

# Палитра
SKY_TOP = (24, 44, 92)
SKY_BOTTOM = (250, 170, 120)
GROUND = (52, 58, 52)
SOIL = (34, 30, 28)
WHITE = (240, 244, 250)
DIM = (150, 160, 180)
PANEL = (10, 16, 30)
ACCENT = (90, 200, 255)
GOOD = (110, 230, 140)
WARN = (255, 200, 90)
BAD = (255, 90, 90)
PAD_COLORS = {PadKind.BASE: (80, 160, 255), PadKind.DEPOT: (255, 160, 60),
              PadKind.CLIENT: (110, 230, 140)}


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Camera:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0

    def follow(self, x: float, y: float, dt: float, snap: bool = False) -> None:
        min_y = HEIGHT / (2 * PPM) - 6       # не показываем слишком много подземелья
        ty = max(y + 6, min_y)
        if snap:
            self.x, self.y = x, ty
            return
        k = 1 - math.exp(-dt * 4)
        self.x += (x - self.x) * k
        self.y += (ty - self.y) * k

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return WIDTH / 2 + (x - self.x) * PPM, HEIGHT / 2 - (y - self.y) * PPM

    def visible(self, x1: float, x2: float, margin: float = 20) -> bool:
        half = WIDTH / (2 * PPM) + margin
        return x2 >= self.x - half and x1 <= self.x + half


class Game:
    def __init__(self, seed: int | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Небесный курьер — симулятор дрона")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 22)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)
        self.font_xl = pygame.font.Font(None, 84)
        self.seed = seed
        self.highscore = self._load_highscore()
        self.sky = self._make_sky()
        self.state = "title"          # title | play | pause | over
        self.show_help = False
        self.new_game()

    # ------------------------------------------------------------------ сервис
    def _load_highscore(self) -> int:
        try:
            return int(json.loads(HIGHSCORE_FILE.read_text())["best"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def _save_highscore(self) -> None:
        try:
            HIGHSCORE_FILE.write_text(json.dumps({"best": self.highscore}))
        except OSError:
            pass

    def _make_sky(self) -> pygame.Surface:
        sky = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            pygame.draw.line(sky, lerp_color(SKY_TOP, SKY_BOTTOM, y / HEIGHT), (0, y), (WIDTH, y))
        return sky

    def new_game(self) -> None:
        self.sim = Simulation(self.seed)
        self.camera = Camera()
        self.camera.follow(self.sim.drone.x, self.sim.drone.y, 0, snap=True)
        self.accumulator = 0.0
        self.over_saved = False
        rng = random.Random(self.sim.seed)
        self.building_surfaces = [self._render_building(b) for b in self.sim.buildings]
        self.mountains = self._make_mountains(rng)
        self.clouds = [[rng.uniform(0, cfg.WORLD_WIDTH), rng.uniform(60, 150),
                        rng.uniform(10, 26), rng.uniform(0.3, 0.6)] for _ in range(26)]
        self.streaks = [[rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT), rng.uniform(20, 60)]
                        for _ in range(45)]

    def _render_building(self, b) -> pygame.Surface:
        w, h = max(2, int(b.width * PPM)), max(2, int(b.height * PPM))
        rng = random.Random(b.seed)
        base = rng.choice([(58, 66, 92), (70, 64, 84), (52, 74, 88), (84, 72, 70), (60, 60, 70)])
        surf = pygame.Surface((w, h))
        surf.fill(base)
        pygame.draw.rect(surf, lerp_color(base, (0, 0, 0), 0.3), (w - 6, 0, 6, h))  # тень
        win_w, win_h = 9, 12
        for wy in range(10, h - 8, 22):
            for wx in range(8, w - 14, 18):
                lit = rng.random() < 0.38
                color = (255, 220, 140) if lit else lerp_color(base, (0, 0, 0), 0.35)
                surf.fill(color, (wx, wy, win_w, win_h))
        pygame.draw.rect(surf, lerp_color(base, (255, 255, 255), 0.25), (0, 0, w, 4))
        return surf

    def _make_mountains(self, rng: random.Random) -> list[tuple[float, float]]:
        pts = []
        x = -200.0
        while x < cfg.WORLD_WIDTH * 0.3 + 600:
            pts.append((x, rng.uniform(30, 90)))
            x += rng.uniform(40, 90)
        return pts

    # ------------------------------------------------------------------ цикл
    def run(self) -> None:
        running = True
        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.on_key(event.key)
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def on_key(self, key: int) -> bool:
        if self.state == "title":
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.state = "play"
            elif key == pygame.K_ESCAPE:
                return False
        elif self.state == "play":
            if key in (pygame.K_p, pygame.K_ESCAPE):
                self.state = "pause"
            elif key == pygame.K_t:
                self.sim.toggle_autopilot()
            elif key == pygame.K_m:
                self.sim.toggle_manual()
            elif key == pygame.K_h:
                self.show_help = not self.show_help
        elif self.state == "pause":
            if key in (pygame.K_p, pygame.K_ESCAPE, pygame.K_RETURN):
                self.state = "play"
            elif key == pygame.K_q:
                return False
            elif key == pygame.K_r:
                self.new_game()
                self.state = "play"
        elif self.state == "over":
            if key in (pygame.K_r, pygame.K_RETURN):
                self.new_game()
                self.state = "play"
            elif key == pygame.K_ESCAPE:
                return False
        return True

    def read_pilot(self) -> PilotInput:
        keys = pygame.key.get_pressed()
        v = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
        h = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        return PilotInput(float(v), float(h))

    def update(self, dt: float) -> None:
        sim = self.sim
        if self.state in ("play", "over"):
            pilot = self.read_pilot() if self.state == "play" else PilotInput()
            self.accumulator += dt
            while self.accumulator >= PHYSICS_DT:
                sim.step(PHYSICS_DT, pilot)
                self.accumulator -= PHYSICS_DT
            if sim.state is not GameState.RUNNING and self.state == "play":
                self.state = "over"
            if self.state == "over" and not self.over_saved:
                self.over_saved = True
                if sim.score > self.highscore:
                    self.highscore = sim.score
                    self._save_highscore()
        self.camera.follow(sim.drone.x, sim.drone.y, dt)

        # Облака и полосы ветра двигаются по ветру.
        wind_x, _ = sim.wind.at(sim.time, sim.drone.y)
        for c in self.clouds:
            c[0] = (c[0] + wind_x * dt * 0.5) % cfg.WORLD_WIDTH
        for s in self.streaks:
            s[0] += wind_x * PPM * dt * 1.6
            if s[0] < -80:
                s[0] += WIDTH + 160
            elif s[0] > WIDTH + 80:
                s[0] -= WIDTH + 160

    # ------------------------------------------------------------------ рисование
    def draw(self) -> None:
        self.screen.blit(self.sky, (0, 0))
        self.draw_background()
        self.draw_world()
        self.draw_drone()
        self.draw_particles()
        self.draw_hud()
        self.draw_minimap()
        self.draw_events()
        if self.show_help and self.state == "play":
            self.draw_help_panel()
        if self.state == "title":
            self.draw_title()
        elif self.state == "pause":
            self.draw_overlay("ПАУЗА", ["P / Esc / Enter — продолжить", "R — заново", "Q — выйти"])
        elif self.state == "over":
            self.draw_game_over()

    def text(self, txt: str, font, color, pos, center: bool = False, shadow: bool = True):
        surf = font.render(txt, True, color)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        if shadow:
            self.screen.blit(font.render(txt, True, (0, 0, 0)), rect.move(2, 2))
        self.screen.blit(surf, rect)
        return rect

    def draw_background(self) -> None:
        cam = self.camera
        # Солнце
        pygame.draw.circle(self.screen, (255, 230, 170), (int(WIDTH * 0.78), int(HEIGHT * 0.62)), 46)
        # Горы (параллакс 0.25)
        par = 0.25
        pts = []
        for mx, mh in self.mountains:
            sx = WIDTH / 2 + (mx - cam.x * par) * PPM * 0.6
            sy = HEIGHT / 2 + 236 + cam.y * PPM * par - mh * PPM * 0.6
            pts.append((sx, sy))
        if pts:
            poly = [(pts[0][0], HEIGHT)] + pts + [(pts[-1][0], HEIGHT)]
            pygame.draw.polygon(self.screen, (92, 84, 120), poly)
        # Облака (параллакс 0.5)
        col = lerp_color((255, 255, 255), SKY_BOTTOM, 0.3)
        for cx, cy, size, _ in self.clouds:
            sx = WIDTH / 2 + (cx - cam.x) * PPM * 0.5
            sy = HEIGHT / 2 - (cy - cam.y) * PPM * 0.5
            if -200 < sx < WIDTH + 200 and -100 < sy < HEIGHT + 100:
                r = int(size * 1.6)
                for ox, oy, rr in ((-r, 4, r), (0, -4, int(r * 1.3)), (r, 4, r)):
                    pygame.draw.circle(self.screen, col, (int(sx + ox), int(sy + oy)), rr)

    def draw_world(self) -> None:
        sim, cam, scr = self.sim, self.camera, self.screen
        t = sim.time

        # Бесполётные зоны
        for z in sim.no_fly:
            if not cam.visible(z.x1, z.x2):
                continue
            x1, y1 = cam.to_screen(z.x1, z.y2)
            x2, y2 = cam.to_screen(z.x2, z.y1)
            rect = pygame.Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
            overlay.fill((255, 40, 40, 45))
            for i in range(-rect.h, rect.w, 28):
                pygame.draw.line(overlay, (255, 60, 60, 70), (i, rect.h), (i + rect.h, 0), 6)
            scr.blit(overlay, rect)
            pygame.draw.rect(scr, (255, 80, 80), rect, 2)
            self.text(f"{z.label}: НЕ ЛЕТАТЬ", self.font_s, BAD, (rect.x + 8, rect.bottom - 24))

        # Здания
        for b, surf in zip(sim.buildings, self.building_surfaces):
            if not cam.visible(b.x, b.right):
                continue
            sx, sy = cam.to_screen(b.x, b.height)
            scr.blit(surf, (int(sx), int(sy)))

        # Земля
        gx, gy = cam.to_screen(0, 0)
        if gy < HEIGHT:
            pygame.draw.rect(scr, GROUND, (0, int(gy), WIDTH, 10))
            pygame.draw.rect(scr, SOIL, (0, int(gy) + 10, WIDTH, HEIGHT))
        # Края карты
        for edge in (0, cfg.WORLD_WIDTH):
            ex, _ = cam.to_screen(edge, 0)
            if -10 < ex < WIDTH + 10:
                pygame.draw.line(scr, BAD, (ex, 0), (ex, HEIGHT), 3)

        # Площадки
        target = sim.autopilot_target() if sim.controller.mode is FlightMode.AUTO else sim.mission.target
        for p in sim.pads:
            if not cam.visible(p.x - p.width, p.x + p.width):
                continue
            color = PAD_COLORS[p.kind]
            x1, y1 = cam.to_screen(p.x - p.width / 2, p.y)
            w = p.width * PPM
            pygame.draw.rect(scr, (30, 30, 36), (x1, y1 - 5, w, 7))
            pygame.draw.rect(scr, color, (x1, y1 - 5, w, 3))
            for i in range(4):  # огни по краям
                on = int(t * 2 + i) % 2 == 0
                lx = x1 + (w - 4) * (i / 3)
                pygame.draw.circle(scr, color if on else (60, 60, 60), (int(lx + 2), int(y1 - 7)), 2)
            if p.kind is PadKind.BASE:
                self.text("H", self.font_m, WHITE, (x1 + w / 2, y1 - 22), center=True)
            self.text(p.name, self.font_s, color, (x1 + w / 2, y1 - 40), center=True)
            if p is target:
                pulse = (math.sin(t * 5) + 1) / 2
                ay = y1 - 70 - pulse * 10
                cx = x1 + w / 2
                pygame.draw.polygon(scr, WARN, [(cx - 12, ay), (cx + 12, ay), (cx, ay + 18)])
                beam = pygame.Surface((int(w), 120), pygame.SRCALPHA)
                beam.fill((*color, int(30 + 30 * pulse)))
                scr.blit(beam, (x1, y1 - 125))

        # Птицы
        for bird in sim.birds:
            bx, by = cam.to_screen(bird.x, bird.y)
            flap = math.sin(bird.phase * 4) * 6
            d = 1 if bird.vx > 0 else -1
            pygame.draw.lines(scr, (30, 30, 40), False,
                              [(bx - 9, by - flap), (bx, by + 2), (bx + 9, by - flap)], 3)
            pygame.draw.circle(scr, (30, 30, 40), (int(bx + 3 * d), int(by + 1)), 2)

        # Полосы ветра — показывают силу и направление
        wind_x, _ = sim.wind.at(t, sim.drone.y)
        length = max(4, abs(wind_x) * 6)
        alpha_col = lerp_color((255, 255, 255), SKY_BOTTOM, 0.5)
        for sx, sy, ln in self.streaks:
            if abs(wind_x) > 0.6:
                pygame.draw.line(scr, alpha_col, (sx, sy), (sx - math.copysign(length * ln / 40, wind_x), sy), 1)

    def draw_drone(self) -> None:
        d = self.sim.drone
        if not d.alive:
            return
        surf = pygame.Surface((90, 60), pygame.SRCALPHA)
        cx, cy = 45, 30
        body = pygame.Rect(0, 0, 26, 9)
        body.center = (cx, cy)
        # Груз под дроном
        if d.payload > 0:
            pygame.draw.line(surf, (40, 40, 40), (cx, cy + 4), (cx, cy + 14), 2)
            pygame.draw.rect(surf, (190, 140, 80), (cx - 7, cy + 14, 14, 11), border_radius=2)
            pygame.draw.line(surf, (120, 80, 40), (cx - 7, cy + 19), (cx + 7, cy + 19), 2)
        # Рама
        pygame.draw.line(surf, (60, 64, 72), (cx - 30, cy - 1), (cx + 30, cy - 1), 4)
        pygame.draw.rect(surf, (230, 232, 240), body, border_radius=4)
        pygame.draw.rect(surf, ACCENT, (cx - 6, cy - 2, 12, 4), border_radius=2)
        # Ножки
        pygame.draw.line(surf, (60, 64, 72), (cx - 9, cy + 4), (cx - 13, cy + 10), 2)
        pygame.draw.line(surf, (60, 64, 72), (cx + 9, cy + 4), (cx + 13, cy + 10), 2)
        # Винты
        for side in (-1, 1):
            mx = cx + side * 30
            pygame.draw.rect(surf, (40, 40, 48), (mx - 3, cy - 6, 6, 6))
            spin = abs(math.sin(d.rotor_phase + side)) if d.throttle > 0.05 else 1.0
            blade = 6 + 10 * spin
            pygame.draw.ellipse(surf, (200, 210, 230), (mx - blade, cy - 9, blade * 2, 4))
        # Огни
        blink = int(self.sim.time * 3) % 2 == 0
        pygame.draw.circle(surf, (255, 60, 60) if blink else (90, 20, 20), (cx - 30, cy + 2), 2)
        pygame.draw.circle(surf, (60, 255, 90) if not blink else (20, 90, 30), (cx + 30, cy + 2), 2)

        rotated = pygame.transform.rotate(surf, -math.degrees(d.angle))
        sx, sy = self.camera.to_screen(d.x, d.y)
        self.screen.blit(rotated, rotated.get_rect(center=(int(sx), int(sy))))

        # Индикатор погрузки
        timer = self.sim.on_pad_timer
        if timer > 0:
            frac = min(1.0, timer / cfg.LOAD_TIME)
            pygame.draw.arc(self.screen, GOOD, (sx - 22, sy - 48, 44, 44), math.pi / 2,
                            math.pi / 2 + frac * 2 * math.pi, 4)

    def draw_particles(self) -> None:
        for p in self.sim.particles:
            sx, sy = self.camera.to_screen(p.x, p.y)
            if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
                fade = min(1.0, p.life)
                col = lerp_color(SKY_BOTTOM, p.color, fade)
                pygame.draw.circle(self.screen, col, (int(sx), int(sy)), max(1, int(p.size)))

    # ------------------------------------------------------------------ HUD
    def bar(self, x, y, w, h, frac, color, label):
        pygame.draw.rect(self.screen, (30, 36, 52), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, color, (x, y, int(w * max(0.0, min(1.0, frac))), h), border_radius=4)
        pygame.draw.rect(self.screen, (90, 100, 120), (x, y, w, h), 1, border_radius=4)
        self.text(label, self.font_s, WHITE, (x + 6, y + 2), shadow=False)

    def panel(self, rect: pygame.Rect) -> None:
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((*PANEL, 170))
        self.screen.blit(s, rect)
        pygame.draw.rect(self.screen, (60, 80, 120), rect, 1, border_radius=6)

    def draw_hud(self) -> None:
        sim, d = self.sim, self.sim.drone
        # Левая панель — телеметрия
        self.panel(pygame.Rect(12, 12, 270, 214))
        alt = d.bottom - sim.surface_height(d.x)
        mode = sim.controller.mode
        mode_col = {FlightMode.MANUAL: WARN, FlightMode.ASSIST: ACCENT, FlightMode.AUTO: GOOD}[mode]
        self.text(f"Режим: {mode.value}", self.font_m, mode_col, (24, 20))
        rows = [
            ("Высота", f"{d.y:6.1f} м   (над землёй {alt:5.1f})"),
            ("Верт. скорость", f"{d.vy:+5.1f} м/с"),
            ("Гор. скорость", f"{d.vx:+5.1f} м/с"),
            ("Наклон", f"{math.degrees(d.angle):+5.0f}°"),
        ]
        for i, (k, v) in enumerate(rows):
            self.text(k, self.font_s, DIM, (24, 50 + i * 20), shadow=False)
            vy_col = BAD if (k == "Верт. скорость" and d.vy < -cfg.SAFE_LANDING_VY and alt < 15) else WHITE
            self.text(v, self.font_s, vy_col, (140, 50 + i * 20), shadow=False)
        bat_col = GOOD if d.battery > 40 else WARN if d.battery > 20 else BAD
        hull_col = GOOD if d.hull > 60 else WARN if d.hull > 30 else BAD
        self.bar(24, 136, 246, 18, d.battery / 100, bat_col, f"Батарея {d.battery:5.1f}%")
        self.bar(24, 160, 246, 18, d.hull / 100, hull_col, f"Корпус {d.hull:5.1f}%")
        self.bar(24, 184, 246, 18, d.throttle, (130, 140, 255), f"Газ {d.throttle * 100:3.0f}%")
        if d.battery < 20 and int(sim.time * 3) % 2 == 0:
            self.text("НИЗКИЙ ЗАРЯД — НА БАЗУ!", self.font_m, BAD, (WIDTH / 2, 120), center=True)

        # Правая панель — миссия
        m = sim.mission
        self.panel(pygame.Rect(WIDTH - 332, 12, 320, 190))
        x0 = WIDTH - 320
        self.text(f"Очки: {sim.score}", self.font_l, WHITE, (x0, 18))
        self.text(f"Рекорд: {max(self.highscore, sim.score)}   Ур. {sim.level}   Доставок: {sim.deliveries}",
                  self.font_s, DIM, (x0, 56), shadow=False)
        stage = "Забрать на" if not m.picked_up else "Доставить на"
        self.text(f"{stage}: {m.target.name}", self.font_m, PAD_COLORS[m.target.kind], (x0, 80))
        fragile = "  (хрупкое!)" if m.fragile else ""
        self.text(f"Груз: {m.cargo_name}, {m.cargo_mass:.1f} кг{fragile}", self.font_s, WHITE, (x0, 106),
                  shadow=False)
        tl = max(0.0, m.time_left)
        t_col = WHITE if tl > 30 else WARN if tl > 10 else BAD
        self.text(f"Время: {int(tl) // 60}:{int(tl) % 60:02d}", self.font_m, t_col, (x0, 126))
        dist = m.target.x - d.x
        arrow = ">>" if dist > 0 else "<<"
        self.text(f"До цели: {abs(dist):4.0f} м {arrow}", self.font_s, WHITE, (x0 + 140, 130), shadow=False)
        if m.picked_up:
            self.bar(x0, 154, 296, 16, m.cargo_health / 100, GOOD if m.cargo_health > 60 else WARN,
                     f"Целость груза {m.cargo_health:3.0f}%")
        self.text(f"Награда: ~{m.reward}" + ("  (x0.5 автопилот)" if m.used_autopilot else ""),
                  self.font_s, DIM, (x0, 176), shadow=False)

        # Ветер
        wind_x, _ = sim.wind.at(sim.time, d.y)
        wx, wy = WIDTH / 2, 40
        self.panel(pygame.Rect(wx - 90, 12, 180, 56))
        self.text(f"Ветер {abs(wind_x):4.1f} м/с", self.font_s, WHITE, (wx, 26), center=True, shadow=False)
        ln = max(6, min(70, abs(wind_x) * 8))
        sgn = 1 if wind_x >= 0 else -1
        col = GOOD if abs(wind_x) < 4 else WARN if abs(wind_x) < 8 else BAD
        pygame.draw.line(self.screen, col, (wx - sgn * ln, 50), (wx + sgn * ln, 50), 4)
        pygame.draw.polygon(self.screen, col, [(wx + sgn * (ln + 10), 50), (wx + sgn * ln, 43),
                                               (wx + sgn * ln, 57)])

        # Стрелка к цели за краем экрана
        target = sim.autopilot_target()
        tx, ty = self.camera.to_screen(target.x, target.y)
        if not (0 < tx < WIDTH and 0 < ty < HEIGHT):
            cx, cy = WIDTH / 2, HEIGHT / 2
            ang = math.atan2(ty - cy, tx - cx)
            r = min(WIDTH, HEIGHT) / 2 - 60
            ax, ay = cx + math.cos(ang) * r, cy + math.sin(ang) * r
            pts = [(ax + math.cos(ang) * 18, ay + math.sin(ang) * 18),
                   (ax + math.cos(ang + 2.5) * 12, ay + math.sin(ang + 2.5) * 12),
                   (ax + math.cos(ang - 2.5) * 12, ay + math.sin(ang - 2.5) * 12)]
            pygame.draw.polygon(self.screen, WARN, pts)

        self.text("H — подсказка   T — автопилот   M — ручной режим   P — пауза", self.font_s, DIM,
                  (WIDTH / 2, HEIGHT - 70), center=True)

    def draw_minimap(self) -> None:
        sim = self.sim
        rect = pygame.Rect(12, HEIGHT - 52, WIDTH - 24, 40)
        self.panel(rect)
        sx = (rect.w - 16) / cfg.WORLD_WIDTH
        sy = (rect.h - 8) / cfg.CEILING
        base_y = rect.bottom - 4

        def mp(x, y):
            return rect.x + 8 + x * sx, base_y - y * sy

        for z in sim.no_fly:
            x1, y1 = mp(z.x1, min(z.y2, cfg.CEILING))
            x2, y2 = mp(z.x2, z.y1)
            pygame.draw.rect(self.screen, (120, 40, 40), (x1, y1, x2 - x1, y2 - y1))
        for b in sim.buildings:
            x1, y1 = mp(b.x, b.height)
            pygame.draw.rect(self.screen, (80, 90, 120), (x1, y1, max(1, b.width * sx), base_y - y1))
        target = sim.autopilot_target()
        for p in sim.pads:
            px, py = mp(p.x, p.y)
            r = 4 if p is target else 2
            pygame.draw.circle(self.screen, PAD_COLORS[p.kind], (int(px), int(py)), r)
        dx, dy = mp(sim.drone.x, min(sim.drone.y, cfg.CEILING))
        pygame.draw.circle(self.screen, WHITE, (int(dx), int(dy)), 3)
        # Рамка видимой области
        half = WIDTH / (2 * PPM)
        vx1, _ = mp(self.camera.x - half, 0)
        vx2, _ = mp(self.camera.x + half, 0)
        pygame.draw.rect(self.screen, (200, 200, 220), (vx1, rect.y + 2, vx2 - vx1, rect.h - 4), 1)

    def draw_events(self) -> None:
        y = 240
        for e in self.sim.events:
            if e.age > 6:
                continue
            self.text(e.text, self.font_m, e.color, (20, y))
            y += 26

    def draw_help_panel(self) -> None:
        lines = [
            "УПРАВЛЕНИЕ",
            "W (стрелка вверх) — вверх,  S (стрелка вниз) — вниз",
            "A / D (стрелки влево/вправо) — наклон",
            "M — переключить РУЧНОЙ режим (W/S меняют газ) и СТАБИЛИЗАЦИЮ",
            "T — автопилот (награда x0.5)",
            "",
            "ПРАВИЛА",
            "Забери груз на складе и доставь клиенту.",
            "Садись мягко: верт. скорость < 2.6 м/с, наклон < 18°.",
            "На базе (H) дрон заряжается и чинится.",
            "Красная зона — бесполётная, там штраф.",
            "Ветер усиливается с высотой и уровнем.",
        ]
        rect = pygame.Rect(WIDTH / 2 - 270, 150, 540, 30 + len(lines) * 24)
        self.panel(rect)
        for i, line in enumerate(lines):
            col = ACCENT if line.isupper() else WHITE
            self.text(line, self.font_s, col, (rect.x + 20, rect.y + 16 + i * 24), shadow=False)

    def draw_overlay(self, title: str, lines: list[str]) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, (0, 0))
        self.text(title, self.font_xl, WHITE, (WIDTH / 2, HEIGHT / 2 - 80), center=True)
        for i, line in enumerate(lines):
            self.text(line, self.font_m, DIM, (WIDTH / 2, HEIGHT / 2 + i * 34), center=True)

    def draw_title(self) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((5, 10, 25, 185))
        self.screen.blit(shade, (0, 0))
        self.text("НЕБЕСНЫЙ КУРЬЕР", self.font_xl, WHITE, (WIDTH / 2, 120), center=True)
        self.text("симулятор управления беспилотником", self.font_m, ACCENT, (WIDTH / 2, 175), center=True)
        lines = [
            "Ты — оператор дрона-доставщика. Забирай грузы со складов и доставляй клиентам.",
            "Следи за батареей, ветром, птицами и бесполётными зонами.",
            "",
            "W/S или стрелки вверх/вниз — набор высоты / снижение",
            "A/D или стрелки влево/вправо — наклон и движение вбок",
            "M — ручной режим газа (для профи)      T — автопилот (награда x0.5)",
            "H — подсказка      P — пауза",
            "",
            "Мягкая посадка: вертикальная скорость < 2.6 м/с, наклон < 18°",
            "На базе H дрон заряжается и ремонтируется, таймер миссии стоит",
        ]
        for i, line in enumerate(lines):
            self.text(line, self.font_m, WHITE, (WIDTH / 2, 240 + i * 32), center=True)
        if int(pygame.time.get_ticks() / 500) % 2 == 0:
            self.text("Нажми ENTER, чтобы взлететь", self.font_l, WARN, (WIDTH / 2, 600), center=True)
        if self.highscore:
            self.text(f"Рекорд: {self.highscore}", self.font_m, DIM, (WIDTH / 2, 650), center=True)

    def draw_game_over(self) -> None:
        sim = self.sim
        reason = "Дрон разбился" if sim.state is GameState.CRASHED else "Батарея разряжена вдали от базы"
        new_record = sim.score >= self.highscore and sim.score > 0
        lines = [
            reason,
            f"Очки: {sim.score}    Доставок: {sim.deliveries}    Уровень: {sim.level}",
            "НОВЫЙ РЕКОРД!" if new_record else f"Рекорд: {self.highscore}",
            "",
            "R / Enter — новая игра      Esc — выход",
        ]
        self.draw_overlay("МИССИЯ ПРОВАЛЕНА", lines)
