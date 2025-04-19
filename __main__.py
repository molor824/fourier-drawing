import pygame as pg
from fourier import FourierSeries

WIDTH = 800
HEIGHT = 600
MAX_N = 1000
TRAIL_FADE_RATE = 1.0 / 8.0
INTERVAL_SPEED = 1.0
FPS = 60

def complex_to_tuple(c: complex):
    return c.real, c.imag

def main():
    pg.init()
    pg.display.set_caption("Fourier Drawing")
    pg.display.set_mode((WIDTH, HEIGHT))

    clock = pg.time.Clock()

    series: FourierSeries | None = None
    series_time = 0.0

    drawing: list[complex] | None = None
    timeframes: list[float] | None = None
    trails: list[tuple[complex, float]] = []

    center = (WIDTH + HEIGHT * 1j) * 0.5

    while True:
        should_exit = False
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    pg.quit()
                    should_exit = True
                case pg.KEYDOWN:
                    key = event.key
                    if key == pg.K_ESCAPE:
                        should_exit = True
                case pg.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        drawing = []
                        timeframes = []
                        series = None
                case pg.MOUSEBUTTONUP:
                    if event.button == 1:
                        if drawing is not None and len(drawing) > 2:
                            series = FourierSeries.from_points(MAX_N, drawing, timeframes)
                            series_time = 0.0
                            trails = []
                        drawing = None
                        timeframes = None
                case pg.MOUSEMOTION:
                    if drawing is not None:
                        x, y = event.pos
                        drawing.append(complex(x, y) - center)
                        timeframes.append(pg.time.get_ticks() / 1000.0)
        if should_exit:
            break

        arrows = list(series.arrows(series_time)) if series else None
        series_time += INTERVAL_SPEED / FPS

        surface = pg.display.get_surface()
        surface.fill((0, 0, 0))

        trails = list(filter(lambda t: t[1] > 0, ((p, i - TRAIL_FADE_RATE / FPS) for (p, i) in trails)))
        for i in range(1, len(trails)):
            trail0, intensity = trails[i - 1]
            trail1, _ = trails[i]
            intensity = round(intensity * 255)
            pg.draw.line(surface, tuple(intensity for _ in range(3)), complex_to_tuple(trail0 + center), complex_to_tuple(trail1 + center), 2)

        if arrows:
            arrow_sum = 0j
            def draw_arrow(arrow: complex):
                nonlocal arrow_sum
                prev_sum = arrow_sum
                arrow_sum = arrow_sum + arrow
                pg.draw.line(surface, pg.Color("#ffffff"), complex_to_tuple(prev_sum + center), complex_to_tuple(arrow_sum + center), 1)
            for i in range(0, series.max_n + 1):
                draw_arrow(arrows[i + series.max_n])
                if i != 0:
                    draw_arrow(arrows[series.max_n - i])
            trails.append((arrow_sum, 1.0))

        if drawing:
            for i in range(1, len(drawing)):
                start = drawing[i - 1]
                end = drawing[i]
                pg.draw.line(surface, pg.Color("#FFFFFF"), complex_to_tuple(start + center), complex_to_tuple(end + center), 1)

        pg.display.flip()

        clock.tick(FPS)

if __name__ == "__main__":
    main()
