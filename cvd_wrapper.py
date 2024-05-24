from screeninfo import get_monitors
import colorblindsim
import pygame

monitor = get_monitors()[0]

WIDTH = monitor.width
HEIGHT = monitor.height
FPS = 60

BACKGROUND = pygame.Color(16, 12, 16)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.FULLSCREEN)
clock = pygame.time.Clock()

title = pygame.font.Font("font.ttf", 96)
sub = pygame.font.Font("font.ttf", 48)

running = True
while running:
    clock.tick(FPS)

    handoff_events = list()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False

        handoff_events.append(event)

    screen.fill(BACKGROUND)

    colorblindsim.draw(
        screen,
        handoff_events,
        (WIDTH, HEIGHT)
    )

    pygame.display.update()

pygame.quit()
