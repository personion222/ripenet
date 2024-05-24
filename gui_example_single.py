import tensorflow as tf
import numpy as np
import pygame
import keras
import cv2


def gradient(surface, top, bottom, drawrect):
    colour_rect = pygame.Surface((1, 2)).convert_alpha()
    pygame.draw.line(colour_rect, top, (0, 0), (0, 0))
    pygame.draw.line(colour_rect, bottom, (0, 1), (0, 1))
    colour_rect = pygame.transform.smoothscale(colour_rect, (drawrect[2], drawrect[3]))
    surface.blit(colour_rect, (drawrect[0], drawrect[1]))


FPS = 30
WIDTH = 1280
HEIGHT = 720

pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RipeNet")

font64 = pygame.font.Font("font.ttf", 64)
font48 = pygame.font.Font("font.ttf", 48)
font32 = pygame.font.Font("font.ttf", 32)

paused_surf = font48.render("paused", True, (255, 255, 255))

gui = pygame.surface.Surface((WIDTH, HEIGHT)).convert_alpha()
gui.fill((0, 0, 0, 0))

gradient(gui, (0, 0, 0, 0), (0, 0, 0, 255), (0, HEIGHT * 0.8, WIDTH, HEIGHT * 0.2))
gradient(gui, (0, 0, 0, 255), (0, 0, 0, 0), (0, 0, WIDTH, HEIGHT * 0.2))

ripe_thresh = 0.5

square_width = min((WIDTH, HEIGHT))

clock = pygame.time.Clock()
running = True
paused = False

screen_ar = WIDTH / HEIGHT

draw_coord = [0, 0]

ripe_model = keras.saving.load_model("single_nets/model2.keras", compile=True)
ripe_model.summary()

vid_cap = cv2.VideoCapture(0)
vid_cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

while running:
    if not paused:
        frame = vid_cap.read()[1][:, :, ::-1]

    height, width, _ = frame.shape

    square_coords = (
        (width // 2 - square_width // 2, width // 2 + square_width // 2),
        (height // 2 - square_width // 2, height // 2 + square_width // 2)
    )

    frame_mod = frame[
        square_coords[1][0]: square_coords[1][1],
        square_coords[0][0]: square_coords[0][1]
    ]

    s_scale_fac = square_width / 64

    frame_mod = (cv2.resize(frame_mod, (64, 64)) / 255).reshape((1, 64, 64, 3))

    ripeness = ripe_model.predict(frame_mod, verbose=None)[0, 0]

    print(ripeness)

    screen.fill((0, 0, 0))

    frame_ar = width / height

    if frame_ar < screen_ar:
        draw_coord[0] = (WIDTH - width) // 4
        scale_fac = HEIGHT / height
        frame_resize = cv2.resize(frame, (0, 0), fx=scale_fac, fy=scale_fac)

    else:
        draw_coord[1] = (HEIGHT - height) // 4
        scale_fac = WIDTH / width
        frame_resize = cv2.resize(frame, (0, 0), fx=scale_fac, fy=scale_fac)

    surf_frame = pygame.image.frombuffer(frame_resize[:, :, ::-1].tobytes(), frame_resize.shape[1::-1], "BGR")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

            if event.key == pygame.K_RIGHT:
                ripe_thresh += 0.1

            if event.key == pygame.K_LEFT:
                ripe_thresh -= 0.1

            ripe_thresh = max((0, min((ripe_thresh, 1))))

    if ripeness > ripe_thresh:
        ripe_txt = "ripe"

    else:
        ripe_txt = "unripe"

    ripe_surf = font48.render(ripe_txt, True, (255, 255, 255))

    ripe_thresh_surf = font48.render(str(round(ripe_thresh, 1) + 0.01)[:3], True, (255, 255, 255))

    screen.blit(surf_frame, draw_coord)
    screen.blit(gui, (0, 0))
    screen.blit(ripe_surf, (24, 16))
    screen.blit(ripe_thresh_surf, (
        WIDTH - ripe_thresh_surf.get_width() - 24,
        16
    ))

    if paused:
        screen.blit(paused_surf, (
            WIDTH // 2 - paused_surf.get_width() // 2,
            HEIGHT * 0.8
        ))

    pygame.display.update()

pygame.quit()
vid_cap.release()
cv2.destroyAllWindows()
