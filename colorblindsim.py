from daltonlens import simulate
import pygame
import cv2

TXTCOL = (241, 196, 255)

pygame.font.init()

font = pygame.font.Font("font.ttf", 32)

colorblind_types = (
    None,
    simulate.Deficiency.DEUTAN,
    simulate.Deficiency.TRITAN,
    simulate.Deficiency.PROTAN
)

cb_idx = 0
severity = 1.0

vid_cap = cv2.VideoCapture(0)
sim = simulate.Simulator_AutoSelect()


def init():
    global vid_cap
    vid_cap = cv2.VideoCapture(0)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


def deinit():
    global vid_cap
    vid_cap.release()


def draw(surf, events, dims):
    global vid_cap
    global cb_idx
    global severity
    screendims = (surf.get_width(), surf.get_height())
    frame = cv2.cvtColor(cv2.rotate(vid_cap.read()[1], cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)

    if colorblind_types[cb_idx] is not None:
        frame = sim.simulate_cvd(frame, colorblind_types[cb_idx], severity=severity)

    if dims[0] / dims[1] > frame.shape[0] / frame.shape[1]:
        scale = dims[1] / frame.shape[1]

    else:
        scale = dims[0] / frame.shape[0]

    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    framesurf = pygame.surfarray.make_surface(frame)
    surf_dims = frame.shape[:2]
    surf.blit(framesurf, (
            screendims[0] // 2 - surf_dims[0] // 2,
            screendims[1] // 2 - surf_dims[1] // 2,
    ))

    for event in events:
        if event.type == pygame.KEYUP:
            cb_idx += int(event.key == pygame.K_d)
            cb_idx -= int(event.key == pygame.K_a)
            cb_idx %= len(colorblind_types)

            severity += float(event.key == pygame.K_w) * 0.2
            severity -= float(event.key == pygame.K_s) * 0.2
            severity = max((0, min((severity, 1))))

    if colorblind_types[cb_idx] is None:
        cb_str = "normal color vision"

    elif colorblind_types[cb_idx] == simulate.Deficiency.DEUTAN:
        cb_str = "deutera"
        if severity >= 1:
            cb_str += "nopia (100%)"

        else:
            cb_str += f"nomaly ({round(severity * 100)}%)"

    elif colorblind_types[cb_idx] == simulate.Deficiency.TRITAN:
        cb_str = "trita"
        if severity >= 1:
            cb_str += "nopia (100%)"

        else:
            cb_str += f"nomaly ({round(severity * 100)}%)"

    else:
        cb_str = "prota"
        if severity >= 1:
            cb_str += "nopia (100%)"

        else:
            cb_str += f"nomaly ({round(severity * 100)}%)"

    cb_txt = font.render(cb_str, True, TXTCOL)
    surf.blit(cb_txt, (15, 15))
