from daltonlens import simulate
import cv2

vid = "s1vid.mp4"
save = "s1tritan.mp4"
deficiency = simulate.Deficiency.TRITAN
severity = 1.0
framemax = 0

sim = simulate.Simulator_AutoSelect()

read = cv2.VideoCapture(vid)
dims = (int(read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = round(read.get(cv2.CAP_PROP_FPS))
print(dims, fps)
write = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, dims)

while True:
    ret, frame = read.read()
    if not ret:
        break

    if framemax:
        if read.get(cv2.CAP_PROP_POS_FRAMES) >= framemax:
            break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framergb_cvd = sim.simulate_cvd(framergb, deficiency, severity)
    frame_cvd = cv2.cvtColor(framergb_cvd, cv2.COLOR_RGB2BGR)
    write.write(frame_cvd)

    cv2.imshow("progress", frame_cvd)
    if cv2.waitKey(1) == ord('q'):
        break

read.release()
write.release()
cv2.destroyAllWindows()
