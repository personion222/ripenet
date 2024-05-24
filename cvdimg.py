from daltonlens import simulate
import cv2

img = "unripeb.png"
save = "unripeb-deutan.png"
deficiency = simulate.Deficiency.DEUTAN
severity = 1.0

sim = simulate.Simulator_AutoSelect()

src = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
colorblind = sim.simulate_cvd(src, deficiency, severity)
cv2.imwrite(save, cv2.cvtColor(colorblind, cv2.COLOR_RGB2BGR))
