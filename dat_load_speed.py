from load_dat import load_dat
from statistics import median

times = load_dat("mod_fruit/mod_annotated", 0.1)[1]

medians = dict.fromkeys(times.keys(), 0)

for key in times.keys():
    medians[key] = median(times[key]) * 1000

print(medians)
