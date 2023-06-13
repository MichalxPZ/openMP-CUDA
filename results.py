import pandas as pd

results = [
    (1000, "Serial", 848538),
    (1000, "OPENMP", 335997),
    (2000, "Serial", 6824267),
    (2000, "OPENMP", 3626471),
    (3000, "Serial", 24332369),
    (3000, "OPENMP", 14290351),
    (4000, "Serial", 55994073),
    (4000, "OPENMP", 33425967),
    (5000, "Serial", 110926232),
    (5000, "OPENMP", 67367441),
    (6000, "Serial", 183704272),
    (6000, "OPENMP", 121806870),
    (7000, "Serial", 328801811),
    (7000, "OPENMP", 179628613),
    (8000, "Serial", 443322364),
    (8000, "OPENMP", 269045048),
    (9000, "Serial", 625262705),
    (9000, "OPENMP", 359967367),
]

df = pd.DataFrame(results, columns=["Size", "Type", "Time (microseconds)"])
df_pivot = df.pivot(index="Size", columns="Type", values="Time (microseconds)")

print(df_pivot)