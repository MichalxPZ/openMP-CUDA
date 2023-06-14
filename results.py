import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Vertices': [100, 200, 300, 400, 500, 600, 700, 800, 900],
    'Serial': [68212, 1052945, 5241891, 18512200, 42976821, 90724379, 167186474, 285444734, 456163822],
    'CUDA': [216134, 294669, 691272, 1401878, 2597696, 4381982, 6863220, 10118992, 14345267]
}

df = pd.DataFrame(data)
df['Serial'] /= 1000  # Convert to milliseconds
df['CUDA'] /= 1000  # Convert to milliseconds

plt.figure(figsize=(10, 6))
plt.plot(df['Vertices'], df['Serial'], marker='o', label='Serial')
plt.plot(df['Vertices'], df['CUDA'], marker='o', label='CUDA')

plt.title('Execution Time Comparison: Serial vs CUDA')
plt.xlabel('Vertices')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.show()

serial_times = df['Serial']
df['Speedup'] = serial_times / df['CUDA']

# Print DataFrame
print(df)

# Plot speedup
plt.figure(figsize=(10, 6))
plt.plot(df['Vertices'], df['Speedup'], marker='o')

plt.title('Speedup Comparison: Serial vs CUDA')
plt.xlabel('Vertices')
plt.ylabel('Speedup')
plt.grid(True)
plt.show()


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
    (9000, "OPENMP", 359967367)
]

df = pd.DataFrame(results, columns=['Vertices', 'Method', 'Time'])
df['Time'] /= 1000  # Convert to milliseconds

plt.figure(figsize=(10, 6))
for method in df['Method'].unique():
    df_method = df[df['Method'] == method]
    plt.plot(df_method['Vertices'], df_method['Time'], marker='o', label=method)

plt.title('Execution Time Comparison: Serial vs OPENMP')
plt.xlabel('Vertices')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.show()

serial_times = df[df['Method'] == 'Serial']['Time']
df['Speedup'] = serial_times.values / df['Time'].values

# Print DataFrame
print(df)

# Plot speedup
plt.figure(figsize=(10, 6))
plt.plot(df['Vertices'], df['Speedup'], marker='o')

plt.title('Speedup Comparison: Serial vs OPENMP')
plt.xlabel('Vertices')
plt.ylabel('Speedup')
plt.grid(True)
plt.show()