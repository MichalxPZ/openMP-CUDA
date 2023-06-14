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


data = {
    'Vertices': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
    'Serial': [848538, 6824267, 24332369, 55994073, 110926232, 183704272, 328801811, 443322364, 625262705],
    'OPENMP': [335997, 3626471, 14290351, 33425967, 67367441, 121806870, 179628613, 269045048, 359967367]
}

df = pd.DataFrame(data)
df['Serial'] /= 1000  # Convert to milliseconds
df['OPENMP'] /= 1000  # Convert to milliseconds

plt.figure(figsize=(10, 6))
plt.plot(df['Vertices'], df['Serial'], marker='o', label='Serial')
plt.plot(df['Vertices'], df['OPENMP'], marker='o', label='OPENMP')

plt.title('Execution Time Comparison: Serial vs OPENMP')
plt.xlabel('Vertices')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.show()

serial_times = df['Serial']
df['Speedup'] = serial_times / df['OPENMP']

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