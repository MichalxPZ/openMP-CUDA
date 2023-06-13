import pandas as pd

data = {
    'Size': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500],
    'CUDA': [102604, 3246, 10168, 23830, 45289, 112962, 166959, 158138, 223841, 307171, 407747, 530189, 672121, 839438, 1032150, 1252047, 1501122, 1781083, 2093780, 2440467],
    'Serial': [1006, 1010, 3001, 4009, 6001, 8994, 11989, 16002, 19996, 25991, 32996, 39999, 49000, 56998, 68001, 78987, 114004, 1006, 1006, 1006],
    'OPENMP': [27000, 24999, 27990, 27993, 26991, 21003, 25001, 28006, 32991, 36999, 64998, 65000, 73982, 80000, 100977, 109002, 90014, 100977, 109002, 90014]
}

df = pd.DataFrame(data)
df['Serial'] = df['Serial'] * df['Size']
df['OPENMP'] = df['OPENMP'] * df['Size']

print(df)

import matplotlib.pyplot as plt

# Utwórz wykres
plt.plot(df['Size'], df['CUDA'], marker='o', label='CUDA')
plt.plot(df['Size'], df['Serial'], marker='o', label='Serial')
plt.plot(df['Size'], df['OPENMP'], marker='o', label='OPENMP')

# Dodaj tytuł i etykiety osi
plt.title('Execution Time Comparison')
plt.xlabel('Size')
plt.ylabel('Execution Time (microseconds)')

# Dodaj legendę
plt.legend()

# Wyświetl wykres
plt.show()

df['OpenMP Acceleration'] = df['Serial'] / df['OPENMP']

# Obliczenie przyspieszenia dla CUDA
df['CUDA Acceleration'] = df['Serial'] / df['CUDA']

# Wykres przyspieszenia dla OpenMP
plt.plot(df['Size'], df['OpenMP Acceleration'], label='OpenMP')
plt.xlabel('Size')
plt.ylabel('Acceleration')
plt.title('OpenMP Acceleration')
plt.legend()
plt.show()

# Wykres przyspieszenia dla CUDA
plt.plot(df['Size'], df['CUDA Acceleration'], label='CUDA')
plt.xlabel('Size')
plt.ylabel('Acceleration')
plt.title('CUDA Acceleration')
plt.legend()
plt.show()