import os
import json
import numpy as np
import pandas as pd

instance_name = 'solver_343747_328556_324836'

folder_path = os.path.join('.', 'data', f'{instance_name}')

f = open(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/data/dummy_problem/weights.json'), 'r')
weights = json.load(f)
f.close()

df = pd.read_csv(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/data/dummy_problem/service.csv'), sep = ',', header = None)
service = df.values

df = pd.read_csv(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/data/dummy_problem/distances.csv'), sep = ',', header = None)
distances = df.values

folder_path = os.path.join('.', 'results', f'{instance_name}')

df = pd.read_csv(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/results/dummy_problem/deposit_locations.csv'), sep = ',', header = None)
deposit_locations = df.values

df = pd.read_csv(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/results/dummy_problem/path.csv'), sep = ',', header = None)
path = df.values

(N_deposits, N_supermarkets) = service.shape

N_constructions = np.sum(deposit_locations)

N_missed_supermarkets = N_supermarkets - len(np.nonzero(np.matmul(deposit_locations, service))[0])

travel_length = np.sum(distances * path)

total_cost = N_constructions * weights['construction'] + N_missed_supermarkets * weights['missed_supermarket'] + travel_length * weights['travel']

print('\n-----------------------COSTS-----------------------')
print('\t\t\t\tQ.TY\t\tCOST')
print(f"DEPOSIT CONSTRUCTIONS\t\t{N_constructions}\tx\t{weights['construction']}")
print(f"MISSED SUPERMARKETS\t\t{N_missed_supermarkets}\tx\t{weights['missed_supermarket']}")
print(f"TRAVEL LENGTH\t\t\t{travel_length}\tx\t{weights['travel']}")
print('')
print(f'TOTAL\t\t\t\t{total_cost}')
print('---------------------------------------------------\n')