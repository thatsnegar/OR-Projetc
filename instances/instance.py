import os
import json
import pandas as pd

class Instance():
    def __init__(self, folder_name):
        folder_path = os.path.join('.', 'data', f'{folder_name}')
        f = open(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/data/dummy_problem/weights.json'), 'r')
        self.weights = json.load(f)
        f.close()
        df = pd.read_csv(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/data/dummy_problem/distances.csv'), sep = ',', header = None)
        self.service = df.values
        df = pd.read_csv(os.path.join(folder_path, '/Users/thatsnegar/Library/Mobile Documents/com~apple~CloudDocs/operational-research/Progetto-Ricerca-Operativa-2024-2025-main/data/dummy_problem/distances.csv'), sep = ',', header = None)
        self.distances = df.values

if __name__ == '__main__':
    inst = Instance('dummy_problem')
