#!/usr/bin/env python3.11
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from collections import deque
from multiprocessing import Pool
from pathlib import Path
from utils import yield_all_graph6, istarmap
from moran import (
  mixed_moran_fix_prob_sys,
  mixed_moran_abs_time_sys,
  mixed_moran_cond_fix_time_sys,
  vec_2_idx,
)


def process(G: nx.Graph, r: float, p_Bd: float, initial_mutant_location: int):
  N = len(G)
  fix_prob = mixed_moran_fix_prob_sys(G, r=r, p_Bd=p_Bd)
  fix_time = mixed_moran_cond_fix_time_sys(G, r=r, p_Bd=p_Bd)
  S = deque([1] + [0]*(N-1))
  for i in range(initial_mutant_location):
    S.rotate(1)
  
  return {
      'N': N,
      'graph': nx.graph6.to_graph6_bytes(G, header=False).hex(),
      'p_Bd': p_Bd,
      'r': r,
      'fix_prob': fix_prob[vec_2_idx(S)],
      'fix_time': fix_time[vec_2_idx(S)],
      'initial_mutant_location': initial_mutant_location,
  }


def main():
  # Constants.
  N = 7
  Ls = np.linspace(0, 1, 100+1)
  Rs = [1] #np.linspace(.25, 1.75, 10+1)

  initial_mutant_locs_times = [4,2,5,0]
  initial_mutant_locs_probs = [0,6,0]
  Gs_hex_times = [
    '464568655f0a',
    '464568754f0a',
    '46456e626f0a',
    '463f3f46770a',
  ]
  Gs_hex_probs = [
    '463f4246770a',
    '463f3f46770a',
    '463f3f46770a',
  ]
  Gs = [
    nx.graph6.from_graph6_bytes(bytes.fromhex(hex_str).strip())
    for hex_str in Gs_hex_probs
  ]

  # path = Path(f'./data/connected-n{N}.g6')
  # with path.open("rb") as f:
  #   num_graphs = sum(1 for _ in f)
  num_graphs = len(Gs)

  data = []
  with Pool() as pool:
    for result in tqdm.tqdm(pool.istarmap(
        process,
        (
          (G, r, p_Bd, initial_mutant_location)
          # for G in yield_all_graph6(Path(f'./data/connected-n{N}.g6'))
          for G, initial_mutant_location in zip(Gs, initial_mutant_locs_probs)
          for r in Rs
          for p_Bd in Ls
        )
      ),
      total=num_graphs*len(Rs)*len(Ls)*N,
    ):
      data.append(result)

  df = pd.DataFrame(data)
  df.to_csv(f'./data/weird-examples-fp-{N}.csv', index=False)
  # df.to_csv(f'./data/connected-n{N}-martin-results.csv')

if __name__ == '__main__':
  main()