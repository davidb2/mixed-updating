import numpy as np
import pandas as pd
import random
import networkx as nx
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Optional

from moran import simulate, SimulationResult

@dataclass(frozen=True)
class MoreContextResult(SimulationResult):
  r: Optional[float] = None
  delta: Optional[float] = None
  leaf: Optional[bool] = None

def simulate_wrapper(trial_number, G, r, delta, leaf):
  # 0 is the center.
  result = simulate(trial_number, G, r, 1-delta, initial_mutant_location=int(leaf))
  return MoreContextResult(
    trial_number=trial_number,
    r=r,
    delta=delta,
    fixed=result.fixed,
    steps=result.steps,
    leaf=leaf,
  )


def main():
  N = 10
  G = nx.star_graph(N-1)
  TRIALS = 10000

  with Pool() as pool:
    results = pool.starmap(
      simulate_wrapper,
      (
        (trial_number, G, r, 1-delta, leaf)
        for trial_number in range(TRIALS) 
        # for delta in np.linspace(0, 1, 100+1)
        for delta in (0, .25, .5, .75, 1)
        for r in np.arange(0.0, 2.0+0.01, 0.01)
        for leaf in (False, True)
      )
    )

  df = pd.DataFrame(results)
  df.to_csv('./data/star-10-results-martin.csv')
  print(df)


if __name__ == '__main__':
  main()