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

def simulate_wrapper(trial_number, G, r, delta):
  result = simulate(trial_number, G, r, 1-delta)
  return MoreContextResult(
    trial_number=trial_number,
    r=r,
    delta=delta,
    fixed=result.fixed,
    steps=result.steps,
  )


def main():
  N = 10
  G = nx.cycle_graph(N)
  TRIALS = 10000

  with Pool() as pool:
    results = pool.starmap(
      simulate_wrapper,
      (
        (trial_number, G, r, 1-delta)
        for trial_number in range(TRIALS) 
        # for r in np.linspace(1, 1, 20+1)
        for r in np.arange(0.1, 2.1, 0.01)
        for delta in (0, .25, .5, .75, 1)
        # for r in (1, 1.1, 1.5)
        # for delta in np.linspace(0, 1, 20+1)
      )
    )

  df = pd.DataFrame(results)
  df.to_csv('./data/cycle-10-results-delta-examples.csv')
  print(df)


if __name__ == '__main__':
  main()