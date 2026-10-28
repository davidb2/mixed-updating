# Code for "Mixed birth-death and death-birth updating in structured populations"

The code requires the following packages:
* `bitarray`
* `matplotlib`
* `networkx`
* `numpy`
* `pandas`
* `scipy`
* `seaborn`
* `tqdm`

To generate all graphs of a certain size, the `geng` program fom the [`nauty`](https://pallini.di.uniroma1.it/) suite was used.
## Optional exact multipartite oracle

[extras/multipartite-order](extras/multipartite-order/README.md) provides an independently written rational count-chain calculator and a proposed stochastic-order proof note for complete multipartite populations. It includes exact fine/count cross-checks and a standalone CLI. The note gives a native-generator criterion and sufficient-class fixation comparison; it does not resolve general-graph fixation monotonicity. Source-author validation has not been obtained.
