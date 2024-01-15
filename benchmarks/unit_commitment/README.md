# Unit Commitment Benchmark

## Demands Nodes

### Model

Demand nodes are associated to:

* Predictions
* Actual values
* Decisions (the commodity flows)

Demand predictions are an input to the system. The demand for time step $t$ and commodity $k$ is referred here as $\hat{d}_{t,k}$. Together, they for a set of sequences:

$$\{\hat{d}_{t,k}\}_{t=1}^{eoh}$$

Demands (and therefore their predictions) depend mostly on time, and in particular on:

* Daily cycle
* Weekly cycle
* Season


Actual demands are in fact stochastic, with limited variability around the predictions. This is modeled in the benchmarks via a 0-centered Gaussian noise. Variance for such noise is likely also time- and commodity-dependent, so that overall we have:

$$\{d_{t,k}\}_{t=1}^{eoh} \quad \text{with} d_{t,k} \sim D_{t,k}$$

Decisions for demand nodes correspond to the flows toward the node, for every commodity and time instant. Their values must be supplied by any problem solution. They are referred to as:

$$\{x_{t,k}\}_{t=1}^{eoh}$$


### Generation

Generating demand predictions is needed to build problem instances. In the benchmark we provide a default generator based on a neuro-probabilistic architecture conditioned over temporal features and on the previous predictions.

$$\hat{d}_{t,k} \sim f(\hat{d} \mid x)$$

Where $x$ contains:

* Time of the day
* Day of the week
* Season
* All predictions up to time $t$
* The considered commodity

The density estimator is trained over real data.

Actual predictions can then be obtained by simply adding Gaussian noise, based on a variance model passed as input.

In our _default implementation_, we assume the actual demands are obtained by adding Gaussian noise centered on the predictions and with variance provided by a regression model:

$$d_{t,k} = \hat{d}_{t,k} + \varepsilon_{t,k}, \quad \varepsilon_{t,k} \sim \mathcal{N}(0, \sigma(x))$$

Where $x$ contains:

* Time of the day
* Day of the week
* Season
* All demands value up to time $t-1$
* All predicted demand values up to time $t$
* The considered commodity

The same approach is used for the buying and selling prices.

### Constraints and Metrics

Decisions may be subject to bound constraints, i.e.:

$$l \leq x_{t,k} \leq u$$

The commodity flows should satisfy the demands:

$$x_{t,k} \geq d_{t,k} $$


## Market Nodes

Market nodes are associated to:

* Buying prices
* Selling prices
* Decisions (corresponding to commodity flows)

Prices (both for buying and selling) change over time and depend on the considered commodity. At planning time one has access to prediction that we models as sequences. We refer to the buying price predictions as:

$$\{\hat{b}_{t,k} \}_{t=1}^{eoh}$$

And to the selling price predictions as:

$$\{\hat{s}_{t,k} \}_{t=1}^{eoh}$$


The same is true for the  buying and selling prices. For the buying prices we have:

$$\{b_{t,k} \}_{t=1}^{eoh}$$

And for the selling prices:

$$\{s_{t,k} \}_{t=1}^{eoh}$$

Decisions for market nodes correspond to the flows from or toward the node (depending on their sign), for every commodity and time instant. Their values must be supplied by any problem solution. They are referred to as:

$$\{x_{t,k}\}_{t=1}^{eoh}$$

### Generation

Generation for market nodes follows the same approach used for the demands.

### Constraints and Metrics

Decisions may be subject to bound constraints, i.e.:

$$l \leq x_{t,k} \leq u$$

Decisions are associated to a cost or profit. For selling a commodity we have:

$$\max(0, x_{t,k}) s_{t,k}$$

While for buying we have:

$$\max(0, -x_{t,k}) b_{t,k}$$


## Machines

The system contains multiple machines for generating and storing energy

* Operational cost
* Setpoint
* State
* Input (indexed over time, commodity)
* Output (indexed over time, commodity)


### Constraints and Metrics

For generation machines:


## Routing Nodes

