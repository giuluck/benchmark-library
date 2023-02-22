# Simple Epidemics Benchmark

This benchmark provides simple simulation approaches for epidemic phenomena, based on classical compartmental models (SIR and SEIR).
The benchmark can be used for epidemic control problems, for learning surrogates, or any other application where evaluating epidemic (or epidemic-like) behavior under controlled conditions can be useful.

There two main `Benchmark` sub-classes:

* `epidemics.SIR`, representing a compartmental model with the Susceptibles, Infected, and Recovered groups
* `epidemics.SEIR`, representing a compartmental model with the Susceptibles, Exposed, Infected, and Recovered groups

In both cases, classical formulations based on continuous variables and differential equations are employed.

## SIR Model

In the case of the SIR model, the epidemics behavior is defined via the following Ordinary Differential Equation:
\begin{align}
\frac{dS}{dt} & = - \beta S I \\
\frac{dI}{dt} & = \beta S I - \gamma I\\
\frac{dR}{dt} & = \gamma I
\end{align}
where $S$, $I$, $R$ represent the amount of population that are respectively susceptible, infected, and have recovered from the epidemic agent.
The benchmark assumes that the population is normalized, i.e. the code explicitly checks whether $S + I + R = 1$.
The parameter $\beta$ represents the infection rate and $\gamma$ is the recovery rate.

Given an initial state $S_0, I_0, R_0$ standard methods can be used to integrate the differential equations and obtain the curve for the three population component over a chosen number of time units. It can be Thought of as running a simulation for a fixed time horizon. Integration is performed via the `scipy.integrate.odeint` method.

The `epidemics.SIR` class allows to perform repeated integrations, with different initial states and different values for $\beta$ and $\gamma$. By doing so, it is possible to simulate the epidemic behavior under varying conditions, including social distancing measures (which act by changing the value of $\beta$) or new therapies (which act by changing $\gamma$).

Formally, the class defines five variables:

* `s0`, representing $S_0$
* `i0`, representing $I_0$
* `r0`, representing $R_0$
* `beta`, representing $\beta$
* `gamma`, representing $\gamma$

There is a single contra int on the input variables, i.e.:

$$S_0 + I_0 + R_0 = 1$$

There is a single parameter:

* `horizon`, representing the number of time units to use for integration

There are three basic metrics, corresponding to the value of each component over time:

* `susceptibles`
* `infected`
* `recovered`

A simple usage example would be:

```python
TO BE ADDED
```

## SEIR Model

A SEIR model extends a SIR model by adding an additional compartment, corresponding to individuals that have been exposed to the epidemic agent, but are not yet showing symptoms and are not yet capable of infecting other individuals. The corresponding Ordinary Differential Equation is:
\begin{align}
\frac{dS}{dt} & = - \beta S I \\
\frac{dE}{dt} & = \beta S I - \alpha I\\
\frac{dI}{dt} & = \alpha E - \gamma E\\
\frac{dR}{dt} & = \gamma I
\end{align}
where $\alpha$ represents the rate at which the infection becomes active.

Accordingly, the `epidemics.SEIR` class has two additional variables:

* `e0`, representing the compoent $E_0$ of the initial state
* `latency`, corresponding to $\alpha$

The constraint on the input variables is also modified accordingly:

$$S_0 + E_0 + I_0 + R_0 = 1$$
