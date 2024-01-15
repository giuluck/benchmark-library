import pandas as pd
from ppsim.plant.execution import SimulationOutput


def demands_gap(output: SimulationOutput) -> pd.DataFrame:
    """Computes the gap on demands across the time horizon. Gaps are clipped to zero (i.e., if a higher quantity of
    commodity is supplied, the gap is null).

    :param output:
        The output of a Unit Commitment query.

    :return:
        A dataframe of clients indexed by timestamp, where each cell indicates the gap in the demand at that timestamp.
    """
    # retrieve data
    flows = output.flows.copy(deep=True)
    demands = output.demands.copy(deep=True)
    # for each edge whose destination is a client (i.e., it is in the demands dataframe) remove the flow quantity
    for edge, flow in flows.items():
        # noinspection PyUnresolvedReferences
        client = edge.destination
        demand = demands.get(client)
        if demand is not None:
            demand -= flow
    # the remainders are then clipped to zero since negative values indicate that more commodity was supplied
    return demands.clip(0)


def costs(output: SimulationOutput) -> pd.DataFrame:
    """Computes the cost of each node (supplier or machine) at each timestamp.

    :param output:
        The output of a Unit Commitment query.

    :return:
        A dataframe of nodes indexed by timestamp, where each cell indicates its cost at that timestamp.
    """
    # retrieve data and build an empty dataframe of costs
    flows = output.flows.copy(deep=True)
    states = output.states.copy(deep=True)
    prices = output.buying_prices.copy(deep=True)
    cost = pd.DataFrame(data=0.0, columns=[*prices.keys(), *states.keys()], index=flows.index)
    # for each edge whose source is a supplier (i.e., it is in the prices dataframe) compute the cost as price * flow
    for edge, flow in flows.items():
        # noinspection PyUnresolvedReferences
        supplier = edge.source
        price = prices.get(supplier)
        if price is not None:
            cost[supplier] += price * flow
    # for each machine (which is in the states dataframe) compute the cost from the respective method
    for machine, state in states.items():
        # noinspection PyUnresolvedReferences
        cost[machine] += state.map(lambda s: machine.cost(state=s))
    return cost
