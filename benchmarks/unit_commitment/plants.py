from typing import Iterable, Union

import numpy as np
from powerplantsim import Plant


def _get_variance(variance: Union[float, Iterable[float]], size: int) -> np.ndarray:
    if isinstance(variance, float):
        variance = np.ones(size) * variance
    else:
        variance = np.array(variance)
        assert len(variance) == size, f"Expected variance vector of size {size}, got {len(variance)}"
    return variance


def simple(
        horizon: Union[float, Iterable[float]],
        heat_demand: Union[float, Iterable[float]],
        gas_purchase: Union[float, Iterable[float]],
        electricity_selling: Union[float, Iterable[float]],
        heat_demand_variance: Union[float, Iterable[float]] = 0.0,
        gas_purchase_variance: Union[float, Iterable[float]] = 0.0,
        electricity_selling_variance: Union[float, Iterable[float]] = 0.0
) -> Plant:
    # noinspection DuplicatedCode
    plant = Plant(horizon=horizon)
    gas_purchase_variance = _get_variance(gas_purchase_variance, size=len(horizon))
    plant.add_extremity(
        kind='supplier',
        name='gas',
        commodity='gas',
        predictions=gas_purchase,
        variance=lambda _, series: gas_purchase_variance[len(series)]
    )
    plant.add_machine(
        name='boiler',
        parents='gas',
        commodity='gas',
        setpoint=[0, 1],
        inputs=[0, 100000],
        outputs={'heat': [0, 90000]}
    )
    plant.add_machine(
        name='chp',
        parents='gas',
        commodity='gas',
        setpoint=[0.5, 1],
        inputs=[67500, 135000],
        outputs={'heat': [27000, 50000], 'elec': [23000, 50000]},
        max_starting=(3, 24)
    )
    heat_demand_variance = _get_variance(heat_demand_variance, size=len(horizon))
    plant.add_extremity(
        kind='customer',
        name='heat',
        commodity='heat',
        parents=['chp', 'boiler'],
        predictions=heat_demand,
        variance=lambda _, series: heat_demand_variance[len(series)]
    )
    electricity_selling_variance = _get_variance(electricity_selling_variance, size=len(horizon))
    plant.add_extremity(
        kind='purchaser',
        name='grid',
        commodity='elec',
        parents='chp',
        predictions=electricity_selling,
        variance=lambda _, series: electricity_selling_variance[len(series)]
    )
    return plant


def medium(
        horizon: Union[float, Iterable[float]],
        heat_demand: Union[float, Iterable[float]],
        gas_purchase: Union[float, Iterable[float]],
        electricity_selling: Union[float, Iterable[float]],
        heat_demand_variance: Union[float, Iterable[float]] = 0.0,
        gas_purchase_variance: Union[float, Iterable[float]] = 0.0,
        electricity_selling_variance: Union[float, Iterable[float]] = 0.0
) -> Plant:
    # noinspection DuplicatedCode
    plant = Plant(horizon=horizon)
    gas_purchase_variance = _get_variance(gas_purchase_variance, size=len(horizon))
    plant.add_extremity(
        kind='supplier',
        name='gas',
        commodity='gas',
        predictions=gas_purchase,
        variance=lambda _, series: gas_purchase_variance[len(series)]
    )
    plant.add_machine(
        name='boiler',
        parents='gas',
        commodity='gas',
        setpoint=[0, 1],
        inputs=[0, 100000],
        outputs={'heat': [0, 90000]}
    )
    plant.add_machine(
        name='chp',
        parents='gas',
        commodity='gas',
        setpoint=[0.5, 1],
        inputs=[67500, 135000],
        outputs={'heat': [27000, 50000], 'elec': [23000, 50000]},
        max_starting=(3, 24)
    )
    plant.add_storage(
        name='hws',
        commodity='heat',
        parents=['chp', 'boiler'],
        capacity=45000,
        charge_rate=15000,
        discharge_rate=15000,
        dissipation=0.02
    )
    heat_demand_variance = _get_variance(heat_demand_variance, size=len(horizon))
    plant.add_extremity(
        kind='customer',
        name='heat',
        commodity='heat',
        parents=['hws', 'chp', 'boiler'],
        predictions=heat_demand,
        variance=lambda _, series: heat_demand_variance[len(series)]
    )
    electricity_selling_variance = _get_variance(electricity_selling_variance, size=len(horizon))
    plant.add_extremity(
        kind='purchaser',
        name='grid',
        commodity='elec',
        parents='chp',
        predictions=electricity_selling,
        variance=lambda _, series: electricity_selling_variance[len(series)]
    )
    return plant


def hard(
        horizon: Union[float, Iterable[float]],
        heat_demand: Union[float, Iterable[float]],
        cooling_demand: Union[float, Iterable[float]],
        gas_purchase: Union[float, Iterable[float]],
        electricity_purchase: Union[float, Iterable[float]],
        electricity_selling: Union[float, Iterable[float]],
        heat_demand_variance: Union[float, Iterable[float]] = 0.0,
        cooling_demand_variance: Union[float, Iterable[float]] = 0.0,
        gas_purchase_variance: Union[float, Iterable[float]] = 0.0,
        electricity_purchase_variance: Union[float, Iterable[float]] = 0.0,
        electricity_selling_variance: Union[float, Iterable[float]] = 0.0
) -> Plant:
    plant = Plant(horizon=horizon)
    gas_purchase_variance = _get_variance(gas_purchase_variance, size=len(horizon))
    plant.add_extremity(
        kind='supplier',
        name='gas',
        commodity='gas',
        predictions=gas_purchase,
        variance=lambda _, series: gas_purchase_variance[len(series)]
    )
    electricity_purchase_variance = _get_variance(electricity_purchase_variance, size=len(horizon))
    plant.add_extremity(
        kind='supplier',
        name='elec',
        commodity='elec',
        predictions=electricity_purchase,
        variance=lambda _, series: electricity_purchase_variance[len(series)]
    )
    plant.add_machine(
        name='boiler',
        parents='gas',
        commodity='gas',
        setpoint=[0, 1],
        inputs=[0, 100000],
        outputs={'heat': [0, 90000]}
    )
    plant.add_machine(
        name='chp',
        parents='gas',
        commodity='gas',
        setpoint=[0.5, 1],
        inputs=[67500, 135000],
        outputs={'heat': [27000, 50000], 'elec': [23000, 50000]},
        max_starting=(3, 24)
    )
    plant.add_storage(
        name='hws',
        commodity='heat',
        parents=['chp', 'boiler'],
        capacity=45000,
        charge_rate=15000,
        discharge_rate=15000,
        dissipation=0.02
    )
    plant.add_machine(
        name='e_chiller',
        parents=['elec', 'chp'],
        commodity='elec',
        setpoint=[0, 1],
        inputs=[0, 7000],
        outputs={'cool': [0, 2000]}
    )
    plant.add_machine(
        name='a_chiller',
        parents=['chp', 'boiler', 'hws'],
        commodity='heat',
        setpoint=[0, 1],
        inputs=[0, 3000],
        outputs={'cool': [0, 2000]}
    )
    heat_demand_variance = _get_variance(heat_demand_variance, size=len(horizon))
    plant.add_extremity(
        kind='customer',
        name='heat',
        commodity='heat',
        parents=['hws', 'chp', 'boiler'],
        predictions=heat_demand,
        variance=lambda _, series: heat_demand_variance[len(series)]
    )
    cooling_demand_variance = _get_variance(cooling_demand_variance, size=len(horizon))
    plant.add_extremity(
        kind='customer',
        name='cool',
        commodity='cool',
        parents=['a_chiller', 'e_chiller'],
        predictions=cooling_demand,
        variance=lambda _, series: cooling_demand_variance[len(series)]
    )
    electricity_selling_variance = _get_variance(electricity_selling_variance, size=len(horizon))
    plant.add_extremity(
        kind='purchaser',
        name='grid',
        commodity='elec',
        parents='chp',
        predictions=electricity_selling,
        variance=lambda _, series: electricity_selling_variance[len(series)]
    )
    return plant
