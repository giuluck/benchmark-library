import pandas as pd

from ppsim import Plant


def simple(df: pd.DataFrame) -> Plant:
    plant = Plant(horizon=df.index)
    plant.add_extremity(
        kind='supplier',
        name='gas',
        commodity='gas',
        predictions=df['gas purchase']
    )
    plant.add_machine(name='boiler', parents='gas', setpoint={
        'setpoint': [0, 1],
        'input': {'gas': [0, 100]},
        'output': {'heat': [0, 90]}
    })
    plant.add_machine(name='chp', parents='gas', max_starting=(3, 24), setpoint={
        'setpoint': [0.5, 1],
        'input': {'gas': [67.5, 135]},
        'output': {'heat': [27, 50], 'elec': [23, 50]}
    })
    plant.add_extremity(
        kind='customer',
        name='heat',
        commodity='heat',
        parents=['chp', 'boiler'],
        predictions=df['heat demand']
    )
    plant.add_extremity(
        kind='purchaser',
        name='grid',
        commodity='elec',
        parents='chp',
        predictions=df['electricity selling']
    )
    return plant


def medium(df: pd.DataFrame) -> Plant:
    plant = Plant(horizon=df.index)
    plant.add_extremity(
        kind='supplier',
        name='gas',
        commodity='gas',
        predictions=df['gas purchase']
    )
    plant.add_machine(name='boiler', parents='gas', setpoint={
        'setpoint': [0, 1],
        'input': {'gas': [0, 100]},
        'output': {'heat': [0, 90]}
    })
    plant.add_machine(name='chp', parents='gas', setpoint={
        'setpoint': [0.5, 1],
        'input': {'gas': [67.5, 135]},
        'output': {'heat': [27, 50], 'elec': [23, 50]}
    })
    plant.add_storage(
        name='hws',
        commodity='heat',
        parents=['chp', 'boiler'],
        capacity=45,
        dissipation=0.02
    )
    plant.add_extremity(
        kind='customer',
        name='heat',
        commodity='heat',
        parents=['chp', 'boiler', 'hws'],
        predictions=df['heat demand']
    )
    plant.add_extremity(
        kind='purchaser',
        name='grid',
        commodity='elec',
        parents='chp',
        predictions=df['electricity selling']
    )
    return plant


def hard(df: pd.DataFrame) -> Plant:
    plant = Plant(horizon=df.index)
    plant.add_extremity(
        kind='supplier',
        name='gas',
        commodity='gas',
        predictions=df['gas purchase']
    )
    plant.add_extremity(
        kind='supplier',
        name='elec',
        commodity='elec',
        predictions=df['electricity purchase']
    )
    plant.add_machine(name='boiler', parents='gas', setpoint={
        'setpoint': [0, 1],
        'input': {'gas': [0, 100]},
        'output': {'heat': [0, 90]}
    })
    plant.add_machine(name='chp', parents='gas', setpoint={
        'setpoint': [0.5, 1],
        'input': {'gas': [67.5, 135]},
        'output': {'heat': [27, 50], 'elec': [23, 50]}
    })
    plant.add_storage(
        name='hws',
        commodity='heat',
        parents=['chp', 'boiler'],
        capacity=45,
        dissipation=0.02
    )
    plant.add_machine(name='e_chiller', parents=['elec', 'chp'], setpoint={
        'setpoint': [0, 1],
        'input': {'elec': [0, 0.7]},
        'output': {'cool': [0, 2]}
    })
    plant.add_machine(name='a_chiller', parents=['chp', 'boiler', 'hws'], setpoint={
        'setpoint': [0, 1],
        'input': {'heat': [0, 3]},
        'output': {'cool': [0, 2]}
    })
    plant.add_extremity(
        kind='customer',
        name='heat',
        commodity='heat',
        parents=['chp', 'boiler', 'hws'],
        predictions=df['heat demand']
    )
    plant.add_extremity(
        kind='customer',
        name='cool',
        commodity='cool',
        parents=['a_chiller', 'e_chiller'],
        predictions=df['cooling demand']
    )
    plant.add_extremity(
        kind='purchaser',
        name='grid',
        commodity='elec',
        parents='chp',
        predictions=df['electricity selling']
    )
    return plant
