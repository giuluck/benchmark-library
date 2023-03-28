from typing import Optional, Callable, List

import numpy as np
from pandas import DataFrame
from scipy.integrate import odeint

from model import Benchmark, Structure, querymethod

# TODO :- consider time horizon greater than the actual span of interest; 
# you have an a posteriori for censored

class Cox(Benchmark):
    
    @staticmethod
    def build(structure: Structure):
        # variables 
        structure.add_custom_variable('h0',
                                      dtype=Callable,
                                      description='baseline hazard function')
        structure.add_custom_variable('x',
                                      dtype=np.ndarray,
                                      description='covariates')
        structure.add_custom_variable('beta',
                                      dtype=np.ndarray,
                                      description='coefficients of covariates')
        # parameters
        structure.add_positive_parameter(
            name='horizon',
            default=1827, # 365 * 5 + 2 where 2 anni bisestili
            strict=True, integer=True, description='the timespan of the simulation')
        #constraint
        # TODO
        # metrics
        # TODO
        
    @querymethod
    def query(self, h0: Callable, x: np.ndarray, beta: np.ndarray) -> DataFrame:
        
        # cox model 
        H = []
        for t in range(self.horizon):
            H.append(h0(t) * np.exp(np.sum(x * beta, axis=1)))
        H = np.array(H)
        
        # build dataframe
        n_pat = x.shape[0]
        id_pat = np.arange(0, n_pat, 1).reshape(-1, 1)
        x = np.concatenate([id_pat, x], axis=1)
        data = np.array([np.concatenate([np.arange(0, self.horizon, 1).reshape(-1, 1),
                         np.repeat(x[i].reshape(1, -1), self.horizon, axis=0),
                         H[:,i].reshape(-1, 1)], axis=1)
                        for i in range(n_pat)])
        data = np.concatenate(data, axis=0)
        cov_name = ['x_{}'.format(i) for i in range(x.shape[1] - 1)]
        data = DataFrame(data, columns=['horizon', 'id_pat'] + cov_name + ['h_t'])   
        
        return data
    
    def __init__(self, name: Optional[str] = None, horizon: int = 1827):
        super(Cox, self).__init__(name=name, seed=None, horizon=horizon)
        
    @property
    def horizon(self) -> int:
        return self._configuration['horizon']
        
    
if __name__ == '__main__':
    
    import numpy as np 
    
    h0 = lambda t: .2
    x = np.random.rand(size=(n_pat, n_cov))
    beta = np.random.rand(size=(n_cov,))
    instance = Cox(h0, x, beta)
        
        
        
            