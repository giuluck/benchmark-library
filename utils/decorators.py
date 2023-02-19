import inspect


# noinspection PyPep8Naming
class delegates:
    def __init__(self, delegator):
        self.delegator = delegator

    def __call__(self, delegated=None, additional: bool = False):
        # if delegated is None, delegate the __init__ of the delegator to the __init__ of its base (delegated) class
        # otherwise, delegate the delegated function to the delegator function
        if delegated is None:
            delegated, delegator = self.delegator.__base__.__init__, self.delegator.__init__
        else:
            delegated, delegator = delegated, self.delegator
        # retrieve the signature of the delegator function, get its parameters and remove the kwargs
        signature = inspect.signature(delegator)
        params = dict(signature.parameters)
        kwargs = params.pop('kwargs')
        # update the list of parameters with all the parameters from the delegated signature that are not yet present
        params.update({k: v for k, v in inspect.signature(delegated).parameters.items() if k not in params})
        # put the "kwargs" back in the parameters list if necessary
        if additional:
            params['kwargs'] = kwargs
        # update the signature of the delegator function and return it
        # noinspection PyTypeChecker
        delegator.__signature__ = signature.replace(parameters=params.values())
        return self.delegator
