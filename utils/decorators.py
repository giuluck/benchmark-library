import inspect
from typing import Optional, Callable, Any


# noinspection PyPep8Naming
class delegates:
    """Decorator for the delegate pattern. It changes the signature of the delegator method in order to include the
    parameter of the delegated one. In case the delegated method is not passed, the decorator is assumed to be attached
    to a class object and the two functions are the constructors of the derived and the base class, respectively.
    """

    def __init__(self, delegated: Optional[Callable] = None, kwargs: bool = False):
        """
        :param delegated:
            The delegated method from which the parameters are retrieved. If None, the decorator is assumed to be
            attached to a class object and the two functions are the constructors of the derived and the base class,
            respectively; default: None.

        :param kwargs:
            Whether or not to keep the **kwargs parameter in the delegator method.
        """
        self.delegated: Optional[Callable] = delegated
        """The delegated method from which the parameters are retrieved."""

        self.kwargs: bool = kwargs
        """Whether or not to keep the **kwargs parameter in the delegator method."""

    def __call__(self, target) -> Any:
        """Changes the signature of the delegator method in order to include the parameter of the delegated one.
        If the delegated is None, the decorator is assumed to be attached to a class object and the two functions are
        the constructors of the derived and the base class, respectively.

        :param target:
            Either a class method or the class itself, in case the delegated is None.

        :return:
            The target itself.
        """
        # if delegated is None, delegate the __init__ of the delegator to the __init__ of its base (delegated) class
        # otherwise, delegate the delegated function to the delegator function
        if self.delegated is None:
            delegated, delegator = target.__base__.__init__, target.__init__
        else:
            delegated, delegator = self.delegated, target
        # retrieve the signature of the delegator method and get its parameters, then retrieve the delegated parameters
        signature = inspect.signature(delegator)
        delegator_params = signature.parameters
        delegated_params = inspect.signature(delegated).parameters
        # create a new dictionary of parameters where non-default base parameters are at the beginning
        params = {k: v for k, v in delegated_params.items() if v.default is v.empty}
        params.update({k: v for k, v in delegator_params.items() if k != 'kwargs' and k not in params})
        params.update({k: v for k, v in delegated_params.items() if v.default is not v.empty and k not in params})
        # add the "kwargs" back in the parameters list if necessary
        if self.kwargs:
            params['kwargs'] = delegator_params['kwargs']
        # update the signature of the delegator function and return it
        # noinspection PyTypeChecker
        delegator.__signature__ = signature.replace(parameters=params.values())
        return target
