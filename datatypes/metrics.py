from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any, Dict

from descriptors import classproperty, cachedproperty
from sklearn.metrics import log_loss, r2_score, mean_squared_error, mean_absolute_error, recall_score, \
    precision_score, f1_score, accuracy_score, roc_auc_score

from datatypes.datatype import DataType, Sample


@dataclass(repr=False, frozen=True)
class Metric(DataType, ABC):
    """Abstract class for benchmark metric."""

    @abstractmethod
    def __call__(self, sample: Sample) -> float:
        """Computes the metric value on a sample <inputs, output>.

        :param sample:
            A Sample object.

        :return:
            The metric value.
        """
        pass


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class SampleMetric(Metric):
    """A metric with custom function computed on a given sample."""

    metric_fn: Callable[[Sample], float] = field(kw_only=True)
    """A function f(sample) -> float which computes the metric on a given sample <inputs, output>."""

    def __call__(self, sample: Sample) -> float:
        return self.metric_fn(sample)


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class ValueMetric(Metric):
    """A metric with custom function computed on a single value of the given sample."""

    value: str = field(default='output', kw_only=True)
    """The input value on which to compute the metric, or 'output' to compute it on the output; default: 'output'."""

    metric_fn: Callable[[Any], float] = field(kw_only=True)
    """A function f(value) -> float which computes the metric on the specified value of the given sample."""

    def __call__(self, sample: Sample) -> float:
        return self.metric_fn(sample.output if self.value == 'output' else sample.inputs[self.value])


# noinspection PyDataclass,PyRedeclaration
@dataclass(repr=False, frozen=True)
class ReferenceMetric(ValueMetric):
    """A metric computed on a single value with respect to a reference value."""

    @classproperty
    def aliases(self) -> Dict[str, Callable[[Any, Any], float]]:
        """The supported metric aliases with their respective metric functions."""
        return dict(
            mae=mean_absolute_error,
            mse=mean_squared_error,
            r2=r2_score,
            crossentropy=log_loss,
            precision=precision_score,
            recall=recall_score,
            f1=f1_score,
            accuracy=accuracy_score,
            auc=roc_auc_score
        )

    metric: str | Callable[[Any, Any], float] = field(kw_only=True)
    """Either a metric alias or a custom function f(reference, value) -> float."""

    reference: Any = field(kw_only=True)
    """The optional reference output value on which to compare the sample output."""

    # redeclare 'metric_fn' as a property to match 'metric'
    metric_fn: Callable[[Any], float] = field(init=False)

    @cachedproperty
    def metric_fn(self) -> Callable[[Any], float]:
        metric = self.aliases.get(self.metric) if isinstance(self.metric, str) else self.metric
        return lambda value: metric(self.reference, value)
