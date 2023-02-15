from benchmarks.ackley.code import Ackley
from benchmarks.benchmark import Benchmark
from benchmarks.epidemics.code import EpidemicControl

# # DYNAMIC CLASS CREATION (THIS WORKS, BUT THE IDE CONSIDER IMPORTS AS ERRORS BECAUSE CLASSES ARE NOT YET AVAILABLE)
# for _, module, is_package in pkgutil.iter_modules(['benchmarks']):
#     if is_package:
#         name = module.title().replace(' ', '')
#         klass = type(name, (Benchmark,), {})
#         klass.package = module
#         globals()[name] = klass
