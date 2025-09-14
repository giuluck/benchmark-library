from benchmarks import beluga_logistics as bl
import json
import os

if __name__ == "__main__":
    # Load a problem description
    prb_fname = os.path.join('benchmarks', 'beluga_logistics', 'test_files', 'problem_j3_r2_oc00_f3_s9_64.json')
    with open(prb_fname) as fp:
        prb = json.load(fp)

    # Load a possible plan for the problem
    plan_fname = os.path.join('benchmarks', 'beluga_logistics', 'test_files', 'problem_j3_r2_oc00_f3_s9_64_plan.json')
    with open(plan_fname) as fp:
        plan = json.load(fp)

    # Build a benchmark instance
    benchmark = bl.BelugaLogisticsDeterministic(name='Test problem')

    # Trigger evaluation
    res = benchmark.query(problem=prb, plan=plan)

    print('Query Output')
    print('=' * 78)
    print(res)
    print()

    print('Metrics')
    print('=' * 78)
    print(benchmark.evaluate())

