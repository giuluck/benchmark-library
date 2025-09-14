from benchmarks import guided_flight_planning as gfp

if __name__ == "__main__":
    benchmark = gfp.GuidedFlightPlanning(name='Test problem', steps=41)
    trajectory = benchmark.query(
                    origin=[48.99566, 2.55216, 392], # 'LFPG'
                    destination=[50.03262, 8.53463, 355.], # 'EDDF'
                    lat=[48.995660, 49.308684, 49.619001, 49.759132, 50.032620],
                    lon=[2.552160, 4.038455, 5.525219, 7.041482, 8.534630],
                    alt=[0.0, 36000.0, 36000.0, 36000.0, 355.0],
                    mass=[78000.000000, 77223.648614, 76945.994792, 76671.747974, 76627.161113])

    print('Computed trajectory')
    print('=' * 78)
    print(trajectory)
    print()
    print('Metrics')
    print('=' * 78)
    print(benchmark.evaluate())

