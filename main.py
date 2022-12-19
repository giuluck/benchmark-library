import seaborn as sns

from benchmarks import EpidemicControl

sns.set_theme(context="talk", style="whitegrid", palette="deep")

if __name__ == "__main__":
    benchmark = EpidemicControl()
    print(benchmark.data)
