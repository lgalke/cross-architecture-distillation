import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs='+', help="Input file(s)")
parser.add_argument("-o", "--plot",  help="Where to write output figure")
parser.add_argument("--columns", help="Interpret input files as tsv and analyze provided columns",
                    default=None, nargs='+')
args = parser.parse_args()


def token_count(s):
    s = str(s).strip()
    tokens = s.split() # on white-space like characters
    return len(tokens)

if not args.columns:
    print("Reading files as one-sentence-per-line")
    seqlens = []
    try:
        for path in args.input:
            with open(path, 'r') as file:
                for line in tqdm(file, desc=file.name):
                    seqlens.append(token_count(line))
    except KeyboardInterrupt:
        pass
    finally:
        seqlens_series = pd.Series(seqlens, name="Sequence lengths")
        print(seqlens_series.describe())
        if args.plot:
            plt.figure()
            sns.histplot(seqlens_series.values)
            plt.savefig(args.plot)
else:
    print("Reading files as TSV")
    for path in args.input:
        df = pd.read_csv(path, sep='\t', error_bad_lines=False)
        for col in args.columns:
            text_series = df[col]
            seqlens_series = text_series.map(token_count)
            print(seqlens_series.describe())

