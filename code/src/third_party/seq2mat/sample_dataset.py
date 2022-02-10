import argparse
import itertools as it
import random
from tqdm import tqdm


def main():
    """Main entry point
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs='+',
                        help="Pathes to one-sentence-per-line text files")
    parser.add_argument("-p", required=True,
                        type=float, help="Prob. of line to be sampled")
    parser.add_argument("-o",
                        required=True, help="Path for output")
    args = parser.parse_args()

    data = []
    full_count, sample_count = {}, {}
    for path in args.path:
        full_count[path] = 0
        sample_count[path] = 0
        with open(path, 'r') as file:
            for line in tqdm(file, desc=path):
                full_count[path] += 1
                if random.random() < args.p:
                    sample_count[path] += 1
                    data.append(line.strip())

    for path in args.path:
        print(f"[{path}] Sampled {sample_count[path]} of {full_count[path]} lines: {sample_count[path] * 100 / full_count[path]:3.2f}%")

    print("Writing", len(data), "lines to", args.o)
    with open(args.o, 'w') as outfile:
        for sentence in tqdm(data):
            print(sentence, file=outfile)

    print("Done")



if __name__ == "__main__":
    main()
