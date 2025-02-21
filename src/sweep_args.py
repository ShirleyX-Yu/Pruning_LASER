import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='tuning hyperparams for the modified LASER model')
    parser.add_argument('--lname',
						help='matrix type',
						type=str,
						default=100)
    parser.add_argument('--rate',
						help='ρ = 1 - 0.1 * rate',
						type=float,
						default=64)
    parser.add_argument('--lnum',
						help='layer number',
						type=int,
						default=0.001)
    args = parser.parse_args()
    return args