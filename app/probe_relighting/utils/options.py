import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--step', default=5, type=int)
parser.add_argument('--reduced', default=1, type=bool)
parser.add_argument('--intensity', default=1, type=int)
parser.add_argument('--mode', default='synthetic', 
                    choices=['synthetic', 'mid'],
                    type=str)
args = parser.parse_args()
