import bz2
from datetime import datetime
import os
from pathlib import Path
import pickle
import random
from matplotlib import pyplot as plt

import numpy as np
import torch
from tqdm import tqdm
import yaml
from addict import Dict

def print_config(config):
    lines = ["\nConfig:"]
    keys, vals, typs = [], [], []
    for key, val in vars(config).items():
        keys.append(key + ":")
        vals.append(_format_value(val))
        typs.append(_format_type(val))
    max_key = max(len(k) for k in keys) if keys else 0
    max_val = max(len(v) for v in vals) if vals else 0
    for key, val, typ in zip(keys, vals, typs):
        key = key.ljust(max_key)
        val = val.ljust(max_val)
        lines.append(f"{key}  {val}  ({typ})")
    return "\n".join(lines)


def _format_value(value):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in value) + "]"
    return str(value)


def _format_type(value):
    if isinstance(value, (list, tuple)):
        assert len(value) > 0, value
        return _format_type(value[0]) + "s"
    return str(type(value).__name__)


def set_seed(seed: int, cuda=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if cuda:
        torch.cuda.manual_seed(np.random.randint(1, 10000))


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)



# Simple ISO 8601 timestamped logger
def log(s):
    msg = '[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s
    tqdm.write(f"{msg}")

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def make_bar(splits, data, path, fname):
    plt.figure()
    plt.bar(splits, data)
    plt.xlabel("Data split")
    plt.ylabel("Num samples")
    plt.tight_layout()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fname = path / fname
    plt.savefig(fname, bbox_inches="tight", dpi=120)
    plt.close()