import numpy as np
import csv
import os

def read_csv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) == 1:
                rows.append(row[0])
            else:
                rows.append(row)
    return rows

def parse(filename):
    content = read_csv(filename)
    params = {}
    for c in content:
        params[c[0]] = os.path.abspath(c[1])
    return params

