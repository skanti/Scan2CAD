import csv

def read(filename, skip_header=False):
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if skip_header:
            next(reader, None)
        for row in reader:
            if len(row) == 1:
                rows.append(row[0])
            else:
                rows.append(row)
    return rows

def read_as_dict(filename, split=","):
    rows = []
    with open(filename, 'r') as csvfile:
        for row in csv.DictReader(csvfile, delimiter=','):
            rows.append(row)
    return rows

def write(filename, rows, mode="w"):
    with open(filename, mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if type(rows[0]) is tuple:
            writer.writerows(rows)
        else:
            writer.writerow(rows)
