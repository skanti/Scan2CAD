import json

def write(filename, data):
	with open(filename, 'w') as outfile:
		json.dump(data, outfile)

def read(filename):
	with open(filename, 'r') as infile:
		return json.load(infile)
