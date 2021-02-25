#!/usr/bin/python
import os, sys

def main():
	if len(sys.argv) != 3:
		sys.exit('Usage: python trace_extractor.py <tracefile.txt> <output>')
	else:
		datafile = sys.argv[1]
		output = sys.argv[2]

	data = open(datafile, 'r')
	out = open(output, 'w')
	trace = []

	for line in data:
		if (line == '0.0 0.0\n'):
			for line in data:
				if not(line.startswith('#')):
					trace.append(line.split(' ',)[1].rstrip())
					if (line == '0.0 0.0\n'):
						if (len(trace) > 1):
							out.write('['+",".join(trace)+']\n')
						out.close()
						out = open(output, 'a')
						trace = []
	out.write('['+",".join(trace)+']\n')
	out.close()
	data.close()		

if __name__ == '__main__':
    main()
