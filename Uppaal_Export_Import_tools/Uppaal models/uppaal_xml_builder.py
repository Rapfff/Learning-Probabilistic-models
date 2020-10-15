#!/usr/bin/python

#Version 1.0 = Empty template creator.

import sys, getopt

def main(argv):
	#TODO figure out file format for input file.
	inputfile = ''
	outputfile = 'default.xml'
	suffix = '.xml'

	#Work in progress
	try:
		opts, args = getopt.getopt(argv, "hi:o:",["ifile=","ofile="])
		for opt, arg in opts:
			if opt == '-h':
				print(sys.argv[0],'-i <inputfile> -o <outputfile>')
				sys.exit()
			elif opt in ("-i", "--ifile"):
				inputfile = arg
			elif opt in ("-o", "--ofile"):
				outputfile = arg
	except getopt.GetoptError:
		print(sys.argv[0],'-i <inputfile> -o <outputfile>')
	else:
		print(len(argv))

	if(not outputfile.endswith(suffix)):
		outputfile+=suffix

	print('Input file is \''+inputfile+'\'')
	print('Output file is \''+outputfile+'\'')

	#Creates an empty xml template, that Uppaal should understand.

	#File header
	header  = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\r\n"
	header += "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>\r\n"
	header += "<nta>\r\n"

	#Global declarations
	gl_decl = "\t<declaration>// Place global declarations here.</declaration>\r\n"

	#Local declarations
	lc_decl = "\t\t<declaration>// Place local declarations here.</declaration>\r\n"


	#Template data, containing locational data of the model, for Uppaal's visual editor
	template  = "\t<template>\r\n"
	template += "\t\t<name x=\"5\" y=\"5\">Template</name>\r\n"
	template += lc_decl
	template += "\t\t<location id=\"id0\" x=\"0\" y=\"0\">\r\n"
	template += "\t\t</location>\r\n"
	template += "\t\t<init ref=\"id0\"/>\r\n"
	template += "\t</template>\r\n"

	#System declarations
	sys_decl  = "\t<system>// Place template instantiations here.\n"
	sys_decl += "Process = Template();\n"
	sys_decl += "// List one or more processes to be composed into a system.\n"
	sys_decl += "system Process;\n"
	sys_decl += "</system>\r\n"

	#Queries for verifier
	queries  = "\t<queries>\r\n"
	queries += "\t\t<query>\r\n"
	queries += "\t\t\t<formula></formula>\r\n"
	queries += "\t\t\t<comment></comment>\r\n"
	queries += "\t\t</query>\r\n"
	queries += "\t</queries>\r\n"
	queries += "</nta>\r\n"

	document = open(outputfile, "w")
	document.write(header+gl_decl+template+sys_decl+queries)

	document.close()

if __name__ == "__main__":
    main(sys.argv[1:])