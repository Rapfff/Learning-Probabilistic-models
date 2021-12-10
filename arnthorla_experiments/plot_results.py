# -*- coding: utf-8 -*-
#import sys, os, datetime, random
import re
import matplotlib.pyplot as plt

#def createFullTargetFilename():
#    createResultsFolderIfNotExists()
#    now = datetime.datetime.now()
#    full_folder_path = fullResultsFolderName()
#    filename = ("result_" +str(now.year)+"_"+str(now.month)+"_"+str(now.day)+
#            "_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+".txt")
#    return os.path.join( full_folder_path, filename )


# Folders and filenames
#current_directory = os.path.join( os.path.dirname(os.path.realpath(__file__)))
#src_filename = "things_200_picturable_words.txt"
#src_folder = os.path.join( current_directory, "wordlists" )
#full_src_filename = os.path.join( src_folder, src_filename )
#target_folder = os.path.join( current_directory, "output" )
#
## Read wordlist from file
#words = []
#with open( full_src_filename ) as file:
#    for line in file:
#        words.append( line.rstrip() )
#
#
## Create file and folder name
#now = datetime.datetime.now()
#filename = (str(now.year)+"_"+str(now.month)+
#        "_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+
#        "_"+str(now.second)+"_"+str(now.microsecond%1000)+".pdf")
#full_target_filename = os.path.join( target_folder, filename )

# Parse results from result file.
results = []
with open( "experiment_1_4_results/result_2021_11_26_18_55_28.txt" ) as file:
    for line in file:
        line = line.split( "|" )
        tmp = []
        for w in line:
            w = w.rstrip()
            if len( w ) > 0:
                tmp.append( w )
        if len(tmp) == 4:
            results.append( tmp )
# Remove headers
results.pop(0)

i = 0
while i < len(results):
    states = results[i][0]
    seq, diff = [],[]
    while i < len(results) and states == results[i][0]:
        seq.append( int(results[i][1]) )
        diff.append( float(results[i][2]) )
        i += 1
    plt.xscale("log")
    #plt.yscale("log")
    plt.plot( seq, diff, label="#states " + str(states) )
plt.xlabel( "#Sequences" )
plt.ylabel( "log-Likelihood" )
plt.legend()
plt.savefig("test.png")

# Plot
#plt.plot([1,2,3,7],[1,2,3,4]) 
#plt.savefig("test.png")

print( results )
