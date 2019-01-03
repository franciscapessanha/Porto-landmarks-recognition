'''
This script renames a set of files in a folder given a label and an initial number
Arguments
Folder: the folder name in the current directory
Label: Label the user wishes to give to the files
Number: Initial number that will list the files with the label
'''

import os, sys
	
def main(folder, label, number):
	os.chdir("../dataset/images/" + str(folder))
	for root, dirs, files in os.walk("."): 
		num = int(number)
		for filename in files:
			print(num)
			sep = filename.split(".")
			if len(sep) < 2: continue
			os.rename(filename, str(label) + str("-") + "%04d" % num + "." + str(sep[1]))
			num += 1
			
			
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])