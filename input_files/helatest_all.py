import subprocess
import sys

import fileinput




def main(argv):
	filename = '/home/input_files/helatest.yml'
	text_to_search = '00_1'
	for i in range(5):
		for j in range(5):
			replacement_text = '0' + str(i) + '_' + str(j)
			text_to_search = replacement_text

			f = fileinput.FileInput(filename, inplace=True, backup='.bak')
			for line in f:
				line.replace(text_to_search, replacement_text)
				print(line)
			f.close()
	# command = 'python celltk/caller.py input_files/helatest.yml' 
	# p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	# out, err = p.communicate()
	# print(out,err)

if __name__ == "__main__":
   main(sys.argv[1:])