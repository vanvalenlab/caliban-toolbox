import subprocess
import sys
import yaml
import fileinput
import os
from csvtonpz import npz_runner



def run_celltk(newdir):
	os.chdir('../')
	fname = "./input_files/new_helatest.yml"

	stream = open(fname, 'r')
	data = yaml.load(stream)
	lst_dir = os.listdir('./' + newdir + 'movie/')
	print(lst_dir)
	common_dir = input('Longest common string from index 0: ')
	print(common_dir)
	base_dir = os.getcwd() + '/' + newdir
	irange = 0
	for term in lst_dir:
		print(term)
		x = term.split(common_dir)[1][0]
		if int(x) > irange:
			irange = int(x)

	jrange = 0
	for term in lst_dir:
		x = term.split(common_dir)[1][2:]
		if int(x) > jrange:
			jrange = int(x)

	if not os.path.exists(base_dir + 'HeLa_output/'):
		os.makedirs(base_dir + 'HeLa_output/')
	print(base_dir)
	for i in range(irange + 1):
		for j in range(jrange + 1):

			if os.path.isdir(base_dir + 'movie/' + common_dir + str(i) + '_0' + str(j)) is True:
				data['operations'][0]['images'] = base_dir + 'movie/' + common_dir + str(i) + '_0' + str(j) + '/raw/'
				data['operations'][0]['labels'] = base_dir + 'movie/' + common_dir + str(i) + '_0' + str(j) + '/annotated/'
				data['operations'][1]['images'][0] = base_dir + 'movie/' + common_dir + str(i) + '_0' + str(j) + '/raw/*'
				data['OUTPUT_DIR'] = base_dir + 'HeLa_output/' + common_dir + str(i) + '_0' + str(j)
				print(data['OUTPUT_DIR'])

				with open(fname, 'w') as yaml_file:
				    yaml_file.write( yaml.dump(data, default_flow_style=False))

				command = 'python celltk/caller.py ' + fname
				p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				out, err = p.communicate()
				print(out,err)
				print('0' + str(i) + '_' + str(j))

	#npz_runner()



if __name__ == "__main__":
   run_celltk(newdir)
