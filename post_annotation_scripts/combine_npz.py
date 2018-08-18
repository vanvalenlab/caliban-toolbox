import numpy as np
import os

def combine():
	output_path = './final'
	base_direc = './movie/'
	movies = os.listdir(base_direc)
	movies.sort()
	children = []
	for movie in movies:
		path = os.path.join(base_direc, movie, 'division.npz')
		if os.path.isfile(path):
			print(movie)
			training_data = np.load(path)
			children.append(training_data['arr_0'].tolist())

	for batch in range(len(children)):
		for i, lst in enumerate(children[batch]):
			children[batch][i] = np.asarray(lst, dtype=int)

	children = np.array(children)
	np.savez(os.path.join(output_path, 'combined_daugthers.npz'), daughters=children)
	data = np.load(os.path.join(output_path, 'combined_daugthers.npz'))

if __name__ == '__main__':
	combine()
