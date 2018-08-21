import numpy as np
import os

def combine_all():
    setlst = os.listdir('./')
    all_sets = []
    for term in setlst:
        if 'set' in term:
            all_sets.append(term)

    for set in all_sets:
        temp = os.listdir(os.path.join('.', set, ))
        base_direc = os.path.join('.', set, 'movie')
        output_path = os.path.join('.', set, 'final')
        partslst = []
        if not 'annotations' in temp:
            partslst = os.listdir(os.path.join('.', set))
        print(partslst)
        if len(partslst) == 0:
            print(base_direc, output_path)
            combine(base_direc, output_path)
        else:
            for part in partslst:
                base_direc = os.path.join('.', set, part, 'movie')
                output_path = os.path.join('.', set, part, 'final')
                combine(base_direc, output_path)


def combine(base_direc, output_path):
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
