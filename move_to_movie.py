import os
import shutil
def into_movie():

	stacked_raw = input('path to stacked raw: ')
	destination = input('path to movie: ')
	raw_segments = os.listdir(stacked_raw)
	for segment in raw_segments:
		i = segment.split('x_')[1][0]
		j = segment.split('y_')[1][0]
		files = os.listdir(os.path.join(stacked_raw, segment))
		output_folder = os.path.join(destination, 'montage_'+str(i)+'_'+str(j), 'raw')
		if os.path.isdir(os.path.join(destination, 'montage_'+str(i)+'_'+str(j))) is False:
			print('montage_missing')
			continue
		elif os.path.isdir(output_folder) is False:
			os.mkdir(output_folder)
		for f in files:
			new_name = ((f.split('slice_')[1]).split('.')[0]).zfill(2) + '.png'
			shutil.copy(os.path.join(stacked_raw, segment, f), os.path.join(output_folder, new_name))

if __name__ == '__main__':
	into_movie()
