import os
import shutil
def move_raw():
	set_path = input('path to full set: ')
	raw = input('raw folder name: ')
	destination = input('set part folder name: ')
	if os.path.isdir(os.path.join(set_path, destination)) is False:
		print("error: invalid destination")
		return None
	raw_frames = os.listdir(os.path.join(set_path, raw))
	raw_frames.sort()
	print(raw_frames)

	num_frames = int(input('number of frames in a montage: '))
	start = int(input('index of first image to copy (remember 0 indexing!): '))

	output_folder = os.path.join(set_path, destination, 'raw')
	if os.path.isdir(output_folder) is False:
		os.mkdir(output_folder)
	for i in range(start, start+num_frames):
		shutil.copy(os.path.join(set_path, raw, raw_frames[i]), os.path.join(output_folder))
if __name__ == '__main__':
	#num_runs = int(input('how many set parts do you need to move in total? '))
	#for x in range(num_runs):
	move_raw()
