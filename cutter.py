
from post_annotation_scripts.cut_raw_segments import cut_raw
num_runs = int(input('how many time to run? '))
for x in range(num_runs):
	cut_raw()
