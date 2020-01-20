import numpy as np
from deepcell_toolbox.pre_annotation import npz_preprocessing
from deepcell_toolbox.post_annotation import npz_postprocessing
import importlib
importlib.reload(npz_preprocessing)

# tests for npz version of the pipeline


# create a test image with tiled unique values across the image
test_img = np.zeros((400, 400, 1))
for i in range(12):
    for j in range(11):
        test_img[(i * 35):(i * 35 + 10), (j * 37):(j * 37 + 8), 0] = (i + 1) * (j + 1)

base_dir = "tests/deepcell_toolbox/"
np.savez(base_dir + "test.npz", X=np.zeros((400, 400, 3)), y=test_img)


npz_preprocessing.crop_npz(npz_name="test.npz", base_dir=base_dir, save_name="test_folder", crop_size=(200, 200),
                           overlap_frac=0.2)

npz_postprocessing.reconstruct_npz(npz_dir=base_dir + "test_folder", original_npz=base_dir + "test.npz")

final_npz = np.load(os.path.join(base_dir, "test_folder", "stitched_npz.npz"))
stitched = final_npz["y"]

assert(np.all(np.equal(stitched[:, :, 0] > 0, test_img[:, :, 0] > 0)))