# tests for npz version of the pipeline

test_img = np.zeros((400, 400, 1))
for i in range(12):
    for j in range(11):
        test_img[(i * 35):(i * 35 + 10), (j * 37):(j * 37 + 8), 0] = (i + 1) * (j + 1)

base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/code/deepcell-toolbox/data/chop_testing/"
np.savez(base_dir + "test.npz", X=np.zeros((400, 400, 3)), y=test_img)

from deepcell_toolbox.pre_annotation import npz_preprocessing
importlib.reload(npz_preprocessing)

npz_preprocessing.crop_npz("test.npz", base_dir, (200, 200), 0.2)


stacked_npz = combine_npz(base_dir + "chopped")


row_start, row_end, row_padding = npz_preprocessing.compute_crop_indices(img_len=400, crop_size=200,
                                                                           overlap_frac=0.2)

X_cropped, padded_shape = npz_preprocessing.crop_images(np.zeros((400, 400, 1)), row_start=row_start, row_end=row_end, col_start=row_start, col_end=row_end,
                                      padding=(row_padding, row_padding, (0, 0)))


stitched = stitch_crops(stacked_npz, padded_shape, row_start, row_end, row_start, row_end)
io.imshow(stitched[:, :, 0])

npz_y = np.load(base_dir + "/chopped/test.npz_crop_9.npz")['y']
io.imshow(npz_y[:, :, 0])
