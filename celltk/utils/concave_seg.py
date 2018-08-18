import numpy as np
from scipy.spatial.distance import cdist
import SimpleITK as sitk
from postprocess_utils import regionprops
from filters import label
import cv2
from skimage.draw import line
from scipy.ndimage import binary_erosion
from mahotas.segmentation import gvoronoi
from skimage.segmentation import find_boundaries
from skimage.morphology import thin
from functools import partial
from track_utils import calc_massdiff, find_one_to_one_assign
from scipy.spatial.distance import cdist


def cut_neck(template, r0, c0, r1, c1):
    """Given a label image and two coordinates, it will draw a line which intensity is 0.
    """
    # rr, cc, _ = line_aa(r0, c0, r1, c1)
    rr, cc = line(r0, c0, r1, c1)
    template[rr, cc] = 0
    return template


def make_candidates(cl_label, s0, s1, e0, e1):
    cut_label = cut_neck(cl_label, s0, s1, e0, e1)
    cand_label = label(cut_label, connectivity=1)
    cand_rps = regionprops(cand_label)
    return cand_rps


def make_candidates_img(cl_label, s0, s1, e0, e1, img):
    cut_label = cut_neck(cl_label, s0, s1, e0, e1)
    cand_label = label(cut_label, connectivity=1)
    cand_rps = regionprops(cand_label, img)
    return cand_rps


def intensity_below_line(img, r0, c0, r1, c1):
    rr, cc = line(r0, c0, r1, c1)
    return img[rr, cc]


def calc_shortest_step_coords(coords, co1, co2):
    co1 = [i for i, c in enumerate(coords) if (c == co1).all()][0]
    co2 = [i for i, c in enumerate(coords) if (c == co2).all()][0]
    return min(abs(co2 - co1), abs(co1 + coords.shape[0] - co2))


def find_oriented_coords(outline):
    cnt = cv2.findContours(outline.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1][0]
    cnt = np.flipud(cnt)
    # assert cv2.contourArea(cnt, oriented=True) > 0
    if not cv2.contourArea(cnt, oriented=True) > 0:
        return None
    return np.fliplr(np.squeeze(cnt))


def keep_labels(ref_labels, conv_labels):
    from collections import Counter
    pairs = []
    for i in [ii for ii in np.unique(conv_labels) if not ii == 0]:
        c = Counter(ref_labels[conv_labels == i])
        pairs.append((i, c.most_common()[0][0]))
    labels = conv_labels.copy()
    for p1, p2 in pairs:
        labels[conv_labels == p1] = p2
    return labels


def levelset_lap(img, labels, NITER=100, CURVE=3, PROP=-1):
    simg = sitk.GetImageFromArray(img, sitk.sitkFloat32)
    seg = sitk.GetImageFromArray(labels)
    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)

    lsFilter = sitk.LaplacianSegmentationLevelSetImageFilter()
    lsFilter.SetMaximumRMSError(0.02)
    lsFilter.SetNumberOfIterations(NITER)
    lsFilter.SetCurvatureScaling(CURVE)
    lsFilter.SetPropagationScaling(PROP)
    lsFilter.ReverseExpansionDirectionOn()
    ls = lsFilter.Execute(init_ls, sitk.Cast(simg, sitk.sitkFloat32))
    ls = label(sitk.GetArrayFromImage(ls) > 0)
    return keep_labels(labels, ls)


def levelset_geo(img, labels, advec=3, propagation=0.75, niter=100):
    """Propagate outwards
    """
    simg = sitk.GetImageFromArray(img, sitk.sitkFloat32)
    labels = label(binary_erosion(labels, np.ones((3, 3))))
    seg = sitk.GetImageFromArray(labels)
    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)

    gfil = sitk.GeodesicActiveContourLevelSetImageFilter()
    gfil.SetMaximumRMSError(0.02)
    gfil.SetNumberOfIterations(niter)
    gfil.SetAdvectionScaling(advec)
    gfil.SetCurvatureScaling(1)
    gfil.SetPropagationScaling(propagation)
    gfil.ReverseExpansionDirectionOn()
    ls = gfil.Execute(init_ls, sitk.Cast(simg, sitk.sitkFloat32))
    ls = label(sitk.GetArrayFromImage(ls) > 0)
    return keep_labels(labels, ls)


def extract_large(labels, AREA=700):
    rps = regionprops(labels)
    retain = filter(lambda x:x.area>AREA, rps)
    badlabels = np.zeros(labels.shape, np.uint16)
    for r in retain:
        badlabels[labels==r.label] = r.label
    return badlabels


def extract_small(labels, AREA=200):
    rps = regionprops(labels)
    retain = filter(lambda x:x.area<AREA, rps)
    badlabels = np.zeros(labels.shape, np.uint16)
    for r in retain:
        badlabels[labels==r.label] = r.label
    return badlabels


def wshed_raw(labels, im):
    """
    return wshed lines
    """
    ia = lambda x: sitk.GetImageFromArray(x)
    ai = lambda x: sitk.GetArrayFromImage(x)
    feature_img = ia(im)
    ws_img = sitk.MorphologicalWatershed(feature_img, level=0, markWatershedLine=True, fullyConnected=True)
    ws = ai(ws_img)
    ws = ws==0
    ws[labels == 0] = 0
    return ws


def cart2pol_angle(x, y):
    phi = np.arctan2(y, x)
    return phi


def calc_clockwise_degree(p, c, q):
    """Return an degree in clockwise if you give three points. c will be a center.
    >>> q = [10, 10]
    >>> c = [0, 0]
    >>> p = [-10, 10]
    >>> calc_closewise_degree(p, c, q)
    90.0
    """
    angle_r = cart2pol_angle(q[0]-c[0], q[1]-c[1]) - cart2pol_angle(p[0]-c[0], p[1]-c[1])
    angle = 180.0 * angle_r/np.pi
    if angle < 0:
        angle += 360.0
    return angle


class CoordsConcave(object):
    """coords needs to be ordered in clockwise.
    """
    def __init__(self, coords, edgelen=5, thres=180):
        self.coords = coords
        self.edgelen = edgelen
        self.thres = thres

    # def run(self):
    #     score, coords = self.calc_neck_score_thres()
    #     return coords

    def calc_neck_score(self):
        """Calculate the score (angle changes) and return the sorted score and the corresponding pixel
        coordinates. Pass the coordinates of outlines without border."""
        ordered_c, edgelen = self.coords, self.edgelen
        nc = np.vstack((ordered_c, ordered_c[:edgelen, :]))
        score = []
        for n, ci in enumerate(nc[:-edgelen, :]):
            try:
                score.append(calc_clockwise_degree(ordered_c[n-edgelen, :], nc[n, :], nc[n+edgelen, :]))
            except:
                pass
        idx = np.flipud(np.argsort(score))
        return np.array(score)[idx], ordered_c[idx]

    def calc_neck_score_thres(self):
        score, s_coords = self.calc_neck_score()
        return score[score > self.thres], s_coords[score > self.thres]


class CoordsConcaveWs(CoordsConcave):
    def __init__(self, coords, wlines, edgelen=5, thres=180):
        self.coords = coords
        self.edgelen = edgelen
        self.thres = thres
        self.wlines = wlines

    def run(self):
        score, coords = self.calc_neck_score_thres()
        return self.limit_coords_ws(coords), score

    def limit_coords_ws(self, coords):
        """modify later to find neighbors?
        """
        cstore = []
        for i in coords:
            if self.wlines[i[0], i[1]]:
                cstore.append(i)
        return np.array(cstore)


def cellfilter(cell, small_area, large_area, major_minor=2.0):
    if cell.area > small_area and cell.area < large_area and cell.major_axis_length/cell.minor_axis_length < major_minor:
        return True
    else:
        return False



class CellCutter(object):
    def __init__(self, bw, img, wlines, small_rad, large_rad=0, EDGELEN=10, THRES=180, CANDS_LIMIT=300):
        """
        Use lower values for CANDS_LIMIT to speed up
        """
        large_rad = 0  # fix it later.

        self.bw = bw
        self.img = img
        self.wlines = wlines
        self.EDGELEN = EDGELEN
        self.THRES = THRES
        self.filfunc = partial(cellfilter, small_area=np.pi*small_rad**2, large_area=np.pi*large_rad**2, major_minor=2.0)
        self.goodcells = []
        # self.large_rad = large_rad
        self.small_rad = small_rad
        self.STEPLIM = self.small_rad * np.pi/2
        self.coords_set = []
        self.cut_coords = []
        self.cut_cells = []
        self.CANDS_LIMIT = CANDS_LIMIT

    def extract_cell_outlines(self):
        o_coords = find_oriented_coords(self.bw)
        if o_coords is not None:
            self.cell.o_coords = o_coords

    def run(self):
        self.prepare_coords_set()
        if not self.coords_set:
            return
        self.collect_goodcells()
        self.make_bw()
        if not self.cell:
            return
        while self.cell.area > self.large_rad ** 2 * np.pi:
            self.remove_coords_set()
            self.collect_goodcells()
            self.make_bw()
            if not self.cell:
                break

    def collect_goodcells(self):
        candidates = self.search_cut_candidates(self.bw.copy(), self.coords_set)
        if not candidates:
            return
        candidate = self.filter_candidates(candidates)
        if not candidate:
            return
        (n1, n2) = candidate.n1, candidate.n2
        self.all_cells = make_candidates(self.bw.copy(), self.coords_set[n1][0][0], self.coords_set[n1][0][1], self.coords_set[n1][1][0], self.coords_set[n1][1][1])
        self.cut_coords.append(self.coords_set[n1])
        for c in self.all_cells:
            if self.filfunc(c):
                self.goodcells.append(c)
            else:
                self.cut_cells.append(c)

    def make_bw(self):
        if self.cut_cells:
            bw = np.zeros(self.bw.shape, np.uint8)
            for cell in self.cut_cells:
                for c in cell.coords:
                    bw[c[0], c[1]] = 1
            self.bw = bw
            self.cell = regionprops(bw, self.img)[0]
            self.cut_cells = []
        else:
            self.cell = []

    def remove_coords_set(self):
        store = []
        for c0, c1 in self.coords_set:
            if self.bw[c0[0], c0[1]] and self.bw[c1[0], c1[1]]:
                store.append((c0, c1))
        self.coords_set = store

    def prepare_coords_set(self):
        self.cell = regionprops(self.bw.astype(np.uint8), self.img)[0]
        if 1 in self.cell.image.shape:
            self.coords_set = []
            return
        self.extract_cell_outlines()
        if not hasattr(self.cell, 'o_coords'):
            return
        coords, _ = CoordsConcaveWs(self.cell.o_coords, wlines=self.wlines, edgelen=self.EDGELEN, thres=self.THRES).run()
        if not coords.any():
            return
        self.coords_set = self.make_sort_coordsset_by_dist(coords)
        if not self.coords_set:
            return
        self.coords_set = self.coords_set[:self.CANDS_LIMIT]
        self.coords_set = self.filter_coords_by_step(self.cell.o_coords, self.coords_set)        

    def make_sort_coordsset_by_dist(self, coords):
        dist = cdist(coords, coords)

        boolarr = np.triu(np.ones(dist.shape, bool), 1)
        a1 = np.tile(range(len(coords)), len(coords)).reshape((len(coords), len(coords)))
        x1, x2 = a1[boolarr], a1.T[boolarr]

        sortorder = np.argsort(dist[boolarr])
        x1, x2 = x1[sortorder], x2[sortorder]
        return [[coords[i1], coords[i2]] for i1, i2 in zip(x1, x2)]

    # def filter_coords_by_dist(self, coords):
    #     dist = cdist(coords, coords)
        # boolarr = np.triu(dist < self.DISTLIM, 1)
    #     a1 = np.tile(range(len(coords)), len(coords)).reshape((len(coords), len(coords)))
    #     x1, x2 = a1[boolarr], a1.T[boolarr]
    #     return [[coords[i1], coords[i2]] for i1, i2 in zip(x1, x2)]

    def filter_coords_by_step(self, coords, coords_set):
        new_coords_set = []
        for c0, c1 in coords_set:
            min_step = calc_shortest_step_coords(coords, c0, c1)
            if min_step > self.STEPLIM:
                new_coords_set.append([c0, c1])
        return new_coords_set

    def filter_candidates(self, candidates):
        candidates = filter(lambda x: x.area > np.pi * self.small_rad ** 2, candidates)
        if len(candidates):
            mindist = min([c.line_total for c in candidates])
            index = [n for n, c in enumerate(candidates) if c.line_total == mindist][0]
            return candidates[index]
        else:
            return []

    def search_cut_candidates(self, bw, coords_set):
        candidates = []
        for n1, (startpt, endpt) in enumerate(coords_set):
            # cut and get candidate regionprops
            cand_rps = make_candidates_img(bw.copy(), startpt[0], startpt[1], endpt[0], endpt[1], self.img)
            if not len(cand_rps) > len(np.unique(label(self.cell.filled_image, connectivity=1))) - 1:
                continue
            if not all([c.area > np.pi * self.small_rad**2 for c in cand_rps]):
                continue
            # scoring
            for n2, c in enumerate(cand_rps):
                # If you get an error here, it might be unresolved error of scikit-image. https://github.com/scikit-image/scikit-image/issues/1470
                # edit _regionprops._RegionProperties.moments and moments_central from "astype(np.uint8)" to "astype(np.float)"
                if c.minor_axis_length > 1:
                    c.n1, c.n2 = n1, n2
                    c.cut_coords = (startpt, endpt)
                    c.cutline_len = np.sqrt((startpt[0] - endpt[0])**2 + (startpt[1] - endpt[1])**2)
                    c.line_total = intensity_below_line(self.img, startpt[0], startpt[1], endpt[0], endpt[1]).sum()
                    candidates.append(c)
        return candidates

    def get_labels(self):
        temp = label(self.bw.copy(), connectivity=1)
        temp = np.zeros(self.bw.shape)

        if not self.goodcells:
            return label(self.bw)
        max_label = temp.max()
        for n, i in enumerate(self.all_cells):
            for c in i.coords:
                temp[c[0], c[1]] = max_label + n + 1
        return temp


def cut_labels(labels, coords_set):
    for i in coords_set:
        if i:
            labels = cut_neck(labels, i[0][0], i[0][1], i[1][0], i[1][1])
    return labels


def levelset_geo_separete(img, labels, niter=10, prop=-1, advec=-0.5, curve=-0.5):
    seg = sitk.GetImageFromArray((labels > 0).astype(np.uint8))
    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)

    mask = find_boundaries(gvoronoi(label(labels, connectivity=1)), mode='inner')
    img1 = img.copy()
    img1[mask] = 0

    img_T1 = sitk.GetImageFromArray(img1)
    lsFilter = sitk.GeodesicActiveContourLevelSetImageFilter()
    lsFilter.SetMaximumRMSError(0.02)
    lsFilter.SetNumberOfIterations(niter)
    lsFilter.SetAdvectionScaling(advec)
    lsFilter.SetCurvatureScaling(curve)
    lsFilter.SetPropagationScaling(prop)
    # lsFilter.ReverseExpansionDirectionOn()

    ls = lsFilter.Execute(init_ls, sitk.Cast(img_T1, sitk.sitkFloat32))
    return label(sitk.GetArrayFromImage(ls) > 0, connectivity=1)


def get_cut_coords(img, labels, small_rad, large_rad, EDGELEN=6, THRES=180):
    largela = extract_large(labels, AREA=np.pi * large_rad**2)
    wlines = wshed_raw(labels > 0, img)
    store = []
    for i in np.unique(largela):
        if i == 0:
            continue
        cc = CellCutter(labels==i, img, wlines, small_rad=small_rad, large_rad=large_rad, EDGELEN=EDGELEN, THRES=THRES)
        cc.run()
        store.append(cc.cut_coords)
    return store


def run_concave_cut(img, labels, small_rad, large_rad, EDGELEN=6, THRES=180):

    store = get_cut_coords(img, labels, small_rad, large_rad, EDGELEN=EDGELEN, THRES=THRES)
    labels = label(cut_labels(labels, [i for j in store for i in j]), connectivity=1)
    return labels

