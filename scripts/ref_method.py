import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, feature, measure, segmentation

GATE = 200  # blank-field rule: (p99.5 - median) < GATE -> no nuclei

def contrast(img):
    return float(np.percentile(img, 99.5) - np.median(img))

def segment_ref(img):
    """Reference: clip-normalize, background-correct, Otsu, distance-transform watershed."""
    if contrast(img) < GATE:
        return np.zeros(img.shape, np.int32)
    x = np.clip(img - np.median(img), 0, None)
    p = np.percentile(x, 99)
    x = filters.gaussian(np.clip(x, 0, p) / p, 1)
    corr = x - filters.gaussian(x, 40)
    mask = corr > filters.threshold_otsu(corr)
    mask = morphology.remove_small_objects(mask, max_size=29)
    mask = ndi.binary_fill_holes(mask)
    dist = filters.gaussian(ndi.distance_transform_edt(mask), 1)
    peaks = feature.peak_local_max(dist, min_distance=7,
                                   labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)

def segment_naive(img):
    """Naive: global Otsu on the raw image, count connected blobs."""
    return measure.label(img > filters.threshold_otsu(img))
