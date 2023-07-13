

import lxml.etree as ET
import gzip
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd


def get_paths_from_traces_file(traces_file):
    tree = ET.parse(traces_file)

    root = tree.getroot()
    all_paths = []
    path_lengths = []
    for path in root.findall('path'):
        length=path.get('reallength')
        path_points = []
        for point in path:
            path_points.append((int(point.get('x')), int(point.get('y')), int(point.get('z'))))
        all_paths.append(path_points)
        path_lengths.append(length)
    return all_paths, path_lengths

def visualise_ordering(points_list, dim):
    rdim, cdim, _ = dim
    vis = np.zeros((rdim, cdim, 3), dtype=np.uint8)

    def get_col(i):
        r = int(255 * i/len(points_list))
        g = 255 - r
        return r, g, 0

    for n, p in enumerate(points_list):
        c, r, _ = p
        wr, wc = 5, 5
        vis[max(0,r-wr):min(rdim,r+wr),max(0,c-wc):min(cdim,c+wc)] = get_col(n)

    return vis

col_map = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

def draw_paths(all_paths, foci_stack):
    im = np.max(foci_stack, axis=0)
    im = (im/np.max(im)*255).astype(np.uint8)
    im = np.dstack((im,)*3)
    im = Image.fromarray(im) #.convert('RGB')
    draw = ImageDraw.Draw(im)
    for i, (p, col) in enumerate(zip(all_paths, col_map)):
        draw.line([(u[0], u[1]) for u in p], fill=col)
        draw.text((p[0][0], p[0][1]), str(i+1), fill=col)
    return im
    

# Sum of measure_stack over regin where mask==1
def measure_from_mask(mask, measure_stack):
    return np.sum(mask * measure_stack)

# Max of measure_stack over region where mask==1
def max_from_mask(mask, measure_stack):
    return np.max(mask * measure_stack)


# Translate mask to point p, treating makss near stack edges correctly
def make_mask_s(p, melem, measure_stack):
    mask = melem
    
    R = melem.shape[0] // 2
    r, c, z = p

    m_data = np.zeros(melem.shape)
    s = measure_stack.shape
    o_1, o_2, o_3 = max(R-r, 0), max(R-c, 0), max(R-z,0)
    e_1, e_2, e_3 = min(R-r+s[0], 2*R), min(R-c+s[1], 2*R), min(R-z+s[2], 2*R)
    m_data[o_1:e_1,o_2:e_2,o_3:e_3] = measure_stack[max(r-R,0):min(r+R,s[0]),max(c-R,0):min(c+R,s[1]),max(z-R,0):min(z+R, s[2])]
    return mask, m_data

# Measure the (mean/max) value of measure_stack about the point p, using
# the structuring element melem. op indicates the appropriate measurement (mean/max)
def measure_at_point(p, melem, measure_stack, op='mean'):
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(melem)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
        return float(max_from_mask(mask, m_data))

# Generate spherical region
def make_sphere(R=5, z_scale_ratio=2.3):
    x, y, z = np.ogrid[-R:R, -R:R, -R:R]
    sphere = x**2 + y**2 + (z_scale_ratio * z)**2 < R**2
    return sphere

# Measure the values of measure_stack at each of the points of points_list in turn.
# Measurement is the mean / max (specified by op) on the spherical region about each point
def measure_all_with_sphere(points_list, measure_stack, op='mean'):
    melem = make_sphere()
    measure_func = lambda p: measure_at_point(p, melem, measure_stack, op)
    return list(map(measure_func, points_list))


# Measure fluorescence levels along ordered skeleton
def measure_chrom2(path, hei10):
    # single chrom - structure containing skeleton (single_chrom.skel) and
    # fluorecence levels (single_chrom.hei10) as Image3D objects (equivalent to ndarray)
    # Returns list of coordinates in skeleton, the ordered path 
    vis = visualise_ordering(path, dim=hei10.shape)
 
    measurements = measure_all_with_sphere(path, hei10, op='mean')
    measurements_max = measure_all_with_sphere(path, hei10, op='max')

    return vis, measurements, measurements_max

def extract_peaks(cell_id, all_paths, path_lengths, measured_traces):

    n = len(all_paths)
    
    
    #headers = ['Cell_ID', 'Trace', 'Trace_length(um)', 'detection_sphere_radius(um)', 'Foci_ID_threshold', 'Foci_per_trace']
    #for i in range(max_n):
    #    headers += [f'Foci{i}_relative_intensity', f'Foci_{i}_position(um)']

    data_dict = {}
    data_dict['Cell_ID'] = [cell_id]*n
    data_dict['Trace'] = range(1, n+1)
    data_dict['Trace_length(um)'] = path_lengths
    data_dict['Detection_sphere_radius(um)'] = [0.2]*n
    data_dict['Foci_ID_threshold'] = [0.4]*n

    
        
    return pd.DataFrame(data_dict)


def analyse_paths(cell_id, foci_file, traces_file):
    foci_stack = tifffile.imread(foci_file)
    all_paths, path_lengths = get_paths_from_traces_file(traces_file)

    all_trace_vis = []
    all_m = []
    for p in all_paths:
        vis, m, _ = measure_chrom2(p,foci_stack.transpose(2,1,0))
        all_trace_vis.append(vis)
        all_m.append(m)

    trace_overlay = draw_paths(all_paths, foci_stack)

    fig, ax = plt.subplots(len(all_paths),1)
    for i, m in enumerate(all_m):
        ax[i].plot(m)

    extracted_peaks = extract_peaks(cell_id, all_paths, path_lengths, all_m)
    
    return trace_overlay, all_trace_vis, fig, extracted_peaks
