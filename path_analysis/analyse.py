
import lxml.etree as ET
import gzip
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from itertools import cycle
from .data_preprocess import analyse_traces
import math
import scipy.linalg as la


def get_paths_from_traces_file(traces_file):
    """
    Parses the specified traces file and extracts paths and their lengths.
    
    Args:
        traces_file (str): Path to the XML traces file.

    Returns:
        tuple: A tuple containing a list of paths (each path is a list of tuples representing points)
               and a list of corresponding path lengths.
    """
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
        path_lengths.append(float(length))
    return all_paths, path_lengths

def calculate_path_length(point_list, voxel_size=(1,1,1)):
    # Simple calculation
    l = 0
    s = np.array(voxel_size)
    for i in range(len(point_list)-1):
        l += la.norm(s * (np.array(point_list[i+1]) - np.array(point_list[i])))
    return l


def calculate_path_length_partials(point_list, voxel_size=(1,1,1)):
    # Simple calculation
    l = [0.0]
    s = np.array(voxel_size)
    for i in range(len(point_list)-1):
        l.append(la.norm(s * (np.array(point_list[i+1]) - np.array(point_list[i]))))
    return np.cumsum(l)


def visualise_ordering(points_list, dim, wr=5, wc=5):
    """
    Visualize the ordering of points in an image.
    
    Args:
        points_list (list): List of points to be visualized.
        dim (tuple): Dimensions of the image (rows, columns, channels).
        wr (int, optional): Width of the region to visualize around the point in the row direction. Defaults to 5.
        wc (int, optional): Width of the region to visualize around the point in the column direction. Defaults to 5.

    Returns:
        np.array: An image array with visualized points.
    """
    # Visualizes the ordering of the points in the list on a blank image.
    rdim, cdim, _ = dim
    vis = np.zeros((rdim, cdim, 3), dtype=np.uint8)

    def get_col(i):
        r = int(255 * i/len(points_list))
        g = 255 - r
        return r, g, 0

    for n, p in enumerate(points_list):
        c, r, _ = map(int, p)
        vis[max(0,r-wr):min(rdim,r+wr+1),max(0,c-wc):min(cdim,c+wc+1)] = get_col(n)

    return vis
    
# A color map for paths
col_map = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
           (255,127,0), (255, 0, 127), (127, 255, 0), (0, 255, 127), (127,0,255), (0,127,255)]

def draw_paths(all_paths, foci_stack, foci_index=None, r=3):
    """
    Draws paths on the provided image stack and overlays markers for the foci
    
    Args:
        all_paths (list): List of paths where each path is a list of points.
        foci_stack (np.array): 3D numpy array representing the image stack.
        foci_index (list, optional): List of list of focus indices (along each path). Defaults to None.
        r (int, optional): Radius for the ellipse or line drawing around the focus. Defaults to 3.

    Returns:
        PIL.Image.Image: An image with the drawn paths.
    """
    im = np.max(foci_stack, axis=0)
    im = (im/np.max(im)*255).astype(np.uint8)
    im = np.dstack((im,)*3)
    im = Image.fromarray(im) 
    draw = ImageDraw.Draw(im)
    for i, (p, col) in enumerate(zip(all_paths, cycle(col_map))):
        draw.line([(u[0], u[1]) for u in p], fill=col)
        draw.text((p[0][0], p[0][1]), str(i+1), fill=col)
    if foci_index is not None:
        for i, (idx, p, col) in enumerate(zip(foci_index, all_paths, cycle(col_map))):
            if len(idx):
                for j in idx:
                    draw.line((int(p[j][0]-r), int(p[j][1]), int(p[j][0]+r), int(p[j][1])), fill=col, width=2)
                    draw.line((int(p[j][0]), int(p[j][1]-r), int(p[j][0]), int(p[j][1]+r)), fill=col, width=2)

    return im
    

def measure_from_mask(mask, measure_stack):
    """
    Compute the sum of measure_stack values where the mask is equal to 1.
    
    Args:
        mask (numpy.ndarray): Binary mask where the measurement should be applied.
        measure_stack (numpy.ndarray): Stack of measurements.
        
    Returns:
        measure_stack.dtype: Sum of measure_stack values where the mask is 1.
    """
    return np.sum(mask * measure_stack)

# Max of measure_stack over region where mask==1
def max_from_mask(mask, measure_stack):
    """
    Compute the maximum of measure_stack values where the mask is equal to 1.
    
    Args:
        mask (numpy.ndarray): Binary mask where the measurement should be applied.
        measure_stack (numpy.ndarray): Stack of measurements.
        
    Returns:
        measure_stack.dtype: Maximum value of measure_stack where the mask is 1.
    """
    return np.max(mask * measure_stack)

def make_mask_s(p, melem, measure_stack):
    """
    Translate a mask to point p, ensuring correct treatment near the edges of the measure_stack.
    
    Args:
        p (tuple): Target point (r, c, z).
        melem (numpy.ndarray): Structuring element for the mask.
        measure_stack (numpy.ndarray): Stack of measurements.
        
    Returns:
        tuple: A tuple containing the translated mask and a section of the measure_stack.
    """
    
    
    # 
    
    R = [u//2 for u in melem.shape]
    
    
    r, c, z = p

    mask = np.zeros(melem.shape)

    m_data = np.zeros(melem.shape)
    s = measure_stack.shape
    o_1, o_2, o_3 = max(R[0]-r, 0), max(R[1]-c, 0), max(R[2]-z,0)
    e_1, e_2, e_3 = min(R[0]-r+s[0], 2*R[0]+1), min(R[1]-c+s[1], 2*R[1]+1), min(R[2]-z+s[2], 2*R[2]+1)
    m_data[o_1:e_1,o_2:e_2,o_3:e_3] = measure_stack[max(r-R[0],0):min(r+R[0]+1,s[0]),max(c-R[1],0):min(c+R[1]+1,s[1]),max(z-R[2],0):min(z+R[2]+1, s[2])]
    mask[o_1:e_1,o_2:e_2,o_3:e_3] = melem[o_1:e_1,o_2:e_2,o_3:e_3] 

    
    return mask, m_data


def measure_at_point(p, melem, measure_stack, op='mean'):
    """
    Measure the mean or max value of measure_stack around a specific point using a structuring element.
    
    Args:
        p (tuple): Target point (r, c, z).
        melem (numpy.ndarray): Structuring element for the mask.
        measure_stack (numpy.ndarray): Stack of measurements.
        op (str, optional): Operation to be applied; either 'mean' or 'max'. Default is 'mean'.
        
    Returns:
        float: Measured value based on the specified operation.
    """

    p = map(int, p)
    if op=='mean':
        mask, m_data = make_mask_s(p, melem, measure_stack)
        melem_size = np.sum(mask)
        return float(measure_from_mask(mask, m_data) / melem_size)
    else:
        mask, m_data = make_mask_s(p, melem, measure_stack)
        return float(max_from_mask(mask, m_data))

# Generate spherical region
def make_sphere(R=5, z_scale_ratio=2.3):
    """
    Generate a binary representation of a sphere in 3D space.
    
    Args:
        R (int, optional): Radius of the sphere. Default is 5.
        z_scale_ratio (float, optional): Scaling factor for the z-axis. Default is 2.3.
        
    Returns:
        numpy.ndarray: Binary representation of the sphere.
    """
    R_z = int(math.ceil(R/z_scale_ratio))
    x, y, z = np.ogrid[-R:R+1, -R:R+1, -R_z:R_z+1]
    sphere = x**2 + y**2 + (z_scale_ratio * z)**2 <= R**2
    return sphere

# Measure the values of measure_stack at each of the points of points_list in turn.
# Measurement is the mean / max (specified by op) on the spherical region about each point
def measure_all_with_sphere(points_list, measure_stack, op='mean', R=5, z_scale_ratio=2.3):
    """
    Measure the values of measure_stack at each point in a list using a spherical region.
    
    Args:
        points_list (list): List of points (r, c, z) to be measured.
        measure_stack (numpy.ndarray): Stack of measurements.
        op (str, optional): Operation to be applied; either 'mean' or 'max'. Default is 'mean'.
        R (int, optional): Radius of the sphere. Default is 5.
        z_scale_ratio (float, optional): Scaling factor for the z-axis. Default is 2.3.
        
    Returns:
        list: List of measured values for each point.
    """
    melem = make_sphere(R, z_scale_ratio)
    measure_func = lambda p: measure_at_point(p, melem, measure_stack, op)
    return list(map(measure_func, points_list))


# Measure fluorescence levels along ordered skeleton
def measure_chrom2(path, hei10, config):
    """
    Measure fluorescence levels along an ordered skeleton.
    
    Args:
        path (list): List of ordered path points (r, c, z).
        hei10 (numpy.ndarray): 3D fluorescence data.
        config (dict): Configuration dictionary containing 'z_res', 'xy_res', and 'sphere_radius' values.
        
    Returns:
        tuple: A tuple containing the visualization, mean measurements, and max measurements along the path.
    """
    scale_ratio = config['z_res']/config['xy_res']
    sphere_xy_radius = int(math.ceil(config['sphere_radius']/config['xy_res']))
    
    vis = visualise_ordering(path, dim=hei10.shape, wr=sphere_xy_radius, wc=sphere_xy_radius)
 
    measurements = measure_all_with_sphere(path, hei10, op='mean', R=sphere_xy_radius, z_scale_ratio=scale_ratio)
    measurements_max = measure_all_with_sphere(path, hei10, op='max', R=sphere_xy_radius, z_scale_ratio=scale_ratio)

    return vis, measurements, measurements_max

def extract_peaks(cell_id, all_paths, path_lengths, measured_traces, config):
    """
    Extract peak information from given traces and compile them into a DataFrame.
    
    Args:
    - cell_id (int or str): Identifier for the cell being analyzed.
    - all_paths (list of lists): Contains ordered path points for multiple paths.
    - path_lengths (list of floats): List containing lengths of each path in all_paths.
    - measured_traces (list of lists): Contains fluorescence measurement values along the paths.
    - config (dict): Configuration dictionary containing:
        - 'peak_threshold': Threshold value to determine a peak in the trace.
        - 'sphere_radius': Radius of the sphere used in fluorescence measurement.

    Returns:
    - pd.DataFrame: DataFrame containing peak information for each path.
    - list of lists: Absolute intensities of the detected foci.
    - list of lists: Index positions of the detected foci.
    - list of lists: Absolute focus intensity threshold for each trace.
    - list of numpy.ndarray: For each trace, distances of each point from start of trace in microns
    """
    
    n_paths = len(all_paths)
    
    data = []
    foci_absolute_intensity, foci_position, foci_position_index, trace_median_intensities, trace_thresholds = analyse_traces(all_paths, path_lengths, measured_traces, config)

    foci_intensities = []
    for path_foci_abs_int, tmi in zip(foci_absolute_intensity, trace_median_intensities):
        foci_intensities.extend(list(path_foci_abs_int - tmi))
        
    mean_intensity = np.mean(foci_intensities)
    trace_positions = []
    
    for i in range(n_paths):

        pl = calculate_path_length_partials(all_paths[i], (config['xy_res'], config['xy_res'], config['z_res']))
        
        path_data = { 'Cell_ID':cell_id,
                      'Trace': i+1,
                      'SNT_trace_length(um)': path_lengths[i],
                      'Measured_trace_length(um)': pl[-1],
                      'Trace_median_intensity': trace_median_intensities[i],
                      'Detection_sphere_radius(um)': config['sphere_radius'],
                      'Foci_ID_threshold': config['peak_threshold'] }
        for j, (idx, u,v) in enumerate(zip(foci_position_index[i], foci_position[i], foci_absolute_intensity[i])):
            if config['use_corrected_positions']:
                path_data[f'Foci_{j+1}_position(um)'] = pl[idx]
            else:
                path_data[f'Foci_{j+1}_position(um)'] = u
            path_data[f'Foci_{j+1}_absolute_intensity'] = v
            path_data[f'Foci_{j+1}_relative_intensity'] = (v - trace_median_intensities[i])/mean_intensity
        data.append(path_data)
        trace_positions.append(pl)
    return pd.DataFrame(data), foci_absolute_intensity, foci_position_index, trace_thresholds, trace_positions


def analyse_paths(cell_id,
                  foci_file,
                  traces_file,
                  config
    ):
    """
    Analyzes paths for the given cell ID using provided foci and trace files.
    
    Args:
        cell_id (int/str): Identifier for the cell.
        foci_file (str): Path to the foci image file.
        traces_file (str): Path to the XML traces file.
        config (dict): Configuration dictionary containing necessary parameters such as resolutions and thresholds.

    Returns:
        tuple: A tuple containing an overlay image of the traces, visualization images for each trace,
               a figure with plotted measurements, and a dataframe with extracted peaks.
    """


    foci_stack = tifffile.imread(foci_file)

    if foci_stack.ndim==2:
        foci_stack = foci_stack[None,:,:]
    
    all_paths, path_lengths = get_paths_from_traces_file(traces_file)

    all_trace_vis = []
    all_m = []
    for p in all_paths:
        vis, m, _ = measure_chrom2(p,foci_stack.transpose(2,1,0), config)
        all_trace_vis.append(vis)
        all_m.append(m)


    extracted_peaks, foci_absolute_intensity, foci_pos_index, trace_thresholds, trace_positions = extract_peaks(cell_id, all_paths, path_lengths, all_m, config)

    
    n_cols = 2
    n_rows = (len(all_paths)+n_cols-1)//n_cols
    fig, ax = plt.subplots(n_rows,n_cols)
    ax = ax.flatten()

    for i, m in enumerate(all_m):
        ax[i].set_title(f'Trace {i+1}')
        ax[i].plot(trace_positions[i], m)
        print(foci_pos_index)
        if len(foci_pos_index[i]):
            ax[i].plot(trace_positions[i][foci_pos_index[i]], np.array(m)[foci_pos_index[i]], 'rx')
            ax[i].set_xlabel('Distance from start (um)')
            ax[i].set_ylabel('Intensity')
            ax[i].axhline(trace_thresholds[i], c='r', ls=':')
    for i in range(len(all_m), n_cols*n_rows):
        ax[i].axis('off')

    plt.tight_layout()
    trace_overlay = draw_paths(all_paths, foci_stack, foci_index=foci_pos_index)

    return trace_overlay, all_trace_vis, fig, extracted_peaks
