

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from scipy.signal import find_peaks
from math import ceil




def thin_peaks(peak_list, dmin=10, voxel_size=(1,1,1), return_larger_peaks=False):
    """
    Remove peaks within a specified distance of each other, retaining the peak with the highest intensity.

    Args:
    - peak_list (list of PeakData): Each element contains:
        - pos (list of float): 3D coordinates of the peak.
        - intensity (float): The intensity value of the peak.
        - key (tuple): A unique identifier or index for the peak (#trace, #peak)
    - dmin (float, optional): Minimum distance between peaks. peaks closer than this threshold will be thinned. Defaults to 10.
    - return_larger_peaks (bool, optional): Indicate larger peak for each thinned peak

    Returns:
    - list of tuples: A list containing keys of the removed peaks.
    if return_larger_peaks
    - list of tuples: A list containing the keys of the larger peak causing the peak to be removed

    Notes:
    - The function uses the L2 norm (Euclidean distance) to compute the distance between peaks.
    - When two peaks are within `dmin` distance, the peak with the lower intensity is removed.
    """
    removed_peaks = []
    removed_larger_peaks = []
    for i in range(len(peak_list)):
        if peak_list[i].key in removed_peaks:
            continue
        for j in range(len(peak_list)):
            if i==j:
                continue
            if peak_list[j].key in removed_peaks:
                continue
            d = (np.array(peak_list[i].pos) - np.array(peak_list[j].pos))*np.array(voxel_size)
            d = la.norm(d)
            if d<dmin:
                hi = peak_list[i].intensity
                hj = peak_list[j].intensity
                if hi<hj:
                    removed_peaks.append(peak_list[i].key)
                    removed_larger_peaks.append(peak_list[j].key)
                    break
                else:
                    removed_peaks.append(peak_list[j].key)
                    removed_larger_peaks.append(peak_list[i].key)

    if return_larger_peaks:
        return removed_peaks, removed_larger_peaks
    else:
        return removed_peaks


@dataclass
class CellData(object):
    """Represents data related to a single cell.

    Attributes:
        pathdata_list (list): A list of PathData objects representing the various paths associated with the cell.
    """
    pathdata_list: list

@dataclass
class RemovedPeakData(object):
    """Represents data related to a removed peak

    Attributes:
        idx (int): Index of peak along path
        dominating_peak (tuple): (path_idx, position along path) for dominating peak
    """
    idx: int 
    dominating_peak: tuple 

@dataclass
class PathData(object):
    """Represents data related to a specific path in the cell.

    This dataclass encapsulates information about the peaks, 
    the defining points, the fluorescence values, and the path length of a specific path.

    Attributes: peaks (list): List of peaks in the path (indicies of positions in points, o_hei10).
        removed_peaks (list): List of peaks in the path which have been removed because of a nearby larger peak
        points (list): List of points defining the path.
        o_hei10 (list): List of (unnormalized) fluorescence intensity values along the path
        SC_length (float): Length of the path.

    """
    peaks: list
    removed_peaks: list
    points: list
    o_hei10: list
    SC_length: float

@dataclass
class PeakData(object):
    pos: tuple
    intensity: float
    key: tuple


def find_peaks2(v, distance=5,  prominence=0.5):
    """
    Find peaks in a 1D array with extended boundary handling.

    The function pads the input array at both ends to handle boundary peaks. It then identifies peaks in the extended array
    and maps them back to the original input array.

    Args:
    - v (numpy.ndarray): 1D input array in which to find peaks.
    - distance (int, optional): Minimum number of array elements that separate two peaks. Defaults to 5.
    - prominence (float, optional): Minimum prominence required for a peak to be identified. Defaults to 0.5.

    Returns:
    - list of int: List containing the indices of the identified peaks in the original input array.
    - dict: Information about the properties of the identified peaks (as returned by scipy.signal.find_peaks).

    """
    pad = int(ceil(distance))+1
    v_ext = np.concatenate([np.ones((pad,), dtype=v.dtype)*np.min(v), v, np.ones((pad,), dtype=v.dtype)*np.min(v)])

    assert(len(v_ext) == len(v)+2*pad)
    peaks, _ = find_peaks(v_ext, distance=distance, prominence=prominence)
    peaks = peaks - pad
    n_peaks = []
    for i in peaks:
        if 0<=i<len(v):
            n_peaks.append(i)
        else:
            raise Exception
    return n_peaks, _
          

def process_cell_traces(all_paths, path_lengths, measured_trace_fluorescence):
    """
    Process traces of cells to extract peak information and organize the data.

    The function normalizes fluorescence data, finds peaks, refines peak information, 
    removes unwanted peaks that might be due to close proximity of bright peaks from 
    other paths, and organizes all the information into a structured data format.

    Args:
        all_paths (list of list of tuples): A list containing paths, where each path is 
                                            represented as a list of 3D coordinate tuples.
        path_lengths (list of float): List of path lengths corresponding to the provided paths.
        measured_trace_fluorescence (list of list of float): A list containing fluorescence 
                                                            data corresponding to each path point.

    Returns:
        CellData: An object containing organized peak and path data for a given cell.

    Note:
        - The function assumes that each path and its corresponding length and fluorescence data 
          are positioned at the same index in their respective lists.
    """
    
    cell_peaks = []

    for points, o_hei10 in zip(all_paths, measured_trace_fluorescence):
                     
        # For peak determination normalize each trace to have mean zero and s.d. 1
        hei10_normalized = (o_hei10 - np.mean(o_hei10))/np.std(o_hei10)
        
        # Find peaks - these will be further refined later
        p,_ = find_peaks2(hei10_normalized, distance=5,  prominence=0.5*np.std(hei10_normalized))
        peaks = np.array(p, dtype=np.int32)

        # Store peak data - using original values, not normalized ones
        peak_mean_heights = [ o_hei10[u] for u in peaks ]
        peak_points = [ points[u] for u in peaks ]
        
        cell_peaks.append((peaks, peak_points, peak_mean_heights))
        
    # Eliminate peaks which have another larger peak nearby (in 3D space, on any chromosome).
    # This aims to remove small peaks in the mean intensity generated when an  SC passes close
    # to a bright peak on another SC - this is nearby in space, but brighter.

    to_thin = []
    for k in range(len(cell_peaks)):
        for u in range(len(cell_peaks[k][0])):
            to_thin.append(PeakData(pos=cell_peaks[k][1][u], intensity=cell_peaks[k][2][u], key=(k, u)))
    
    # Exclude any peak with a nearby brighter peak (on any SC)
    removed_peaks, removed_larger_peaks = thin_peaks(to_thin, return_larger_peaks=True)

    # Clean up and remove these peaks
    new_cell_peaks = []
    removed_cell_peaks = []
    removed_cell_peaks_larger = []
    for path_idx in range(len(cell_peaks)):
        path_retained_peaks = []
        path_removed_peaks = []
        path_peaks = cell_peaks[path_idx][0]

        for peak_idx in range(len(path_peaks)):
            if (path_idx, peak_idx) not in removed_peaks:
                path_retained_peaks.append(path_peaks[peak_idx])
            else:
                # What's the larger point?
                idx = removed_peaks.index((path_idx, peak_idx))
                larger_path, larger_idx = removed_larger_peaks[idx]        
                path_removed_peaks.append(RemovedPeakData(idx=path_peaks[peak_idx], dominating_peak=(larger_path, cell_peaks[larger_path][0][larger_idx])))
                ###
                 
        new_cell_peaks.append(path_retained_peaks)
        removed_cell_peaks.append(path_removed_peaks)
        
    cell_peaks = new_cell_peaks
    pd_list = []
    
    # Save peak positions, absolute HEI10 intensities, and length for each SC
    for k in range(len(all_paths)):
        
        points, o_hei10 = all_paths[k], measured_trace_fluorescence[k]

        peaks = cell_peaks[k]
        removed_peaks = removed_cell_peaks[k]
        
        pd = PathData(peaks=peaks, removed_peaks=removed_peaks, points=points, o_hei10=o_hei10, SC_length=path_lengths[k])
        pd_list.append(pd)

    cd = CellData(pathdata_list=pd_list)

    return cd


alpha_max = 0.4


# Criterion used for identifying peak as a focus - normalized (with mean and s.d.)
# hei10 levels being above 0.4 time maximum peak level
def focus_criterion(pos, v, alpha=alpha_max):
    """
    Identify and return positions where values in the array `v` exceed a certain threshold.

    The threshold is computed as `alpha` times the maximum value in `v`.

    Args:
    - pos (numpy.ndarray): Array of positions.
    - v (numpy.ndarray): 1D array of values, e.g., intensities.
    - alpha (float, optional): A scaling factor for the threshold. Defaults to `alpha_max`.

    Returns:
    - numpy.ndarray: Array of positions where corresponding values in `v` exceed the threshold.
    """
    if len(v):
        idx = (v>=alpha*np.max(v))
        return np.array(pos[idx])
    else:
        return np.array([], dtype=np.int32)

def analyse_celldata(cell_data, config):
    """
    Analyse the provided cell data to extract focus-related information.

    Args:
        cd (CellData): An instance of the CellData class containing path data information.
        config (dictionary): Configuration dictionary containing 'peak_threshold' and 'threshold_type'
                             'peak_threshold' (float) - threshold for calling peaks as foci
                             'threshold_type' (str) = 'per-trace', 'per-foci'

    Returns:
        tuple: A tuple containing:
            - foci_rel_intensity (list): List of relative intensities for the detected foci.
            - foci_pos (list): List of absolute positions of the detected foci.
            - foci_pos_index (list): List of indices of the detected foci.
            - dominated_foci_data (list): List of RemovedPeakData indicating positions of removed peaks and the index of the larger peak
            - trace_median_intensities (list): Per-trace median intensity
            - trace_thresholds (list): Per-trace absolute threshold for calling peaks as foci
    """
    foci_abs_intensity = []
    foci_pos = []
    foci_pos_index = []
    dominated_foci_data = []
    trace_median_intensities = []
    trace_thresholds = []
    
    peak_threshold = config['peak_threshold']

    threshold_type = config['threshold_type']

    if threshold_type == 'per-trace':
        """
        Call extracted peaks as foci if intensity - trace_mean > peak_threshold * (trace_max_foci_intensity - trace_mean)
        """
        
        for path_data in cell_data.pathdata_list:
            peaks = np.array(path_data.peaks, dtype=np.int32)

            # Normalize extracted fluorescent intensities by subtracting mean (and dividing
            # by standard deviation - note that the latter should have no effect on the results).
            h = np.array(path_data.o_hei10)
            h = h - np.mean(h)
            h = h/np.std(h)
            # Extract foci according to criterion
            foci_idx = focus_criterion(peaks, h[peaks], peak_threshold) 
            print('peaks', peaks, h[peaks], foci_idx, np.mean(path_data.o_hei10))
            
            #
            removed_peaks = path_data.removed_peaks
            removed_peaks_idx = np.array([u.idx for u in removed_peaks], dtype=np.int32)

            
            if len(peaks):
                trace_thresholds.append((1-peak_threshold)*np.mean(path_data.o_hei10) + peak_threshold*np.max(np.array(path_data.o_hei10)[peaks]))
            else:
                trace_thresholds.append(None)

            if len(removed_peaks):
                if len(peaks):
                    threshold = (1-peak_threshold)*np.mean(path_data.o_hei10) + peak_threshold*np.max(np.array(path_data.o_hei10)[peaks])
                else:
                    threshold = float('-inf')

                
                removed_peak_heights = np.array(path_data.o_hei10)[removed_peaks_idx]
                dominated_foci_idx = np.where(removed_peak_heights>threshold)[0]
                
                dominated_foci_data.append([removed_peaks[i] for i in dominated_foci_idx])
            else:
                dominated_foci_data.append([])
                
            pos_abs = (foci_idx/len(path_data.points))*path_data.SC_length
            foci_pos.append(pos_abs)
            foci_abs_intensity.append(np.array(path_data.o_hei10)[foci_idx])
            
            foci_pos_index.append(foci_idx)
            trace_median_intensities.append(np.median(path_data.o_hei10))
            
    elif threshold_type == 'per-cell':
        """
        Call extracted peaks as foci if intensity - trace_mean > peak_threshold * max(intensity - trace_mean)
        """
        max_cell_intensity = float("-inf")
        for path_data in cell_data.pathdata_list:

            # Normalize extracted fluorescent intensities by subtracting mean (and dividing
            # by standard deviation - note that the latter should have no effect on the results).
            h = np.array(path_data.o_hei10)
            h = h - np.mean(h)
            max_cell_intensity = max(max_cell_intensity, np.max(h))

        for path_data in cell_data.pathdata_list:
            peaks = np.array(path_data.peaks, dtype=np.int32)

            # Normalize extracted fluorescent intensities by subtracting mean (and dividing
            # by standard deviation - note that the latter should have no effect on the results).
            h = np.array(path_data.o_hei10)
            h = h - np.mean(h)

            foci_idx = peaks[h[peaks]>peak_threshold*max_cell_intensity]

            removed_peaks = path_data.removed_peaks
            removed_peaks_idx = np.array([u.idx for u in removed_peaks], dtype=np.int32)

            trace_thresholds.append(np.mean(path_data.o_hei10) + peak_threshold*max_cell_intensity)

            if len(removed_peaks):
                threshold = np.mean(path_data.o_hei10) + peak_threshold*max_cell_intensity

                removed_peak_heights = np.array(path_data.o_hei10)[removed_peaks_idx]
                dominated_foci_idx = np.where(removed_peak_heights>threshold)[0]
                
                dominated_foci_data.append([removed_peaks[i] for i in dominated_foci_idx])
            else:
                dominated_foci_data.append([])

            pos_abs = (foci_idx/len(path_data.points))*path_data.SC_length
            foci_pos.append(pos_abs)
            foci_abs_intensity.append(np.array(path_data.o_hei10)[foci_idx])
            
            foci_pos_index.append(foci_idx)
            trace_median_intensities.append(np.median(path_data.o_hei10))            
            
    else:
        raise NotImplementedError
    
    return foci_abs_intensity, foci_pos, foci_pos_index, dominated_foci_data, trace_median_intensities, trace_thresholds

def analyse_traces(all_paths, path_lengths, measured_trace_fluorescence, config):
    
    cd = process_cell_traces(all_paths, path_lengths, measured_trace_fluorescence)

    return analyse_celldata(cd, config)

    


