from path_analysis.data_preprocess import *
import numpy as np
import pytest


def test_thin_points():
    # Define a sample point list
    points = [
        PeakData([0, 0, 0], 10, 0),
        PeakData([1, 1, 1], 8, 1),
        PeakData([10, 10, 10], 12, 2),
        PeakData([10.5, 10.5, 10.5], 5, 3),
        PeakData([20, 20, 20], 15, 4)
    ]
    
    # Call the thin_points function with dmin=5 (for example)
    removed_indices = thin_peaks(points, dmin=5)
    
    # Check results
    # Point at index 1 ([1, 1, 1]) should be removed since it's within 5 units distance of point at index 0 and has lower intensity.
    # Similarly, point at index 3 ([10.5, 10.5, 10.5]) should be removed as it's close to point at index 2 and has lower intensity.
    assert set(removed_indices) == {1, 3}

    # Another simple test to check if function does nothing when points are far apart
    far_points = [
        PeakData([0, 0, 0], 10, 0),
        PeakData([100, 100, 100], 12, 1),
        PeakData([200, 200, 200], 15, 2)
    ]
    
    removed_indices_far = thin_peaks(far_points, dmin=5)
    assert len(removed_indices_far) == 0  # Expect no points to be removed


def test_find_peaks2():

    # Basic test
    data = np.array([0, 0, 0, 0, 0, 0, 5, 0, 3, 0])
    peaks, _ = find_peaks2(data)
    assert set(peaks) == {6}  # Expected peaks at positions 6

    # Basic test
    data = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    peaks, _ = find_peaks2(data)
    assert set(peaks) == {1}  # Expected peaks at positions 1


    # Test with padding impacting peak detection
    data = np.array([3, 2.9, 0, 0, 0, 3])
    peaks, _ = find_peaks2(data)
    assert set(peaks) == {0,5}  # Peaks at both ends

    # Test with close peaks
    data = np.array([3, 0, 3])
    peaks, _ = find_peaks2(data)
    assert set(peaks) == {2}  # Peak at right end only
    # Test with close peaks
    
    
    # Test with close peaks
    data = np.array([3, 0, 3])
    peaks, _ = find_peaks2(data, distance=1)
    assert set(peaks) == {0,2}  # Peaks at both ends

    # Test with close peaks
    data = np.array([0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3])
    peaks, _ = find_peaks2(data, distance=1)
    assert set(peaks) == {2,7}  # Peak at centre (rounded to the left) of groups of maximum values

    # Test with prominence threshold
    data = np.array([0, 1, 0, 0.4, 0])
    peaks, _ = find_peaks2(data, prominence=0.5)
    assert peaks == [1]  # Only the peak at position 1 meets the prominence threshold


def test_focus_criterion():
    pos = np.array([0, 1, 2, 3, 4, 6])
    values = np.array([0.1, 0.5, 0.2, 0.8, 0.3, 0.9])

    # Basic test
    assert np.array_equal(focus_criterion(pos, values), np.array([1, 3, 6]))  # only values 0.8 and 0.9 exceed 0.4 times the max (which is 0.9)

    # Empty test
    assert np.array_equal(focus_criterion(np.array([]), np.array([])), np.array([]))

    # Test with custom alpha
    assert np.array_equal(focus_criterion(pos, values, alpha=0.5), np.array([1, 3, 6]))

    # Test with a larger alpha
    assert np.array_equal(focus_criterion(pos, values, alpha=1.0), [6])  # No values exceed the maximum value itself

    # Test with all values below threshold
    values = np.array([0.1, 0.2, 0.3, 0.4])

    assert np.array_equal(focus_criterion(pos[:4], values), [1,2,3])  # All values are below 0.4 times the max (which is 0.4)
 
@pytest.fixture
def mock_data():
    all_paths = [ [ (0,0,0), (0,2,0), (0,5,0), (0,10,0), (0,15,0), (0,20,0)], [ (1,20,0), (1,20,10), (1,20,20)  ] ] # Mock paths
    path_lengths = [ 2.2, 2.3 ]  # Mock path lengths
    measured_trace_fluorescence = [ [100, 8, 3, 2, 3, 49], [38, 2, 20] ]  # Mock fluorescence data
    return all_paths, path_lengths, measured_trace_fluorescence

def test_process_cell_traces_return_type(mock_data):
    all_paths, path_lengths, measured_trace_fluorescence = mock_data
    result = process_cell_traces(all_paths, path_lengths, measured_trace_fluorescence)
    assert isinstance(result, CellData), f"Expected CellData but got {type(result)}"

def test_process_cell_traces_pathdata_list_length(mock_data):
    all_paths, path_lengths, measured_trace_fluorescence = mock_data
    result = process_cell_traces(all_paths, path_lengths, measured_trace_fluorescence)
    assert len(result.pathdata_list) == len(all_paths), f"Expected {len(all_paths)} but got {len(result.pathdata_list)}"
    
def test_process_cell_traces_pathdata_path_lengths(mock_data):
    all_paths, path_lengths, measured_trace_fluorescence = mock_data
    result = process_cell_traces(all_paths, path_lengths, measured_trace_fluorescence)
    path_lengths = [p.SC_length for p in result.pathdata_list]
    expected_path_lengths = [2.2, 2.3]
    assert  path_lengths == expected_path_lengths, f"Expected {expected_path_lengths} but got {path_lengths}"
    
def test_process_cell_traces_peaks(mock_data):
    all_paths, path_lengths, measured_trace_fluorescence = mock_data
    result = process_cell_traces(all_paths, path_lengths, measured_trace_fluorescence)
    print(result)
    peaks = [p.peaks for p in result.pathdata_list]
    assert peaks == [[0,5],[]]
    
# Mock data
@pytest.fixture
def mock_celldata():
    pathdata1 = PathData(peaks=[0, 5], points=[(0,0,0), (0,2,0), (0,5,0), (0,10,0), (0,15,0), (0,20,0)], removed_peaks=[], o_intensity=[100, 8, 3, 2, 3, 69], SC_length=2.2)
    pathdata2 = PathData(peaks=[2], points=[(1,20,0), (1,20,10), (1,20,20) ], removed_peaks=[RemovedPeakData(0, (0,5))], o_intensity=[38, 2, 20], SC_length=2.3)
    return CellData(pathdata_list=[pathdata1, pathdata2])

def test_analyse_celldata(mock_celldata):
    data_frame, foci_absolute_intensity, foci_position_index, dominated_foci_data, trace_median_intensity, trace_thresholds = analyse_celldata(mock_celldata, {'peak_threshold': 0.4, 'threshold_type':'per-trace'})
    assert len(data_frame) == len(mock_celldata.pathdata_list), "Mismatch in dataframe length"
    assert len(foci_absolute_intensity) == len(mock_celldata.pathdata_list), "Mismatch in relative intensities length"
    assert len(foci_position_index) == len(mock_celldata.pathdata_list), "Mismatch in positions length"

    assert list(map(list, foci_position_index)) == [[0, 5], [2]]


def test_analyse_celldata_per_cell(mock_celldata):
    data_frame, foci_absolute_intensity, foci_position_index, dominated_foci_data, trace_median_intensity, trace_thresholds = analyse_celldata(mock_celldata, {'peak_threshold': 0.4, 'threshold_type':'per-cell'})
    assert len(data_frame) == len(mock_celldata.pathdata_list), "Mismatch in relative intensities length"
    assert len(foci_absolute_intensity) == len(mock_celldata.pathdata_list), "Mismatch in positions length"
    assert len(foci_position_index) == len(mock_celldata.pathdata_list), "Mismatch in position indices length"
    assert list(map(list, foci_position_index)) == [[0, 5], []]

