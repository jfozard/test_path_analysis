
import pytest
from path_analysis.analyse import *
import numpy as np
from math import pi
import xml.etree.ElementTree as ET
from PIL import ImageChops

def test_draw_paths_no_error():
    all_paths = [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    foci_stack = np.zeros((5, 5, 5))
    foci_stack[0,0,0] = 1.0
    foci_index = [[0], [1]]
    r = 3

    try:
        im = draw_paths(all_paths, foci_stack, foci_index, r)
    except Exception as e:
        pytest.fail(f"draw_paths raised an exception: {e}")

def test_draw_paths_image_size():
    all_paths = [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    foci_stack = np.zeros((5, 5, 5))
    foci_stack[0,0,0] = 1.0

    foci_index = [[0], [1]]
    r = 3

    im = draw_paths(all_paths, foci_stack, foci_index, r)
    assert im.size == (5, 5), f"Expected image size (5, 5), got {im.size}"

def test_draw_paths_image_modified():
    all_paths = [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
    foci_stack = np.zeros((5, 5, 5))
    foci_stack[0,0,0] = 1.0
    foci_index = [[0], [1]]
    r = 3

    im = draw_paths(all_paths, foci_stack, foci_index, r)
    blank_image = Image.new("RGB", (5, 5), "black")

    # Check if the image is not entirely black (i.e., has been modified)
    diff = ImageChops.difference(im, blank_image)
    assert diff.getbbox() is not None, "The image has not been modified"



def test_calculate_path_length_partials_default_voxel():
    point_list = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]
    expected_result = np.array([0.0, 1.0, 1.0+np.sqrt(2)])
    result = calculate_path_length_partials(point_list)
    np.testing.assert_allclose(result, expected_result, atol=1e-5)

def test_calculate_path_length_partials_custom_voxel():
    point_list = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    voxel_size = (1, 2, 1)
    expected_result = np.array([0.0, 1.0, 3.0])
    result = calculate_path_length_partials(point_list, voxel_size=voxel_size)
    np.testing.assert_allclose(result, expected_result, atol=1e-5)

def test_calculate_path_length_partials_single_point():
    point_list = [(0, 0, 0)]
    expected_result = np.array([0.0])
    result = calculate_path_length_partials(point_list)
    np.testing.assert_allclose(result, expected_result, atol=1e-5)

def test_get_paths_from_traces_file():
    # Mock the XML traces file content
    xml_content = '''<?xml version="1.0"?>
    <root>
        <path reallength="5.0">
            <point x="1" y="2" z="3"/>
            <point x="4" y="5" z="6"/>
        </path>
        <path reallength="10.0">
            <point x="7" y="8" z="9"/>
            <point x="10" y="11" z="12"/>
        </path>
    </root>
    '''
    
    # Create a temporary XML file
    with open("temp_traces.xml", "w") as f:
        f.write(xml_content)
    
    all_paths, path_lengths = get_paths_from_traces_file("temp_traces.xml")
    
    expected_paths = [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]]
    expected_lengths = [5.0, 10.0]
    
    assert all_paths == expected_paths, f"Expected paths {expected_paths}, but got {all_paths}"
    assert path_lengths == expected_lengths, f"Expected lengths {expected_lengths}, but got {path_lengths}"

    # Clean up temporary file
    import os
    os.remove("temp_traces.xml")
    
    
def test_measure_chrom2():
    # Mock data
    path = [(2, 3, 4), (4, 5, 6), (9, 9, 9)]  # Sample ordered path points
    hei10 = np.random.rand(10, 10, 10)  # Random 3D fluorescence data
    config = {
        'z_res': 1,
        'xy_res': 0.5,
        'sphere_radius': 2.5
    }

    # Function call
    _, measurements, measurements_max = measure_chrom2(path, hei10, config)
    
    # Assertions
    assert len(measurements) == len(path), "Measurements length should match path length"
    assert len(measurements_max) == len(path), "Max measurements length should match path length"
    assert all(0 <= val <= 1 for val in measurements), "All mean measurements should be between 0 and 1 for this mock data"
    assert all(0 <= val <= 1 for val in measurements_max), "All max measurements should be between 0 and 1 for this mock data"

def test_measure_chrom2_z():
    # Mock data
    path = [(2, 3, 4), (4, 5, 6)]  # Sample ordered path points
    _,_,hei10 = np.meshgrid(np.arange(10), np.arange(10), np.arange(10))  # 3D fluorescence data - z dependent
    config = {
        'z_res': 1,
        'xy_res': 0.5,
        'sphere_radius': 2.5
    }

    # Function call
    _, measurements, measurements_max = measure_chrom2(path, hei10, config)
    
    # Assertions
    assert len(measurements) == len(path), "Measurements length should match path length"
    assert len(measurements_max) == len(path), "Max measurements length should match path length"
    assert all(measurements == np.array([4,6])) 
    assert all(measurements_max == np.array([6,8])) 

def test_measure_chrom2_z2():
    # Mock data
    path = [(0,0,0), (2, 3, 4), (4, 5, 6)]  # Sample ordered path points
    _,_,hei10 = np.meshgrid(np.arange(10), np.arange(10), np.arange(10))  # 3D fluorescence data - z dependent
    config = {
        'z_res': 0.25,
        'xy_res': 0.5,
        'sphere_radius': 2.5
    }

    # Function call
    _, measurements, measurements_max = measure_chrom2(path, hei10, config)
    
    # Assertions
    assert len(measurements) == len(path), "Measurements length should match path length"
    assert len(measurements_max) == len(path), "Max measurements length should match path length"
    assert all(measurements_max == np.array([9,9,9])) 

    
def test_measure_from_mask():
    mask = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    measure_stack = np.array([
        [2, 4, 2],
        [4, 8, 4],
        [2, 4, 2]
    ])
    result = measure_from_mask(mask, measure_stack)
    assert result == 24  # Expected sum: 4+4+8+4+4
        
def test_max_from_mask():
    mask = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    measure_stack = np.array([
        [2, 5, 2],
        [4, 8, 3],
        [2, 7, 2]
    ])
    result = max_from_mask(mask, measure_stack)
    assert result == 8  # Expected max: 8
        

def test_measure_at_point_mean():
    measure_stack = np.array([
        [[2, 2, 2, 0], [4, 4, 6, 0], [3, 3, 2, 0], [0, 0, 0, 0]],
        [[4, 4, 4, 0], [8, 8, 8, 0], [4, 4, 4, 0], [0, 0, 0, 0]],
        [[3, 3, 3, 0], [6, 6, 4, 0], [3, 2, 2, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    p = (1, 1, 1)
    melem = np.ones((3, 3, 3))
    result = measure_at_point(p, melem, measure_stack, op='mean')
    assert result == 4, "Expected mean: 4"

def test_measure_at_point_mean_off1():
    measure_stack = np.array([
        [[2, 2, 2, 0], [4, 4, 6, 0], [5, 5, 2, 0], [0, 0, 0, 0]],
        [[4, 4, 4, 0], [8, 8, 8, 0], [4, 4, 4, 0], [0, 0, 0, 0]],
        [[3, 3, 3, 0], [6, 6, 4, 0], [3, 2, 2, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    p = (0, 0, 0)
    melem = np.ones((3, 3, 3))
    result = measure_at_point(p, melem, measure_stack, op='mean')
    assert result == 4.5,  "Expected mean: 4.5"

def test_measure_at_point_mean_off2():
    measure_stack = np.array([
        [[2, 2, 2, 0], [4, 4, 6, 0], [5, 5, 2, 0], [0, 0, 0, 0]],
        [[4, 4, 4, 0], [8, 8, 8, 0], [4, 4, 4, 0], [0, 0, 0, 0]],
        [[3, 3, 3, 0], [6, 6, 4, 0], [3, 2, 2, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    p = (3, 1, 1)
    melem = np.ones((3, 3, 3))
    print(measure_stack[p[0], p[1], p[2]])
    
    result = measure_at_point(p, melem, measure_stack, op='mean')
    assert result == 32/18  # Expected mean: 4.5
    
def test_measure_at_point_mean_off3():
    measure_stack = np.array([
        [[2, 2, 2, 0], [4, 4, 6, 0], [5, 5, 2, 0], [0, 0, 0, 0]],
        [[4, 4, 4, 0], [8, 8, 8, 0], [4, 4, 4, 0], [0, 0, 0, 0]],
        [[3, 3, 3, 0], [6, 6, 4, 0], [3, 2, 2, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    p = (3, 1, 1)
    melem = np.ones((1, 1, 3))
    print(measure_stack[p[0], p[1], p[2]])
    
    result = measure_at_point(p, melem, measure_stack, op='mean')
    assert result == 0,  "Expected mean: 4.5"
    
def test_measure_at_point_mean_off3():
    measure_stack = np.array([
        [[2, 2, 2, 0], [4, 4, 6, 0], [5, 5, 2, 0], [0, 0, 0, 0]],
        [[4, 4, 4, 0], [8, 8, 8, 0], [4, 4, 4, 0], [0, 0, 0, 0]],
        [[3, 3, 3, 0], [6, 6, 4, 0], [3, 2, 2, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ])
    p = (3, 1, 1)
    melem = np.ones((3, 1, 1))
    print(measure_stack[p[0], p[1], p[2]])
    
    result = measure_at_point(p, melem, measure_stack, op='mean')
    assert result == 3, "Expected mean: 4.5"
    
    
def test_measure_at_point_max():
    measure_stack = np.array([
        [[2, 2, 2], [4, 4, 4], [2, 2, 2]],
        [[4, 5, 4], [8, 7, 9], [4, 4, 4]],
        [[2, 2, 2], [4, 4, 4], [2, 2, 2]]
    ])
    p = (1, 1, 1)
    melem = np.ones((3, 3, 3))
    result = measure_at_point(p, melem, measure_stack, op='max')
    assert result == 9, "Expected max: 9"
    

def test_make_sphere_equal():
    R = 5
    z_scale_ratio = 1.0
    
    sphere = make_sphere(R, z_scale_ratio)
    
    # Check the returned type
    assert isinstance(sphere, np.ndarray), "Output should be a numpy ndarray"
    
    # Check the shape
    expected_shape = (2*R+1, 2*R+1, 2*R+1)
    assert sphere.shape == expected_shape, f"Expected shape {expected_shape}, but got {sphere.shape}"
    
    assert (sphere[:,:,::-1] == sphere).all(), f"Expected symmetrical mask"
    assert (sphere[:,::-1,:] == sphere).all(), f"Expected symmetrical mask"
    assert (sphere[::-1,:,:] == sphere).all(), f"Expected symmetrical mask"
    assert abs(np.sum(sphere)-4/3*pi*R**3)<10, f"Expected approximate volume to be correct"
    assert (sphere[R,R,0] == 1), f"Expected centre point on top plane to be within sphere"
    assert (sphere[R+1,R,0] == 0), f"Expected point next to centre on top plane to be outside sphere"

import pandas as pd


# 1. Test basic functionality
def test_extract_peaks_basic():
    cell_id = 1
    all_paths = [[[0, 0], [1, 1]]]
    path_lengths = [1.41]  # length of the above path
    measured_traces = [[100, 200]]  # fluorescence along the path
    config = {'peak_threshold': 0.4, 'sphere_radius': 2, 'xy_res': 1, 'z_res': 1, 'use_corrected_positions': True}
    
    df, foci_abs_int, foci_pos_idx, _, _, _ = extract_peaks(cell_id, all_paths, path_lengths, measured_traces, config)
    
    # Now add your assertions to validate the result
    assert len(df) == 1, "Expected one row in DataFrame"
    assert df['Cell_ID'].iloc[0] == cell_id, "Unexpected cell_id"
    # Add more assertions here based on expected values

# 2. Test multiple paths
def test_extract_peaks_multiple_paths():
    cell_id = 1
    all_paths = [[[0, 0], [1, 1]], [[1, 1], [2, 2]]]
    path_lengths = [1.41, 1.41]
    measured_traces = [[100, 200], [100, 150]]
    config = {'peak_threshold': 0.4, 'sphere_radius': 2, 'xy_res': 1, 'z_res': 1, 'use_corrected_positions': True}

    df, _, _, _, _, _ = extract_peaks(cell_id, all_paths, path_lengths, measured_traces, config)
    
    assert len(df) == 2, "Expected two rows in DataFrame"
    # Add more assertions here
