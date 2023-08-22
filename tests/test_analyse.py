
from path_analysis.analyse import *
import numpy as np
from math import pi
import xml.etree.ElementTree as ET


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
