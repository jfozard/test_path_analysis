
import pytest
from path_analysis.analyse import *
from path_analysis.data_preprocess import RemovedPeakData
import numpy as np
from math import pi
import xml.etree.ElementTree as ET
from PIL import ImageChops

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

@pytest.fixture(scope="module")
def script_loc(request):
    '''Return the directory of the currently running test script'''

    return Path(request.fspath).parent 

def test_image_1(script_loc):
            
    config = {  'sphere_radius': 0.1984125,
                'peak_threshold': 0.4,
                'xy_res': 0.0396825,
                'z_res': 0.0909184,
                'threshold_type': 'per-cell',
                'use_corrected_positions': True,
                'screening_distance': 10,
            }

    data_loc = script_loc.parent.parent / 'test_data' / 'hei10 ++ 15.11.19 p22s2 image 9'


    image_input = data_loc / 'HEI10.tif'
    path_input = data_loc / 'SNT_Data.traces'

    paths, traces, fig, extracted_peaks = analyse_paths('Cell', image_input, path_input, config)

    assert np.allclose(extracted_peaks['SNT_trace_length(um)'], [61.47, 70.40, 51.93, 43.94, 62.24], atol=1e-2 )
    assert np.allclose(extracted_peaks['SNT_trace_length(um)'], extracted_peaks['Measured_trace_length(um)'], atol=1e-8 )
    assert list(extracted_peaks['Trace_foci_number']) == [2,3,2,2,3]

def test_image_2(script_loc):
            
    config = {  'sphere_radius': 0.1984125,
                'peak_threshold': 0.4,
                'xy_res': 0.0396825,
                'z_res': 0.0909184,
                'threshold_type': 'per-cell',
                'use_corrected_positions': True,
                'screening_distance': 10,
              }

    data_loc = script_loc.parent.parent / 'test_data' / 'z-optimised'


    image_input = data_loc / 'HEI10.tif'
    path_input = data_loc / 'ZYP1.traces'

    paths, traces, fig, extracted_peaks = analyse_paths('Cell', image_input, path_input, config)

    assert np.allclose(extracted_peaks['SNT_trace_length(um)'], extracted_peaks['Measured_trace_length(um)'], atol=1e-8 )
    assert list(extracted_peaks['Trace_foci_number']) == [2,2,1,2,1]
    
def test_image_3(script_loc):
            
    config = {  'sphere_radius': 0.1984125,
                'peak_threshold': 0.4,
                'xy_res': 0.0396825,
                'z_res': 0.1095510,
                'threshold_type': 'per-trace',
                'use_corrected_positions': True,
                'screening_distance': 10,
                
            }

    data_loc = script_loc.parent.parent / 'test_data' / 'arenosa SN A1243 image 18-20230726T142725Z-001' / 'arenosa SN A1243 image 18'


    image_input = data_loc / 'HEI10.tif'
    path_input = data_loc / 'SNT_Data.traces'

    paths, traces, fig, extracted_peaks = analyse_paths('Cell', image_input, path_input, config)

    assert np.allclose(extracted_peaks['SNT_trace_length(um)'], extracted_peaks['Measured_trace_length(um)'], atol=1e-8 )
    assert list(extracted_peaks['Trace_foci_number']) == [2,1,1,1,2,1,1,1]

def test_image_4(script_loc):
            
    config = {  'sphere_radius': 10.,
                'peak_threshold': 0.4,
                'xy_res': 1,
                'z_res': 1,
                'threshold_type': 'per-trace',
                'use_corrected_positions': True,
                'screening_distance': 10,

            }

    data_loc = script_loc.parent.parent / 'test_data' / 'mammalian 2D-20230821T180708Z-001' / 'mammalian 2D' / '1' 


    image_input = data_loc / 'C2-Pachytene SIM-1.tif'
    path_input = data_loc / 'SNT_Data.traces'

    paths, traces, fig, extracted_peaks = analyse_paths('Cell', image_input, path_input, config)

    assert np.allclose(extracted_peaks['SNT_trace_length(um)'], extracted_peaks['Measured_trace_length(um)'], atol=1e-8 )

    valid_results = [{1}, {1}, {2, 3}, {1, 2}, {1, 2}, {1}, {1}, {2}, {1}, {1}, {1, 2}, {1}, {1, 2}, {1, 2}, {1}, {1}, {1}, {1}, {1}]
    measured = extracted_peaks['Trace_foci_number']

    print(measured)
    assert len(measured) == len(valid_results)
    assert(all(m in v for m,v in zip(measured, valid_results)))



def test_image_5(script_loc):
            
    config = {  'sphere_radius': 0.3,
                'peak_threshold': 0.4,
                'xy_res': 0.1023810,
                'z_res': 1,
                'threshold_type': 'per-trace',
                'use_corrected_positions': True,
                'screening_distance': 10,

            }

    data_loc = script_loc.parent.parent / 'test_data' / 'mammalian 2D-20230821T180708Z-001' / 'mammalian 2D' / '2' 


    image_input = data_loc / 'C1-CNTD1FHFH CSHA 1in5000 22612 Slide 6-102-1.tif'
    path_input = data_loc / 'SNT_Data.traces'

    paths, traces, fig, extracted_peaks = analyse_paths('Cell', image_input, path_input, config)

    assert np.allclose(extracted_peaks['SNT_trace_length(um)'], extracted_peaks['Measured_trace_length(um)'], atol=1e-8 )

    valid_results = [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1]
    measured = extracted_peaks['Trace_foci_number']

    assert list(measured) == valid_results


    


