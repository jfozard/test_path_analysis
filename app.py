
import gradio as gr
from tifffile import imread
from PIL import Image
from path_analysis.analyse import analyse_paths
import numpy as np


# Function to preview the imported image
def preview_image(file1):
    if file1:
        im = imread(file1.name)
        print(im.ndim, im.shape)
        if im.ndim>2:
            return Image.fromarray(np.max(im, axis=0))
        else:
            return Image.fromarray(im)
    else:
        return None


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Inputs for cell ID, image, and path
            cellid_input = gr.Textbox(label="Cell ID", placeholder="Image_1")
            image_input = gr.File(label="Input foci image")
            image_preview = gr.Image(label="Max projection of foci image")
            image_input.change(fn=preview_image, inputs=image_input, outputs=image_preview)
            path_input = gr.File(label="SNT traces file")

            # Additional options wrapped in an accordion for better UI experience
            with gr.Accordion("Additional options ..."):
                sphere_radius = gr.Number(label="Trace sphere radius (um)", value=0.1984125, interactive=True)
                peak_threshold = gr.Number(label="Peak relative threshold", value=0.4, interactive=True)
                # Resolutions for xy and z axis
                with gr.Row():
                    xy_res = gr.Number(label='xy-yesolution (um)', value=0.0396825, interactive=True)
                    z_res = gr.Number(label='z resolution (um)', value=0.0909184, interactive=True)
                # Resolutions for xy and z axis

                threshold_type = gr.Radio(["per-trace", "per-cell"], label="Threshold-type", value="per-trace", interactive=True)

                 
        # The output column showing the result of processing            
        with gr.Column():
            trace_output = gr.Image(label="Overlayed paths")
            image_output=gr.Gallery(label="Traced paths")
            plot_output=gr.Plot(label="Foci intensity traces")
            data_output=gr.DataFrame(label="Detected peak data")#, "Peak 1 pos", "Peak 1 int"])
            data_file_output=gr.File(label="Output data file (.csv)")


    def process(cellid_input, image_input, path_input, sphere_radius, peak_threshold, xy_res, z_res, threshold_type):

        config = { 'sphere_radius': sphere_radius,
                   'peak_threshold': peak_threshold,
                   'xy_res': xy_res,
                   'z_res': z_res,
                   'threshold_type': threshold_type }
                   
        
        paths, traces, fig, extracted_peaks = analyse_paths(cellid_input, image_input.name, path_input.name, config)
        extracted_peaks.to_csv('output.csv')
        print('extracted', extracted_peaks)
        return paths, [Image.fromarray(im) for im in traces], fig, extracted_peaks, 'output.csv'

            
    with gr.Row():
        greet_btn = gr.Button("Process")
        greet_btn.click(fn=process, inputs=[cellid_input, image_input, path_input, sphere_radius, peak_threshold, xy_res, z_res, threshold_type], outputs=[trace_output, image_output, plot_output, data_output, data_file_output], api_name="process")


if __name__ == "__main__":
    demo.launch()
