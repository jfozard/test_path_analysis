
import gradio as gr
from tifffile import imread
from PIL import Image
import matplotlib.pyplot as plt
from analyse import analyse_paths
import numpy as np

def process(cell_id, foci_file, traces_file):
    paths, traces, fig, extracted_peaks = analyse_paths(cell_id, foci_file.name, traces_file.name)
    extracted_peaks.to_csv('tmp')
    return paths, [Image.fromarray(im) for im in traces], fig, extracted_peaks, 'tmp'

def preview_image(file1):
    if file1:
        im = imread(file1.name)
        print(im.shape)
        return Image.fromarray(np.max(im, axis=0))
    else:
        return None


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            cellid_input = gr.Textbox(label="Cell ID", placeholder="Image_1")
            image_input = gr.File(label="Input foci image")
            image_preview = gr.Image(label="Max projection of foci image")
            image_input.change(fn=preview_image, inputs=image_input, outputs=image_preview)
            path_input = gr.File(label="SNT traces file")

        with gr.Column():
            trace_output = gr.Image(label="Overlayed paths")
            image_output=gr.Gallery(label="Traced paths")
            plot_output=gr.Plot(label="Foci intensity traces")
            data_output=gr.DataFrame(label="Detected peak data")#, "Peak 1 pos", "Peak 1 int"])
            data_file_output=gr.File(label="Output data file (.csv)")

    with gr.Row():
        greet_btn = gr.Button("Process")
        greet_btn.click(fn=process, inputs=[cellid_input, image_input, path_input], outputs=[trace_output, image_output, plot_output, data_output, data_file_output], api_name="process")


if __name__ == "__main__":
    demo.launch()
