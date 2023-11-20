
# Gradio tool for quantifying foci on meiotic chromosomes

## Local Use (Python)

Install 
``` pip install -r (requirements.txt).```

To run demo
``` python app.py ```

Browse to http://127.0.0.1:7860

## Local use (Docker container)

Docker container - create from Dockerfile

### On Linux

Clone the repository

```
git clone https://github.com/jfozard/test_path_analysis
cd test_path_analysis
```

Build
```
docker build -t meioquant .
```

Run
```
docker run -dp 127.0.0.1:7860:7860 meioquant
```

Browse to http://127.0.0.1:7860

### On Windows

(TODO)

## Basic usage

![usage](https://github.com/jfozard/test_path_analysis/assets/4390954/3519b10f-041e-4fa1-b8b2-8a648fd66c9c)

Upload foci-containing stack (HEI10). Also upload axis traces extracted by SNT (SNT_Data.traces).
Set X-Y and Z resolutions appropriately in "Additional options"

Online demo at https://huggingface.co/spaces/JFoz/test_path_analysis
