# Motion Heatmap

This project contains the code for utilizing heatmaps in Python. The heatmaps here work in conjunction with YOLOv8 to track peoples' average movement in an area, over an extended period of time. 

The heatmaps here show the most commonly travelled paths and points of interest from a stationary CCTV camera's video feed. This has many practical applications in the real world such as tracking customer interests and hangaround times in retail settings.

<br/>


## Demo
![Demo video for heatmaps](./assets/HeatmapDemo.gif)

<br/>

## Usage

Clone the repository to local system with Python. Create a virtual environment and run the following:

```bash
pip install ultralytics opencv-python
```

Within main.py and ul_heatmaps.py, change "video.mp4" within CustomVideoCapture() to any local video file with appropriate file path or an RSTP/RTMP/HTTP video stream.

To run the custom heatmaps code:

```bash
python main.py
```

To run the Ultralytics Heatmap code:

```bash
python ul_heatmaps.py
```

<br/>


## Requirements

- Python 3.11
- YOLOv8 (Ultralytics)
- OpenCV

<br/>