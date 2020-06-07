# Multi-Camera-CPU_Workload
This code base acts as a foundation to enable multiple cameras to be processed in real time by single Intel CPU.
Multi camera processing in real time is one of the biggest workloads a computer vision pipeline can throw at the CPU. 

Intel CPUs, although supports parallelism will struggle to parallel process multiple streams with a deep learning workload, because the process utilizes multiple cores simultaneosly causing the pipeline to bottleneck when parallel processing is initiated.

Batch wise stream processing from a buffered queue manager based on OpenCV populates real-time streams into inference batches.
Each inference batch is processed and results are visualized on to a display sink.

![img](https://github.com/vyzboy92/Multi-Camera-CPU_Workload/blob/master/utils/images/multicam.png)

## Prerequisites
1. OpenVINO
2. OpenCV
3. imutils
4. Python 3.5 or higher
6. numpy

## Run Demo
1. Open ```utils/config.py``` and enter the video sources in ```video_inputs``` list. The program supports webcam, rtsp and video file inputs.
2. Run ```python3 multi_camera_CPU_inference.py```
