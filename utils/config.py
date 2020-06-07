import cv2

frame_size = (640,360)

no_input = cv2.imread('utils/images/no_input.jpg')
no_input = cv2.resize(no_input,frame_size)

video_inputs = ['video_1.mp4', 'video_2.mp4']

inputs = len(video_inputs)
