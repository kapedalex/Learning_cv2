import cv2
import sys
import matplotlib.pyplot as plt

video_input_file_name = "src/race_cut.mp4"


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis('off')


def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
tracker_type = tracker_types[1]

if tracker_type == 'BOOSTING':
    # tracker = cv2.legacy_TrackerBoosting.create()
    print('legacy code')
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'CSRT':
    print('legacy code')
    # tracker = cv2.legacy_TrackerCSRT.create()
elif tracker_type == 'TLD':
    # tracker = cv2.legacy_TrackerTLD.create()
    print('legacy code')
elif tracker_type == 'MEDIANFLOW':
    # tracker = cv2.legacy_TrackerMedianFlow.create()
    print('legacy code')
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
else:
    print('legacy code')
    # tracker = cv2.legacy_TrackerMOSSE.create()

video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()

if not video.isOpened():
    print("can not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = 'race-' + tracker_type + '.mp4'
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (width, height))

bbox = (420, 285, 160, 100)
displayRectangle(frame, bbox)

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break

    timer = cv2.getTickCount()

    ok, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure", (80, 140), (0, 0, 255))

    drawText(frame, tracker_type + "Tracker", (80, 60))
    drawText(frame, "FPS: " + str(int(fps)), (80, 100))

    video_out.write(frame)

video.release()
video_out.release()
