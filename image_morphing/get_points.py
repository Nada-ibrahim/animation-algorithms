# import the necessary packages
import argparse
import cv2
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
from PIL import Image

refPt = []
source_line = []
destination_line = []
array_source = []
array_destination = []


class line:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def draw_lines(event, x, y, flags, param):
    global startingPoint, endingPoint, p1, flag, S, E

    # if the left mouse button was clicked, record the starting
    if event == cv2.EVENT_LBUTTONDOWN:

        S = np.array([x, y])
        startingPoint = (x, y)


    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:

        E = np.array([x, y])
        endingPoint = (x, y)

        if flag == False:
            cv2.arrowedLine(source, startingPoint, endingPoint, (0, 255, 0), 2)
            cv2.imshow("source", source)
            p1 = line(startingPoint, endingPoint)
            P = line(S, E)
            f_source.write(str(startingPoint[0]) + " " + str(startingPoint[1]) + " " + str(endingPoint[0]) + " " + str(
                endingPoint[1]) + " ")
            source_line.append(p1)
            array_source.append(P)
            flag = True
            print("source")
        elif flag == True:
            cv2.arrowedLine(destination, startingPoint, endingPoint, (0, 255, 0), 2)
            cv2.imshow("destination", destination)
            p1 = line(startingPoint, endingPoint)
            P = line(S, E)
            f_dest.write(str(startingPoint[0]) + " " + str(startingPoint[1]) + " " + str(endingPoint[0]) + " " + str(
                endingPoint[1]) + " ")
            destination_line.append(p1)
            array_destination.append(P)
            print("dest")

            flag = False


def resize_image(img_path, input_shape=(300, 400)):
    image = Image.open(img_path)
    iw, ih = image.size
    h, w = input_shape
    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    new_image.save(img_path)


source = "ellen.png"
destination = "putin.jpg"
resize_image(source)
resize_image(destination)
source = cv2.imread(source)
clone1 = source.copy()
cv2.namedWindow("source")
cv2.setMouseCallback("source", draw_lines)

f_source = open("source_points.txt", "w")
f_dest = open("dest_points.txt", "w")
flag = False
destination = cv2.imread(destination)
clone2 = destination.copy()
cv2.namedWindow("destination")
cv2.setMouseCallback("destination", draw_lines)

while True:

    cv2.imshow("source", source)
    key = cv2.waitKey(1) & 0xFF

    cv2.imshow("destination", destination)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        break

cv2.destroyAllWindows()
f_source.close()
f_dest.close()
