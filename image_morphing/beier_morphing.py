

import cv2
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('module://backend_interagg')
from image_morphing.line import line
import matplotlib.pyplot as plt


def get_lines(f_path):
    lines = []
    f = open(f_path, "r")
    points = f.readline().strip().split(' ')
    for i in range(0, len(points), 4):
        l = line(np.array([int(points[i]), int(points[i + 1])]),
                 np.array([int(points[i + 2]), int(points[i + 3])]))
        lines.append(l)
    return lines


def perp(v):
    if v[0] == 0 and v[1] == 0:
        raise ValueError('zero vector')

    return np.array([-v[1], v[0]])


def get_u_v(x, d_line):
    line_vector = d_line.end - d_line.start
    u = ((x - d_line.start).dot(line_vector)) / (np.linalg.norm(line_vector) * np.linalg.norm(line_vector))
    v = ((x - d_line.start).dot(perp(line_vector))) / (np.linalg.norm(line_vector))
    return u, v


def get_x(s_line, u, v):
    line_vector = s_line.end - s_line.start
    x_dash = s_line.start + u * line_vector + (v * perp(line_vector)) / np.linalg.norm(line_vector)
    return x_dash


def calculate_weight(d_line, x, v, u, a, p, b):
    length = np.linalg.norm(d_line.end - d_line.start)
    if 0 <= u <= 1:
        dist = abs(v)
    elif u < 0:
        dist = np.linalg.norm(x - d_line.start)
    else:
        dist = np.linalg.norm(x - d_line.end)

    return ((length ** p) / (a + dist)) ** b


def warping(source, source_lines, interpolation_lines):
    # warped = np.zeros(source.shape)
    warped = np.array(source)
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            x = np.array([i, j])
            d_sum = np.array([0, 0])
            w_sum = 0
            for l in range(len(source_lines)):
                interpolated_line = interpolation_lines[l]
                s_line = source_lines[l]

                u, v = get_u_v(x, interpolated_line)
                x_dash = get_x(s_line, u, v)
                di = x_dash - x
                w = calculate_weight(interpolated_line, x, v, u, a=10, p=0, b=1)
                d_sum = d_sum + w * di
                w_sum = w_sum + w

            x_dash = np.array(np.round(x + d_sum / w_sum), dtype=int)
            x_dash[x_dash < 0] = 0
            if x_dash[0] > source.shape[0] - 1:
                x_dash[0] = source.shape[0] - 1
            if x_dash[1] > source.shape[1] - 1:
                x_dash[1] = source.shape[1] - 1
            warped[x[0], x[1]] = source[x_dash[0], x_dash[1]]
            # print(x_dash)
    return warped


def cross_dissolve(warped_source, warped_destination, frame_no, total_frames):
    ratio = frame_no / total_frames
    return (1 - ratio) * warped_source + ratio * warped_destination


def get_interpolation_lines(source_lines, destination_lines, frame_no, total_frames):
    interpolation_lines = []
    for l in range(len(source_lines)):
        start_diff = destination_lines[l].start - source_lines[l].start
        end_diff = destination_lines[l].end - source_lines[l].end
        interpolation_lines.append(line(source_lines[l].start + ((frame_no / total_frames) * start_diff),
                                        source_lines[l].end + ((frame_no / total_frames) * end_diff)))
    return interpolation_lines


def morph_image(source, destination, total_frames):
    source_path = "Aya_points.txt"
    dest_path = "Nada_points.txt"
    source_lines = get_lines(source_path)
    destination_lines = get_lines(dest_path)
    for x in range(1, total_frames):
        interpolation_lines = get_interpolation_lines(source_lines, destination_lines, x, total_frames)
        warped_destination = warping(destination, destination_lines, interpolation_lines)
        plt.imshow(warped_destination)
        plt.show()
        warped_source = warping(source, source_lines, interpolation_lines)
        plt.imshow(warped_source)
        plt.show()
        warped_source = warped_source.astype(int)
        warped_destination = warped_destination.astype(int)
        morphed_image = cross_dissolve(warped_source, warped_destination, x, total_frames).astype('uint8')
        # cv2.imwrite(str(x) + ".jpg", morphed_image)
        new_im = Image.fromarray(morphed_image)
        new_im.save(str(x)+".jpg")



source = plt.imread("Aya.jpg")
destination = plt.imread("Nada.jpg")
plt.imshow(source)
plt.show()
plt.imshow(destination)
plt.show()
morph_image(source, destination, 5)
