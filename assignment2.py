import json
from math import sqrt, acos, degrees, atan, pi
import numpy as np

f = open('json_folder/sit/Li_Peizhuo_sit_keypoints.json', 'r')  # It depends on the name of your JSON file. I named it 'sitting'
content = f.read()
a = json.loads(content)
f.close()
b = a['people']
data = b[0]['pose_keypoints_2d']  # data is a list which contains all the coordinates
print(data)


def separate(datas):
    point = []
    points = []
    for i in range(0, len(datas), 3):
        point.append(datas[i])
        point.append(datas[i + 1])
        points.append(point)
        point = []
    return points


output = separate(data)
print(output)


def classification(input):
    distance_joint9_10 = sqrt((input[9][0] - input[10][0]) ** 2 + (input[9][1] - input[10][1]) ** 2)
    distance_joint10_11 = sqrt((input[10][0] - input[11][0]) ** 2 + (input[10][1] - input[11][1]) ** 2)
    distance_joint9_11 = sqrt((input[9][0] - input[11][0]) ** 2 + (input[9][1] - input[11][1]) ** 2)

    # apply cosine rule
    knee_angle = degrees(acos((distance_joint9_10 ** 2 + distance_joint10_11 ** 2 - distance_joint9_11 ** 2) / (
                2 * distance_joint9_10 * distance_joint10_11)))

    print(knee_angle)

    if knee_angle <= 170:
        return "sitting"
    else:
        return "standing"


result = classification(output)
print(result)
