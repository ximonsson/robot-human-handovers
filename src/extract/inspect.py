import numpy as np
import cv2


sample_data = \
"""#data/recordings/eight/rgb/25.jpg
17:(165.979,290.842)(157.228,264.49)(131.054,273.646)(139.61,299.697):(148.378,282.176)
-0.748461,-0.302265,328.017,0.508082,-0.834367,373.881,0.000777055,0.000204543,1,
116.972,210.425,38.4291,13.6075,-77.1957
"""


ROI_W = 300
ROI_H = 300


class Grasp:
    """
    Grasp region on an object.
    Represents a rotated angle.
    """

    def __init__(self, x, y, w, h, a):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.a = a

    def __str__(self):
        return "(%f, %f): [%f x %f]: %f" % (self.x, self.y, self.w, self.h, self.a)

    def box(self):
        return (self.x, self.y), (self.w, self.h), self.a



def parse_data(data):
    """
    Parse handover data connected to a frame.
    Returns the filepath to frame, AprilTag ID, Homography matrix and Grasp.
    """
    lines = data.split("\n")

    # tag ID
    tid = lines[1].split(":")[0]

    # homograpy matrix
    h = lines[2][:-1].split(",")
    H = np.zeros((3, 3), np.float64)
    for i in range(len(h)):
        H[int(i/3)][i%3] = np.float64(h[i])

    # grasp region
    grasp_data = list(map(np.float64, lines[3].split(",")))
    g = Grasp(grasp_data[0], grasp_data[1], grasp_data[2], grasp_data[3], grasp_data[4])

    return lines[0][1:], int(tid), H, g



def display(f, tid, H, g):
    """
    Display the handover data in form of the object, it's orientation during the handover
    and the grasping region.
    """
    # load item image
    item = cv2.imread("data/objects/%d.jpg" % tid)
    item = cv2.resize(item, (ROI_W, ROI_H))
    item = cv2.flip(item, 1) # flip it because it is an image from the kinect
    # draw the grasping region
    box = cv2.boxPoints(g.box())
    box = np.int0(box)
    cv2.drawContours(item, [box], 0, (0, 0, 255))
    # warp it to the same perspective as in the handover
    item = cv2.warpPerspective(item, H, (item.shape[0], item.shape[1]))

    # display everything
    cv2.imshow("opencv frame", cv2.flip(cv2.imread(f), 1))
    cv2.imshow("opencv data", item)



def discard(i):
    """
    Discard the frame with index i the handover data connected to it from the dataset.
    """
    pass



def keep(i):
    """
    Keep the frame with index i the handover data connected to it within the dataset.
    """
    pass



fp, tid, H, g = parse_data(sample_data)
display(fp, tid, H, g)
while True:
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    elif k == ord('s'):
        keep(0)
    elif k == ord('d'):
        discard(0)
