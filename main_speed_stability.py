import numpy as np
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
#import scipy

from argparse import ArgumentParser
import os




def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent

def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)

    return diss

def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

filenames = [ 'F1_1_3_1.ts', 'F1_1_3_2.ts', 'F1_1_4_1.ts', 'F1_1_4_2.ts', 'F1_2_3_1.ts', 'F1_2_3_2.ts', 'F2_1_2_2.ts', 'F5_1_2_1.ts', 'F5_1_2_2.ts', 'F5_2_2_1.ts']

parser = ArgumentParser()

parser.add_argument(
        '--video', required=True,
        help="Path to the video."
        )
args = parser.parse_args()

filename = os.path.expanduser(args.video)



print(f'file: {filename} started processing')
pts = deque(maxlen=10)
cap = cv2.VideoCapture(filename)

#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(filename+'.mp4', fourcc, 30.0, (800, 400), True)
#out_stability = cv2.VideoWriter(filename+'stability.mp4', fourcc, 30.0, (800, 400), True)
pTime = 0
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=30,
                      blockSize=7)

# Parameters for lucas
# kanade optical flow
lk_params = dict(winSize=(30, 30),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (200, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = old_frame[:400, :, :]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
counter = 0
speed = deque(maxlen=24)
stability = deque(maxlen=100)

shown_speed = 0
shown_stability = 0

ent_old = 0

speed_list = []
stability_list = []
while (1):
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #print(counter)
    if counter>=1800:
        break
    counter += 1
    # if counter < 1640:
    #    continue
    ret, frame = cap.read()
    if ret:

        frame = frame[:400, :, :]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if counter % 100 ==0:
        # speed = []
        if counter % 10 == 0:
            # speed = []
            mask = np.zeros_like(old_frame)
            p0_update = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            p0 = np.concatenate((p0, p0_update), axis=0)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # print(p0)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        dX = []
        dY = []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            dX.append(a - c)
            dY.append(b - d)
            #print(a, b, c, d)
            new_mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 0, 255), 2)
            frame = cv2.circle(frame,(int(a), int(b)), 5, (255, 0, 255), -1)
        # print(np.mean(dX))
        # print(np.mean(dY))

        img = mask
        img = cv2.add(frame, new_mask)
        mask = new_mask
        img=frame
        cur_speed = np.sqrt(np.mean(dX) ** 2 + np.mean(dY) ** 2) * 30
        speed.appendleft(cur_speed)

        if counter % 150 == 0:###was 10
            shown_speed = np.mean(speed)
            speed_list.append(shown_speed)

        cv2.putText(img, f'speed: {int(shown_speed)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        if counter%1==0:
            frame_gra = rescale_linear(frame_gray, 0, 254).astype(int)
            ent = fast_glcm_entropy(frame_gra)
            #diss = fast_glcm_dissimilarity(frame_gra)
            ent = rescale_linear(ent, 0, 255).astype('uint8')
            # n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image,
            #                                                                   connectivity=4)

            _, ent = cv2.threshold(ent, 140, 140, cv2.THRESH_BINARY_INV)
            ent_sum = np.sum(ent/255)

            stability.appendleft(ent_sum-ent_old)

            if counter % 150 == 0:
                shown_stability = np.mean(stability)
                stability_list.append(shown_stability)
            cv2.putText(ent, f'stability: {int(shown_stability)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)

            ent_old = ent_sum
            cv2.imshow('glcm', 255 - ent)
            #cv2.imshow('diss', diss)
        cv2.imshow('frame', img)

        #img = cv2.resize(img, (800, 400), cv2.INTER_LANCZOS4)
        #print(img.shape)
        #out.write(img)

        #img = cv2.resize(255 - ent, (800, 400), cv2.INTER_LANCZOS4)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #print(img.shape)
        #out_stability.write(img)
        key = cv2.waitKey(1) & 0xff

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            out.release()
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


    else:
        cap.release()
        continue



        #cv2.destroyAllWindows()
#cv2.destroyAllWindows()
#cap.release()
#out.release()

# import json
#
# row = []
#
# for i in range(len(speed_list)):
#     row.append({'speed': speed_list[i], 'stability' : stability_list[i]})
#
# with open(filename+'.json', 'w') as outfile:
#   json.dump(row, outfile)


