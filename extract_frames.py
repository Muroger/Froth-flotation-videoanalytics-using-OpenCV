import cv2
vidcap = cv2.VideoCapture('stock-footage-copper-sulfide-loaded-air-bubbles-on-a-jameson-cell-at-the-flotation-plant-in-a-copper-mine.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./frames3/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1