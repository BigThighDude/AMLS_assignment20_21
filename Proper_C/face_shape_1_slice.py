from csv import reader
import numpy as np
import cv2
from matplotlib import pyplot as plt

directory = './cartoon_set/'
csv_src = directory+'labels.csv'
img_src = directory+'img/'

with open(csv_src) as file:
    labels = list(reader(file, delimiter='\t'))
    labels = labels[1:]

no_pics = len(labels)
y_name = []
dat = np.zeros(shape=(no_pics,2))
for i in range(no_pics):
    y_name.append(labels[i][3])
    dat[i][0] = int(labels[i][1])
    dat[i][1] = int(labels[i][2])

no_slice = 4
points = np.zeros(shape=(no_pics,1))
counter = 0

for m in range(4, 5):
    img = cv2.imread(img_src+y_name[m])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sample1 = img[:][290]
    sample1 = sample1[157:173]
    avg = np.zeros(shape=(len(sample1),1), dtype=float)
    for i in range(0, len(sample1)):
        avg[i] = np.mean(sample1[i])
    min_val = avg.min()
    pos_list = np.where(avg==min_val)
    pos_mid = pos_list[0][int(len(pos_list[0])/2)]
    avg_at_pos = avg[pos_mid]
    print(avg_at_pos)
    if avg_at_pos<34:
        points[m] = pos_mid
    else:
        points[m] = 0
print(points[4:5])




############ shows sample slice #################
rgb_array = sample1
img = np.array(rgb_array, dtype=int).reshape((1, len(rgb_array), 3))
plt.imshow(img, extent=[0, 16000, 0, 1], aspect='auto')
plt.show()


############ face edges ##############
# 250,356 - lower lip
# 250,391 - under lowest chin

# 160,365 - pos3 - 1st
# 250,365 - pos3 - 2nd

# 215,335 - pos2 - 1st
# 130,335 - pos2 - 2nd

# 157,290 - pos1 - 1st
# 173,290 - pos1 - 2nd
