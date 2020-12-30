import pickle
import csv
from sklearn import svm

with open(src_csv) as file:
    dat_read = reader(file, delimiter='\t') # data is tab spaces
    temp = list(dat_read)[1:]   # remove first element of list (headers)
    for n in range (0, len(temp)):  # go through each element
        y_name.append(temp[n][1])   # second element is the name
        y_gen.append(temp[n][2])    # third element is gender
        y_smile.append(temp[n][3])   # fourth element is smiling or not

for i in range(0,11):
    