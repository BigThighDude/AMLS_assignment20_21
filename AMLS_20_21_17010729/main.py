import time
import A1.A1 as ax
import A2.A2 as ay
import B1.B1 as bx
import B2.B2 as by


start = time.time()
########### Task A1: Gender Detection ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start1 = time.time()
modelax = ax.main(0, None, None)
trainax = ax.main(1, modelax, 0)
testax = ax.main(2, modelax, 0)
end1 = time.time()

# ########### Task A2: Smile Detection ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start2 = time.time()
modelay = ay.main(0, None, None)
trainay = ay.main(1, modelay, 0)
testay = ay.main(2, modelay, 0)
end2 = time.time()

########### Task B1: Face Shape Recognition ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start3 = time.time()
modelbx = bx.main(0, None, None)
trainbx = bx.main(1, modelbx, 0)
testbx = bx.main(2, modelbx, 0)
end3 = time.time()

# ########### Task B2: Eye Colour Recognition ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start4 = time.time()
modelby = by.main(0, None, None)
trainby = by.main(1, modelby, 0)
testby = by.main(2, modelby, 0)
end4 = time.time()

print("Task A1 - Training accuracy:\t", str(trainax)[:5], "\nTask A1 - Unseen accuracy:\t\t", str(testax)[:5], "\nTask A2 - Training accuracy:\t", str(trainay)[:5], "\nTask A2 - Unseen accuracy:\t\t", str(testay)[:5], "\nTask B1 - Training accuracy:\t", str(trainbx)[:5], "\nTask B1 - Unseen accuracy:\t\t", str(testbx)[:5], "\nTask B2 - Training accuracy:\t", str(trainby)[:5],"\nTask B2 - Unseen accuracy:\t\t", str(testby)[:5])

end = time.time()
print("Time for A1:\t\t", end1-start1)
print("Time for A2:\t\t", end2-start2)
print("Time for B1:\t\t", end3-start3)
print("Time for B2:\t\t", end4-start4)
print("Total time elapsed:\t", end-start)

# ~12 seconds to run with pickle files
# 6 minutes to run without pickle files
