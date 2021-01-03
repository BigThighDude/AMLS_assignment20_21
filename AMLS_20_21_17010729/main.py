import time
import A1.A1 as ax
import A2.A2 as ay
import B1.B1 as bx
import B2.B2 as by


start = time.time()
########### Task A1: Gender Detection ###########
# Arg1 - Create: 0, Train: 1, Test: 2
start1 = time.time()
modelax = ax.main(0, None)
trainax = ax.main(1, modelax)
testax = ax.main(2, modelax)
end1 = time.time()

# ########### Task A2: Smile Detection ###########
# Arg1 - Create: 0, Train: 1, Test: 2
start2 = time.time()
modelay = ay.main(0, None)
trainay = ay.main(1, modelay)
testay = ay.main(2, modelay)
end2 = time.time()

# ########### Task B1: Face Shape Recognition ###########
# Arg1 - Create: 0, Train: 1, Test: 2
start3 = time.time()
modelbx = bx.main(0, None)
trainbx = bx.main(1, modelbx)
testbx = bx.main(2, modelbx)
end3 = time.time()

# ########### Task B2: Eye Colour Recognition ###########
# Arg1 - Create: 0, Train: 1, Test: 2
start4 = time.time()
modelby = by.main(0, None)
trainby = by.main(1, modelby)
testby = by.main(2, modelby)
end4 = time.time()

print("Task A1 - Training accuracy:\t", str(trainax)[:4], "\nTask A1 - Unseen accuracy:\t\t", str(testax)[:4], "\nTask A2 - Training accuracy:\t", str(trainay)[:4], "\nTask A2 - Unseen accuracy:\t\t", str(testay)[:4], "\nTask B1 - Training accuracy:\t", str(trainbx)[:4],"\nTask B1 - Unseen accuracy:\t\t", str(testbx)[:4], "\nTask B2 - Training accuracy:\t", str(trainby)[:4],"\nTask B2 - Unseen accuracy:\t\t", str(testby)[:4])

end = time.time()
print("Time for A1:\t\t", end1-start1)
print("Time for A2:\t\t", end2-start2)
print("Time for B1:\t\t", end3-start3)
print("Time for B2:\t\t", end4-start4)
print("Total time elapsed:\t",end-start)

# 3.5 seconds to run with pickle files
# 260 seconds to run without pickle files

