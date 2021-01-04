import time
import A1.A1 as ax
import A2.A2 as ay
import B1.B1 as bx
import B2.B2 as by


start = time.time()
# ########### Task A1: Gender Detection ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start1 = time.time()
modelax = ax.main(0, None, None)    # Create model
trainax = ax.main(1, modelax, 0)    # Train model
testax = ax.main(2, modelax, 0)     # Test model
end1 = time.time()

# ########### Task A2: Smile Detection ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start2 = time.time()
modelay = ay.main(0, None, None)    # Create model
trainay = ay.main(1, modelay, 0)    # Train model
testay = ay.main(2, modelay, 0)     # Test model
end2 = time.time()

# ########### Task B1: Face Shape Recognition ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start3 = time.time()
modelbx = bx.main(0, None, None)    # Create model
trainbx = bx.main(1, modelbx, 0)    # Train model
testbx = bx.main(2, modelbx, 0)     # Test model
end3 = time.time()

# ########### Task B2: Eye Colour Recognition ###########
# Arg1 - Create: 0, Train: 1, Test: 2
# Arg3 - Hide Confusion Matrix: 0, Show Confusion Matrix: 1
start4 = time.time()
modelby = by.main(0, None, None)    # Create model
trainby = by.main(1, modelby, 0)    # Train model
testby = by.main(2, modelby, 0)     # Test model
end4 = time.time()

print("Task A1 - Training accuracy:\t", str(trainax)[:5], "\nTask A1 - Unseen accuracy:\t\t", str(testax)[:5], "\nTask A2 - Training accuracy:\t", str(trainay)[:5], "\nTask A2 - Unseen accuracy:\t\t", str(testay)[:5], "\nTask B1 - Training accuracy:\t", str(trainbx)[:5], "\nTask B1 - Unseen accuracy:\t\t", str(testbx)[:5], "\nTask B2 - Training accuracy:\t", str(trainby)[:5],"\nTask B2 - Unseen accuracy:\t\t", str(testby)[:5])

end = time.time()
print("Time for A1:\t\t", end1-start1)
print("Time for A2:\t\t", end2-start2)
print("Time for B1:\t\t", end3-start3)
print("Time for B2:\t\t", end4-start4)
print("Total time elapsed:\t", end-start)

# ######### Accuracy Results #########
# Task A1 - Training accuracy:	 0.937
# Task A1 - Unseen accuracy:		 0.929
# Task A2 - Training accuracy:	 0.908
# Task A2 - Unseen accuracy:		 0.908
# Task B1 - Training accuracy:	 0.967
# Task B1 - Unseen accuracy:		 0.967
# Task B2 - Training accuracy:	 1.0
# Task B2 - Unseen accuracy:		 1.0

# ######### Without Pickle Files #########
# Time for A1:		 53.720837116241455
# Time for A2:		 3.75040864944458
# Time for B1:		 70.45920777320862
# Time for B2:		 49.23138475418091
# Total time elapsed:	 177.16183829307556

######### With Pickle Files #########
# Time for A1:		 3.464149236679077
# Time for A2:		 3.779435634613037
# Time for B1:		 3.475635290145874
# Time for B2:		 0.08607816696166992
# Total time elapsed:	 10.805298328399658



