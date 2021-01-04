##Open the main file and run as is or comment out the clearly labelled sections if you wish to exclude specific tasks. Can change impur argument to generate confusion matrix for each of the tasks, as well as for the training or testing outcomes.

Need to upload 4 files to the dataset folder in the same format as the given data:
- Datasets
  - cartoon_set
    - img
    - labels
  - cartoon_set_test
    - img
    - labels
  - celeba
    - img
    - labels
  - celeba_test
    - img
    - labels

The program takes 6 minutes total to generate all the data required. A1 generates facial landmark data for both the training/validation set as well as the unseen set. A2 uses the same landmark data so takes less time. B1 and B2 each take samples from different areas of the image so both take some time.
All the samples taken are stored in pickle files so running the program again takes much less time: 12 seconds.
