**EVA4_S6_Regularization.ipynb

- This notebook is having code base and results of Squeeze-Expand CNN model for MNIST dataset.
- Objective is to plot and compare validation accuracies and test loss and thereby determine which regularization method - L1, L2, both combined or not having both - gives better results in MNIST.
- Model is having 9680 parameters and was ran 4 times for 40 epochs (1) w/o L1 or L2 (2) With L1 (3) With L2 (4) With L1 & L2.
- Also 25 mis-classified images are plotted for each of these scenarios along with its actual labels & wrongly predicted labels.
- Values chosen for (L1, L2) for the 4 scenarios are (0,0),(0.001,0),(0,0.0005),(0.001,0.0005)].
- Values for L1(0.001) and L2(0.0005) were arrived based on experiments ran seperately with L1 alone & with L2 alone. Refer the notebook descriptions below.
