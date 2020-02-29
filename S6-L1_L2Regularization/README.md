EVA4_S6_Regularization.ipynb
----------------------------
- This notebook is having code base and results of Squeeze-Expand CNN model for MNIST dataset.
- Objective is to plot and compare validation accuracies and test loss and thereby determine which regularization method - L1, L2, both combined or not having both - gives better results in MNIST.
- Model is having 9680 parameters and was ran 4 times for 40 epochs (1) w/o L1 or L2 (2) With L1 (3) With L2 (4) With L1 & L2.
- Also 25 mis-classified images are plotted for each of these scenarios along with its actual labels & wrongly predicted labels.
- Values chosen for (L1, L2) for the 4 scenarios are (0,0),(0.001,0),(0,0.0005),(0.001,0.0005)].
- Values for L1(0.001) and L2(0.0005) were arrived based on experiments ran seperately with L1 alone & with L2 alone. Please refer the notebook sections below regarding both these.
- Validation accuracy & Test loss plots are as below. Conclusions are as follows:
  - Using L1 alone is the best choice for MNIST data set as it is giving a stable smoother curve with higher accuracy and lower test loss.
  - Adding L2 is not helping as curve is rocky and accuracies/test losses are not good compared to 'with L1' and 'w/o L1 or L2' models.
  - Even without regularization, model is performing well. This could be because MNIST is a straight-forward dataset. Hence regularization won't do much improvement because kernel values easily gets generalized and there are not much challenges for kernels to fall into specialization trap. 
  - However incase of other images (eg: human emotion), outliers could be more and features in images will be much different. Inorder to cater to these images, some of the kernels will get into specialization mode while training. This could lead to overfitting i.e. good training accuracy but not that great train accuracy. In such cases, L1 regularization will help.
  - Hence, a moderate value of L1 regularization & no use of L2 regularization is most desirable.
  
![VaL_Acc&Test_Loss](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/Val_Test%20Accuracies.png)
