EVA4_S6_Regularization.ipynb
----------------------------
- This notebook is having code base and results of Squeeze-Expand CNN model for MNIST dataset.
- Objective is to plot and compare validation accuracies and test loss and thereby determine which regularization method - L1, L2, both combined or not having both - gives better results in MNIST.
- Model is having 9680 parameters and was ran 4 times for 40 epochs (1) w/o L1 or L2 (2) With L1 (3) With L2 (4) With L1 & L2.
- Also 25 mis-classified images are plotted for each of these scenarios along with its actual labels & wrongly predicted labels.
- Values chosen for (L1, L2) for the 4 scenarios are (0,0),(0.001,0),(0,0.0005),(0.001,0.0005)].
- Values for L1(0.001) and L2(0.0005) were arrived based on experiments ran seperately with L1 alone & with L2 alone. Please refer the notebook sections below - EVA4_S6_With_L1_Regularization.ipynb & EVA4_S6_With_L2_Regularization.ipynb regarding these findings.
- Validation accuracy & Test loss plots are as below. Conclusions are as follows:
  - Using L1 alone is the best choice for MNIST data set as it is giving a stable smoother curve with higher accuracy and lower test loss.
  - Adding L2 is not helping as curve is rocky and accuracies/test losses are not good compared to 'with L1' and 'w/o L1 or L2' models.
  - Even without regularization, model is performing well. This could be because MNIST is a straight-forward dataset. Hence regularization won't do much improvement because kernel values easily gets generalized and there are not much challenges for kernels to fall into specialization trap. 
  - However incase of other images (eg: human emotion), outliers could be more and features in images will be much different. Inorder to cater to these images, some of the kernels will get into specialization mode while training. This could lead to overfitting i.e. good training accuracy but not that great train accuracy. In such cases, L1 regularization will help.
  - Hence, a moderate value of L1 regularization & no use of L2 regularization is most desirable.
  
![VaL_Acc&Test_Loss](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/Val_Test%20Accuracies.png)

 - Misclassified images while using L1 = 0.001 alone

![L1 =0.001 Wrong 25](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/25%20Misclassied%20Images_With%20L1%3D0.001.png)

 - Misclassified images while using L2 = 0.0005 alone
 
 ![L2 =0.0005 Wrong 25](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/25%20Misclassied%20Images_With%20L2%3D0.0005.png)
 
EVA4_S6_With_L1_Regularization.ipynb
------------------------------------
- This model was ran before running 'EVA4_S6_With_L1_Regularization.ipynb' to determine best L1 value.
- Chosen L1 = 0.001 as it seems to stable and giving accuracies and test losses than other values.
- L1 = 0.0005 is also giving good results. But test losses seems to be slightly less stable compared to L1 = 0.001
- Plots for accuracies and losses are as below.
![L1 Only-Accuracy & Losses]
