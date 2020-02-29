** Analysis & work jointly done by Anilkumar N Bhatt and Maruthi Srinivas **
----------------------------------------------------------------------------

- This repository is having code bases and results of Squeeze-Expand CNN model for MNIST dataset.
- Objective is to plot and compare validation accuracies and test loss and thereby determine which regularization method - **with L1**, **with L2**, **both combined** or **not having any of L1 or L2** - gives better results in MNIST and for which all values.
- Model is having 9680 parameters and was ran 4 times each for 40 epochs (1) w/o L1 or L2 (2) With L1 (3) With L2 (4) With L1 & L2 for 2 different set of L1 & L2 values.
- Also 25 misclassified images are plotted for 'with L1' and 'with L2' scenarios along with its actual labels & wrongly predicted labels.
- First time, model was ran for the 4 scenarios with values for (L1, L2) as (0,0),(0.001,0),(0,0.0005),(0.001,0.0005) respectively.
  - Code base for the same is [Code with L1 Regularization_V1](EVA4_S6_Regularization_V1.ipynb)
  - Results are as follows:
      - [L1 = 0, L2 = 0] : Maxium test accuracy: 99.47, Achieved in epoch: 25, Max Train accuracy : 99.49, Achieved in epoch : 37
      - [L1 = 0.001, L2 = 0] : Maxium test accuracy: 99.45, Achieved in epoch: 28, Max Train accuracy : 99.46, Achieved in epoch : 38
      - [L1 = 0, L2 = 0.0005] : Maxium test accuracy: 99.47, Achieved in epoch: 19, Max Train accuracy : 99.13, Achieved in epoch : 35
      - [L1 = 0.001, L2 = 0.0005] : Maxium test accuracy: 99.30, Achieved in epoch: 12, Max Train accuracy : 99.12, Achieved in epoch : 39
  - Validation accuracy & Test loss plots are as below:
  ![VaL_Acc&Test_Loss](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/Val_Test%20Accuracies.png)
- Second time, model was ran for the 4 scenarios with values for (L1, L2) as (0,0),(0.0005,0),(0,0.0001),(0.0005,0.0001)].
  - Code base for the same is [Code with L1 Regularization_V2](EVA4_S6_Regularization_V2.ipynb)
  - Results are as follows:
      - [L1 = 0, L2 = 0] :Maxium test accuracy: 99.47, Achieved in epoch: 38, Max Train accuracy : 99.50, Achieved in epoch : 33
      - [L1 = 0.0005, L2 = 0] : Maxium test accuracy: 99.43, Achieved in epoch: 39, Max Train accuracy : 99.48, Achieved in epoch : 38
      - [L1 = 0, L2 = 0.0001] : Maxium test accuracy: 99.43, Achieved in epoch: 27, Max Train accuracy : 99.39, Achieved in epoch : 38
      - [L1 = 0.0005, L2 = 0.0001] : Maxium test accuracy: 99.42, Achieved in epoch: 33, Max Train accuracy : 99.39, Achieved in epoch : 27
  - Validation accuracy & Test loss plots are as below:
  ![VaL_Acc&Test_Loss](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/Val_Test%20Accuracies_Final%20L1_L2%20Model_V2.png)
- Values for L1 and L2 were arrived based on experiments ran seperately with L1 alone & with L2 alone. Please refer the sections below - **EVA4_S6_With_L1_Regularization.ipynb** & **EVA4_S6_With_L2_Regularization.ipynb** regarding these findings.
- **Conclusions are as follows:**
  - Using L1 = 0.001 alone is the best choice for MNIST data set as it is giving a stable smoother curve with higher accuracy and lower test loss.
  - Adding L2 is not helping as accuracies/test losses are not good compared to 'with L1' and 'w/o L1 or L2' models and curve appears to be turbulent.
  - Model without regularization is performing better than model with regularization. This could be because MNIST is a straight-forward dataset. Hence regularization won't do much improvement because kernel values easily gets generalized and there are not much challenges for kernels to fall into specialization trap. 
  - However incase of other images (eg: human emotion), outliers could be more and features in images will be more complex. Inorder to cater to these images, some of the kernels will get into specialization mode while training. This could lead to overfitting i.e. good training accuracy but not that great train accuracy. In such cases, L1 regularization will help to reduce overfitting by making weights of these specialized kernels to zero thereby killing them.
  - Incase of MNIST, not using regularization is the best option
  - Next best option is to use a moderate value for L1 regularization & no use of L2 regularization.
  - If L2 regularization has to be used, best option is to go for a lower value i.e 0.0001 is better than 0.0005.
  
 - **Misclassified images while using L1 = 0.001 alone**

![L1 =0.001 Wrong 25](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/25%20Misclassied%20Images_With%20L1%3D0.001.png)

 - **Misclassified images while using L2 = 0.0005 alone**
 
 ![L2 =0.0005 Wrong 25](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/25%20Misclassied%20Images_With%20L2%3D0.0005.png)
 
[With_L1_Regularization only](EVA4_S6_With_L1_Regularization.ipynb)
------------------------------------
- This model was ran before running '**EVA4_S6_With_L1_Regularization_V1.ipynb**' and '**EVA4_S6_With_L1_Regularization_V2.ipynb**'to analyze the behaviour of MNIST for various L1 values.
- Chosen L1 = 0.001 as it seems to stable and giving better test accuracies and test losses than other values.
- L1 = 0.0005 is also giving good results. But test loss variation seems to be slightly turbulent compared to L1 = 0.001
- Plots for accuracies and losses are as below for different L1 Values.
![L1 Only-Accuracy & Losses](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/Val%20Accuracy_Losses%20for%20diff%20L1%20values.png)

[With_L2_Regularization only](EVA4_S6_With_L2_Regularization.ipynb)
------------------------------------
- This model was ran before running '**EVA4_S6_With_L1_Regularization_V1.ipynb**' and '**EVA4_S6_With_L1_Regularization_V2.ipynb**'to analyze the behaviour of MNIST for various L2 values.
- L2 = 0.0001 is the best value compared to L2 = 0.0005 as former seems to be more stable and gives better test accuracies and test losses compared to other value.
- Plots for accuracies and losses are as below.
![L2 Only-Accuracy & Losses](https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase1/blob/master/S6-L1_L2Regularization/Val%20Accuracy_Losses%20for%20diff%20L2%20values.png)
