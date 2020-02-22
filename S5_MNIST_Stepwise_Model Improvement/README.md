Approach Followed: (Refer S5_Writeup.doc for Target, Results and Analysis Details)

1.	Basic Skeleton

2.	Reduce Parameters + Add Batch-norm

3.	Reduce parameters, modify Max-Pooling interval, add dropout

4.	Introduce GAP + FC in final layers

5.	Use image augmentation and fine-tune LR

6.	Use AvgPool2D for GAP and 1x1 for FC (Best model among all - Test accuracy 99.47% in 15 epochs)

7.	Further Improvements (Not fruitful, best model remains 6)
