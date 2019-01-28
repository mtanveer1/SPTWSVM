Description of files in SPTWSVM codes.

-----------------Scripts------------------
1. main.m : Used to calculate sparsity of SPTWSVM or TWSVM or Sparse Pin SVM models for linear case.

2. main_kernel.m : Used to calculate sparsity of SPTWSVM or TWSVM or Sparse Pin SVM models for non-linear case.

3. main_with_optimal_c.m : Used to calculate accuracy of SPTWSVM or TWSVM or Sparse Pin SVM models for linear case.

4. main_with_optimal_c_kernel.m : Used to calculate accuracy of SPTWSVM or TWSVM or Sparse Pin SVM models for non-linear case.

5. adding_noise.m : For adding noise in each feature (with zero mean gaussian distribution) in the dataset.

---------------Functions------------------
6. TSVM.m : Function which returns accuracy, non-zero dual variables and training time for linear TSVM. Inputs taken are (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, and value of c.

7. TSVM_Kernel.m : Function which returns accuracy, non-zero dual variables and training time for non-linear TSVM (kernel generated surfaces). Inputs taken are (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c, and value of gamma.

8. Sparse_Pin_SVM.m : Function which returns accuracy, non-zero dual variables and training time for linear Sparse Pin SVM. Inputs taken are (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c, value of epsilon, and value of tau.

9. Sparse_Pin_SVM_Kernel.m : Function which returns accuracy, non-zero dual variables and training time for non-linear Sparse Pin SVM. Inputs taken are (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c, value of gamma, value of epsilon, and value of tau.

10. Sparse_Pin_TSVM.m : Function which returns accuracy, non-zero dual variables and training time for linear SPTWSVM. Inputs taken are (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c, epsilon value, and tau value.

11. Sparse_Pin_TSVM_Kernel.m : Function which returns accuracy, non-zero dual variables and training time for non-linear SPTWSVM (kernel generated surfaces). Inputs taken are (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c, value of gamma, epsilon value, and tau value.

12. RBF.m : Used to calculate RBF kernel value between two vectors u and v.