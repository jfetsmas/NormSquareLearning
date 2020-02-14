# Norm Square Learning

This is a repository containing codes and results for the NormSquareLearning project.

All codes are implemented using tensorflow 1.7.0 python 3.5, except the path norm regularization code (.py files that involve
pathreg in the file name.) which uses tensorflow 2.0.0b1 python 3.6.

To replicate the experimental results. First you need to create two directories /data, and /models.
After running sample_*.py (where * can be chosen from square, quartic and cos), the data should be ready in the data folder.

The first objective is to explore the architectural bias among the three architectures.
Three architecures: GN, LCN, LN are implemented respectively in global_network.py, locally_connected_network.py, and local_network.py
By running the three networks using the three data sets, we can obtain the summary_*.txt files and the best models obtained by
early stopping are now saved in the /models folder.

plot_compare_arch.py can then be used to visualize the difference in performance of the three architectures for certain task.
pathnorm_gen_square.py can be used to compute the path norm of the models saved in the /models folder, and plot_pathnorm_global_network.py
can then be used to plot the growth of path norm with respect to the input dimension.

The file norm_loading.py embed the trained LN into GN and then do the training. It can reproduce the result norm_loading_d_20_epoch_1000.txt
if we specify --d 20 --epoch 1000 in the arguments. plot_sanity.py can then be used on the .txt file to produce a visualization.

Our second objective is to numerically obtain the rates associated with sample complexity.
summary_variousNsamples_fixedratio.txt is obtained by running global_network.py with different Nsamples arguments. The plot_Nsamples.py
file can then visualize the rate of decrease in test loss with respect to the increase in number of samples.

Three regularization techniques are implemented in the files global_network_L1.py, global_network_L2.py, and global_network_pathreg.py
As mentioned in the second paragraph, L1 and L2 are implemented using keras in tf1.7, while path norm regularization is in tf2.0
The results are again stored in the associated summary.txt files, and plot_L1.py, plot_L2.py, plot_pathnorm_pathreg.py can be
used to visualize the results (The GN, LN, LCN restults are also in the plot as a reference).
