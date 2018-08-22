# 3D-SRCNN

The code was written in MATLAB 2014a, and tested on Windows 10/7.

-------------------------------------------------------------------------
I. RUNNING TRAINING CODE

1. Download MatConvNet package from the following link:
“http://www.vlfeat.org/matconvnet/” and put floder matconvnet into the root of folder TRAIN_Code_v1.0

2. Compile MatConvNet,See “http://www.vlfeat.org/matconvnet/install/” for details.If you have GPU and Cudnn，Please compile MatConvNet
with GPU and Cudnn support to have the best performance.

3. Download vlfeat package from the following link:
“http://www.vlfeat.org/” and put floder vlfeat into the root of folder TRAIN_Code_v1.0

4. Download all training and test sets from “ link：https://pan.baidu.com/s/1Uia-qedVtmyxX9L37Tl7EA password：rqmt ”

5. Copy the "Dataset/Trainset" into the Image_database/3D-train_L and Image_database/3D-train_R folder and Copy the "Dataset/Testset" into the Image_database/3D-test_L and Image_database/3D-test_R folder

6. Generate training data , see "Stereo_difference_DATABASE.m".Note that it 
takes a long time for this function to process the training data

7. Run "Stereo_differ_SRCNN.m" to start the training.Convergence happens after roughly 900 epochs.

The code writes the network in the "Data/3D-differ_srcnn-experiment/s2" folder. Once the converged  network is obtained, you can copy it from "Data/3D-differ_srcnn-experiment/s2" to "Data/TrainedNetwork/3D-differ_srcnn-experiment/s2" folder to test the system using the new network.

-------------------------------------------------------------------------
II. VERSION HISTORY

v1.0 - Initial release 

