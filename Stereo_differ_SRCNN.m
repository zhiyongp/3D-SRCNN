%% TRAIN
clear all;close all;clc
%% Locate the directory where the current M file is located
root = fileparts(mfilename('fullpath')) ;
cd(root) ;
%% Add necessary folders
addpath(genpath('matconvnet'));
addpath(genpath('Data'));
addpath(genpath('Functions'));
addpath(genpath('vlfeat'));
addpath(genpath('Image_database'));
%% run vlfeat/toolbox/vl_setup ;run matconvnet/matlab/vl_setupnn
setup ;
%% Import training data set
database=load('Data/3D_differ-SRCNN-data/3d-train_differ-database-s2.mat') ;
%% initialize a CNN architecture
net = initializeCharacter3D_differ_CNN_PZY() ;
%% train and evaluate the CNN
trainOpts.batchSize = 128 ;
trainOpts.numEpochs = 918 ;
trainOpts.continue = true ;
trainOpts.useGpu = true ;%If there is no GPU support, change to false and select the CPU.
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'Data/3D-differ_srcnn-experiment/s2' ;
tic;
%% Call training function in MatConvNet
[net,info] = cnn_train_3D_differ_pzy(net, database, trainOpts) ;


