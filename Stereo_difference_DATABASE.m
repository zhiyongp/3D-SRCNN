%% DATABASE
clear all;close all;clc
%% Locate the directory where the current M file is located
root = fileparts(mfilename('fullpath')) ;
cd(root) ;
%% Add necessary folders
addpath(genpath('matconvnet'));
addpath(genpath('Data'));
addpath(genpath('Functions'));
addpath(genpath('Image_database'));
%% settings
folder_train_L = './Image_database/3D-train_L';
folder_train_R = './Image_database/3D-train_R';

size_input = 33;
size_label = 21;
scale = 2;
stride = 14;

%% initialization
data = zeros(size_input, size_input, 1, 1,'single');
label = zeros(size_label, size_label, 1, 1,'single');
padding = abs(size_input - size_label)/2;
count = 0;

%% generate training data
filepaths_train_L = dir(fullfile(folder_train_L,'*.bmp'));
filepaths_train_R = dir(fullfile(folder_train_R,'*.bmp'));
tic;    
for i = 1 : length(filepaths_train_L)
    
    image_train_L = imread(fullfile(folder_train_L,filepaths_train_L(i).name));
    image_train_L = rgb2ycbcr(image_train_L);
    image_train_L = im2single(image_train_L(:, :, 1));
    image_train_L = modcrop(image_train_L, scale);
    
    image_train_R = imread(fullfile(folder_train_R,filepaths_train_R(i).name));
    image_train_R = rgb2ycbcr(image_train_R);
    image_train_R = im2single(image_train_R(:, :, 1));
    image_train_R = modcrop(image_train_R, scale);
    
    [hei,wid] = size(image_train_R);
    image_train_R_LR = imresize(imresize(image_train_R,1/scale,'bicubic'),[hei,wid],'bicubic');
    image_train_differ=image_train_L-image_train_R_LR; 
    image_train_differ_label=image_train_L-image_train_R; 

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input_train = image_train_differ(x : x+size_input-1, y : y+size_input-1);
            subim_label_train = image_train_differ_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1;
            data(:, :, 1, count) = subim_input_train;
            label(:, :, 1, count) = subim_label_train;
        end
    end
end
toc;
order = randperm(count);
data(:,:,1,:) = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% save
save('Data/3D_differ-SRCNN-data/3d-train_differ-database-s2.mat', 'data','label','scale','-v7.3');
