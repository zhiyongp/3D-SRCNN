%% 输出原始全分辨率图片和低分辨率图片
%% 定位到当前M文件所在的目录
root = fileparts(mfilename('fullpath')) ;
cd(root) ;
scale=2;
X1=imread('E:\SRCNN_result\单视点对比结果\最新my\RGB\3d-srcnn-Poznan_Street_00_1920x1088_rec_cam04-1.bmp');%第一阶段右视点重建图
X1 = rgb2ycbcr(X1);
X1 = im2single(X1(:,:,1));
X1= modcrop(X1, scale);

X2=imread('E:\SRCNN_result\单视点对比结果\最新my\RGB\3d-srcnn_IBP-Poznan_Street_00_1920x1088_rec_cam04-1.bmp');%第二阶段右视点重建图
X2 = rgb2ycbcr(X2);
X2 = im2single(X2(:,:,1));
X2= modcrop(X2, scale);

im_R = imread('3D-test_R_singleview/Poznan_Street_00_1920x1088_rec_cam04-1.bmp');%原始右视点全分辨图片
im_R = modcrop(im_R , scale);
im_L= imread('3D-test_L_singleview/Poznan_Street_00_1920x1088_rec_cam05-1.bmp');%原始左视点全分辨图片
im_L= modcrop(im_L, scale);


im_L_T = rgb2ycbcr(im_L);
im_L_T_Y = im2single(im_L_T(:,:,1));

% [hei,wid,~] = size(im_R);
im_R_T = rgb2ycbcr(im_R);
im_R_T = im2single(im_R_T);%取y分量 im2single()与single()/255一样
im_input_test_down_rgb=imresize(im_R,1/scale,'bicubic');%下采样的右视点RGB
im_input_test_down_Y=imresize(im_R_T(:,:,1),1/scale,'bicubic');%LR(大小缩小4倍)
image_test_R_LR_Y = imresize(im_input_test_down_Y,scale,'bicubic');%测试的低分辨率右视点(经过Bicubic插值后的)
im_input_diff=im_L_T_Y-image_test_R_LR_Y;%输入的初始视点差值
im_L_T_Y_shave=shave(im_L_T_Y, [scale, scale]);
im_guji_diff =im_L_T_Y_shave-X1;%输入的估计视点差值



im_L_output_RGB= im_L;%输出裁剪后的原始左全分辨彩色
im_R_output_RGB= im_R ;%输出裁剪后的原始右全分辨彩色
im_R_output_DOWN_RGB= im_input_test_down_rgb; %输出裁剪后的下采样的右视点彩色
im_L_T_Y_output= im_L_T_Y ;
im_L_T_Y_output=uint8(im_L_T_Y_output*255);%输出裁剪后的原始左Y
im_R_test_down_Y_output=im_input_test_down_Y ;
im_R_test_down_Y_output=uint8(im_R_test_down_Y_output*255);%输出裁剪后的下采样的右Y
image_test_R_LR_Y_output=image_test_R_LR_Y ;
image_test_R_LR_Y_output=uint8(image_test_R_LR_Y_output*255);%输出插值的右Y
im_input_diff_output= map_pzy(im_input_diff);
im_input_diff_output= shave(im_input_diff_output, [scale, scale]);%输出初始差值

im_guji_diff_output=map_pzy(im_guji_diff);%输出估计差值
X1_output=uint8(X1*255);%输出第一阶段的重建Y
X2_output=uint8(X2*255);%输出第二阶段的重建Y

imwrite(im_L_output_RGB,['E:\SRCNN_result\框图结果图\','pozan_L_RGB','.jpg']);
imwrite(im_R_output_RGB,['E:\SRCNN_result\框图结果图\','pozan_R_RGB','.jpg']);
imwrite(im_R_output_DOWN_RGB,['E:\SRCNN_result\框图结果图\','pozan_R_DOWN_RGB','.jpg']);
imwrite(im_L_T_Y_output,['E:\SRCNN_result\框图结果图\','pozan_L_Y','.jpg']);
imwrite(im_R_test_down_Y_output,['E:\SRCNN_result\框图结果图\','pozan_R_DOWN_Y','.jpg']);
imwrite(image_test_R_LR_Y_output,['E:\SRCNN_result\框图结果图\','pozan_R_LR_Y','.jpg']);
imwrite(im_input_diff_output,['E:\SRCNN_result\框图结果图\','pozan_ini_d','.jpg']);
imwrite(im_guji_diff_output,['E:\SRCNN_result\框图结果图\','pozan_guji_d','.jpg']);
imwrite(X1_output,['E:\SRCNN_result\框图结果图\','pozan_di1_SR','.jpg']);
imwrite(X2_output,['E:\SRCNN_result\框图结果图\','pozan_di2_SR','.jpg']);
