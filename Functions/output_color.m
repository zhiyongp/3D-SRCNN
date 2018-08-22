%% ���ԭʼȫ�ֱ���ͼƬ�͵ͷֱ���ͼƬ
%% ��λ����ǰM�ļ����ڵ�Ŀ¼
root = fileparts(mfilename('fullpath')) ;
cd(root) ;
scale=2;
X1=imread('E:\SRCNN_result\���ӵ�ԱȽ��\����my\RGB\3d-srcnn-Poznan_Street_00_1920x1088_rec_cam04-1.bmp');%��һ�׶����ӵ��ؽ�ͼ
X1 = rgb2ycbcr(X1);
X1 = im2single(X1(:,:,1));
X1= modcrop(X1, scale);

X2=imread('E:\SRCNN_result\���ӵ�ԱȽ��\����my\RGB\3d-srcnn_IBP-Poznan_Street_00_1920x1088_rec_cam04-1.bmp');%�ڶ��׶����ӵ��ؽ�ͼ
X2 = rgb2ycbcr(X2);
X2 = im2single(X2(:,:,1));
X2= modcrop(X2, scale);

im_R = imread('3D-test_R_singleview/Poznan_Street_00_1920x1088_rec_cam04-1.bmp');%ԭʼ���ӵ�ȫ�ֱ�ͼƬ
im_R = modcrop(im_R , scale);
im_L= imread('3D-test_L_singleview/Poznan_Street_00_1920x1088_rec_cam05-1.bmp');%ԭʼ���ӵ�ȫ�ֱ�ͼƬ
im_L= modcrop(im_L, scale);


im_L_T = rgb2ycbcr(im_L);
im_L_T_Y = im2single(im_L_T(:,:,1));

% [hei,wid,~] = size(im_R);
im_R_T = rgb2ycbcr(im_R);
im_R_T = im2single(im_R_T);%ȡy���� im2single()��single()/255һ��
im_input_test_down_rgb=imresize(im_R,1/scale,'bicubic');%�²��������ӵ�RGB
im_input_test_down_Y=imresize(im_R_T(:,:,1),1/scale,'bicubic');%LR(��С��С4��)
image_test_R_LR_Y = imresize(im_input_test_down_Y,scale,'bicubic');%���Եĵͷֱ������ӵ�(����Bicubic��ֵ���)
im_input_diff=im_L_T_Y-image_test_R_LR_Y;%����ĳ�ʼ�ӵ��ֵ
im_L_T_Y_shave=shave(im_L_T_Y, [scale, scale]);
im_guji_diff =im_L_T_Y_shave-X1;%����Ĺ����ӵ��ֵ



im_L_output_RGB= im_L;%����ü����ԭʼ��ȫ�ֱ��ɫ
im_R_output_RGB= im_R ;%����ü����ԭʼ��ȫ�ֱ��ɫ
im_R_output_DOWN_RGB= im_input_test_down_rgb; %����ü�����²��������ӵ��ɫ
im_L_T_Y_output= im_L_T_Y ;
im_L_T_Y_output=uint8(im_L_T_Y_output*255);%����ü����ԭʼ��Y
im_R_test_down_Y_output=im_input_test_down_Y ;
im_R_test_down_Y_output=uint8(im_R_test_down_Y_output*255);%����ü�����²�������Y
image_test_R_LR_Y_output=image_test_R_LR_Y ;
image_test_R_LR_Y_output=uint8(image_test_R_LR_Y_output*255);%�����ֵ����Y
im_input_diff_output= map_pzy(im_input_diff);
im_input_diff_output= shave(im_input_diff_output, [scale, scale]);%�����ʼ��ֵ

im_guji_diff_output=map_pzy(im_guji_diff);%������Ʋ�ֵ
X1_output=uint8(X1*255);%�����һ�׶ε��ؽ�Y
X2_output=uint8(X2*255);%����ڶ��׶ε��ؽ�Y

imwrite(im_L_output_RGB,['E:\SRCNN_result\��ͼ���ͼ\','pozan_L_RGB','.jpg']);
imwrite(im_R_output_RGB,['E:\SRCNN_result\��ͼ���ͼ\','pozan_R_RGB','.jpg']);
imwrite(im_R_output_DOWN_RGB,['E:\SRCNN_result\��ͼ���ͼ\','pozan_R_DOWN_RGB','.jpg']);
imwrite(im_L_T_Y_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_L_Y','.jpg']);
imwrite(im_R_test_down_Y_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_R_DOWN_Y','.jpg']);
imwrite(image_test_R_LR_Y_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_R_LR_Y','.jpg']);
imwrite(im_input_diff_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_ini_d','.jpg']);
imwrite(im_guji_diff_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_guji_d','.jpg']);
imwrite(X1_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_di1_SR','.jpg']);
imwrite(X2_output,['E:\SRCNN_result\��ͼ���ͼ\','pozan_di2_SR','.jpg']);
