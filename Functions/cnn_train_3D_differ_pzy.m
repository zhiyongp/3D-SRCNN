function [net, info] = cnn_train_3D_differ_pzy(net, imdb, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.train = [] ;
opts.label = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.00001 ;
opts.momentum = 0.9 ;
%opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = imdb.data ; end
if isempty(opts.label), opts.label = imdb.label ; end

% -------------------------------------------------------------------------
%                        Network initialization
% -------------------------------------------------------------------------

for ceng=1:numel(net.layers)
  if ~strcmp(net.layers{ceng}.type,'conv'), continue; end
  net.layers{ceng}.filtersMomentum = zeros(size(net.layers{ceng}.filters), ...
    class(net.layers{ceng}.filters)) ;
  net.layers{ceng}.biasesMomentum = zeros(size(net.layers{ceng}.biases), ...
    class(net.layers{ceng}.biases)) ; 
  if ~isfield(net.layers{ceng}, 'filtersLearningRate')
    net.layers{ceng}.filtersLearningRate = 1 ;
  end
  if ~isfield(net.layers{ceng}, 'biasesLearningRate')
    net.layers{ceng}.biasesLearningRate = 1 ;
  end
  if ~isfield(net.layers{ceng}, 'filtersWeightDecay')
    net.layers{ceng}.filtersWeightDecay = 1 ;
  end
  if ~isfield(net.layers{ceng}, 'biasesWeightDecay')
    net.layers{ceng}.biasesWeightDecay = 1 ;
  end
end



%% 
if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for Gc=1:numel(net.layers)
    if ~strcmp(net.layers{Gc}.type,'conv'), continue; end
    net.layers{Gc}.filtersMomentum = gpuArray(net.layers{Gc}.filtersMomentum) ;
    net.layers{Gc}.biasesMomentum = gpuArray(net.layers{Gc}.biasesMomentum) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

info.train.loss = [] ;
info.test.cnn_loss = [] ;
info.test.cnn_IBP_loss = [] ;

info.train.loss_per_epoch=[];
info.test.cnn_loss_per_epoch=[];
info.test.cnn_IBP_loss_per_epoch=[];

lr = 0 ;
res = [] ;
tic;
for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to where we stopped
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file'), continue ; end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end

  
  %% 
 
  if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  end
  
  %% 
  train = opts.train ;
  label = opts.label ;

%   info.train.loss(end+1) = 0 ;
%   info.train.speed(end+1) = 0 ;

  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
    for l=1:numel(net.layers)
      if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
      net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
      net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
    end
  end
  
[H,W,~,number]=size(train);
  for t=1:opts.batchSize:number
    % get next image batch and labels
    batch_image = train(:,:,:,t:min(t+opts.batchSize-1, number)) ;
    batch_label = label(:,:,:,t:min(t+opts.batchSize-1, number)) ;
    batch_time = tic ;
    fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(number/opts.batchSize)) ;
    if opts.prefetch
      nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      batch_image = gpuArray(batch_image) ;
    end

    % backprop
    net.layers{end}.class = batch_label ;
    res = vl_simplenn_3D_differ_pzy(net, batch_image, one, res, ...
      'conserveMemory', opts.conserveMemory, ...
      'sync', opts.sync) ;

    % gradient step
    for l=1:numel(net.layers)
      if ~strcmp(net.layers{l}.type, 'conv'), continue ; end

      net.layers{l}.filtersMomentum = ...
        opts.momentum * net.layers{l}.filtersMomentum ...
          - (lr * net.layers{l}.filtersLearningRate) * ...
          (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
          - (lr * net.layers{l}.filtersLearningRate) / numel(opts.batchSize) * res(l).dzdw{1} ;

      net.layers{l}.biasesMomentum = ...
        opts.momentum * net.layers{l}.biasesMomentum ...
          - (lr * net.layers{l}.biasesLearningRate) * ...
          (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
          - (lr * net.layers{l}.biasesLearningRate) / numel(opts.batchSize) * res(l).dzdw{2} ;

      net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
      net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    end

    % print information
    batch_time = toc(batch_time) ;
    speed = opts.batchSize/batch_time ;
   
    index=fix(t/opts.batchSize)+1;
    
    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    imdff = bsxfun(@minus, res(end-1).x, batch_label) ;
    imdff= abs(imdff);
    imdff = imdff(:);

    SAD = mean(imdff);
%     psnr = 20*log10(255/rmse);
 %%  
    if opts.useGpu  
      SAD = gather(SAD) ; 
    end
%%    
    info.train.loss(index)= SAD;
    fprintf(' SAD %.6f ', ...
      info.train.loss(index)) ;
    fprintf('\n') ;

    % debug info
    if opts.plotDiagnostics
      figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
    end
  end % next batch
  
%% evaluation on validation set-PZY_3D_SRCNN
 [f1_size,~,~,f1_n]=size(net.layers{1}.filters );
 [f2_size,~,f2_channel,f2_n]=size(net.layers{3}.filters );
 [f3_size,~,f3_channel,~]=size(net.layers{5}.filters );

%% generate testing data
folder_test_L= './Image_database/3D-test_L/';
folder_test_R= './Image_database/3D-test_R/';
filepaths_test_L = dir(fullfile(folder_test_L,'*.bmp'));
filepaths_test_R=dir(fullfile(folder_test_R,'*.bmp'));

for ar = 1 : length(filepaths_test_L)
    
    image_test_L = imread(fullfile(folder_test_L,filepaths_test_L(ar).name));
    image_test_L = rgb2ycbcr(image_test_L);
    image_test_L = im2single(image_test_L(:, :, 1));
    image_test_L = modcrop(image_test_L, imdb.scale);
    
    image_test_R = imread(fullfile(folder_test_R,filepaths_test_R(ar).name));
    image_test_R = rgb2ycbcr(image_test_R);
    image_test_R = im2single(image_test_R(:, :, 1));
    image_test_R = modcrop(image_test_R, imdb.scale);
    
    [hei,wid] = size(image_test_R);
    image_test_R_down=imresize(image_test_R,1/imdb.scale,'bicubic');
    image_test_R_LR = imresize(image_test_R_down,[hei,wid],'bicubic');
    image_test_differ=image_test_L-image_test_R_LR;

           
    im_label_test = image_test_R;
       
    image_test_input=zeros(hei, wid);
    image_test_input(:,:)=image_test_differ(:,:);

if opts.useGpu
  image_test_input = gpuArray(image_test_input) ;
end
    %% conv1
    conv1_data = zeros(hei, wid, f1_n);
    conv1_data_relu = zeros(hei, wid, f1_n);
if opts.useGpu
  conv1_data = gpuArray(conv1_data) ;
  conv1_data_relu = gpuArray(conv1_data_relu) ;
end
for i1 = 1 : f1_n
    conv1_subfilter = reshape(net.layers{1}.filters(:,:,1,i1), f1_size, f1_size);
    conv1_data(:,:,i1) = imfilter(double(image_test_input(:,:)), double(conv1_subfilter), 'same', 'replicate');
    conv1_data_relu(:,:,i1) = max(conv1_data(:,:,i1) + net.layers{1}.biases(i1), 0);
end
    %% conv2
    conv2_data = zeros(hei, wid, f2_n);
    conv2_data_relu = zeros(hei, wid, f2_n);
if opts.useGpu
  conv2_data = gpuArray(conv2_data) ;
  conv2_data_relu = gpuArray(conv2_data_relu) ;
end
for i2 = 1 : f2_n
    for j2 = 1 : f2_channel
    conv2_subfilter = reshape(net.layers{3}.filters(:,:,j2,i2), f2_size, f2_size);
    conv2_data(:,:,i2) = conv2_data(:,:,i2) + imfilter(double(conv1_data_relu(:,:,j2)), double(conv2_subfilter), 'same', 'replicate');
    end
    conv2_data_relu(:,:,i2) = max(conv2_data(:,:,i2) + net.layers{3}.biases(i2), 0);
end
    %% conv3
    conv3_data = zeros(hei, wid);
if opts.useGpu
  conv3_data = gpuArray(conv3_data) ;
end
for i3 = 1 : f3_channel
    conv3_subfilter = reshape(net.layers{5}.filters(:,:,i3,1), f3_size, f3_size);
    conv3_data(:,:) = conv3_data(:,:) + imfilter(double(conv2_data_relu(:,:,i3)), double(conv3_subfilter), 'same', 'replicate');
end
%% SRCNN reconstruction
diff_SR = conv3_data(:,:) + net.layers{5}.biases;
if opts.useGpu  
      diff_SR = gather(diff_SR) ; 
end
im_SR= image_test_L-diff_SR;
clear diff_SR conv3_data conv2_data_relu conv2_data conv1_data  conv1_data_relu image_test_input conv1_subfilter conv2_subfilter conv3_subfilter
%% backprojection
maxIter=20;
[im_IBP_SR] = backprojection(im_SR, image_test_R_down, maxIter);
%% remove border
im_SR = shave(uint8(im_SR * 255), [imdb.scale, imdb.scale]);
im_label_ori = shave(uint8(im_label_test * 255), [imdb.scale, imdb.scale]);
im_IBP_SR= shave(uint8(im_IBP_SR * 255), [imdb.scale, imdb.scale]);

%% compute PSNR
info.test.cnn_IBP_loss(ar) = compute_psnr(im_label_ori,im_IBP_SR);
if epoch==1
imwrite(uint8(im_label_ori),['.\Data\3D_differ-result\ori', '-',filepaths_test_R(ar).name(1:end-4),'.png']);
end
if mod(epoch,10)==0
imwrite(uint8(im_IBP_SR),['.\Data\3D_differ-result\3d-diff-srcnn_IBP','-',filepaths_test_R(ar).name(1:end-4),'-epoch',num2str(epoch), '.png']);
end
end

  %% save
  
  info.train.loss_per_epoch(epoch) = mean(info.train.loss)  ;
  info.test.cnn_IBP_loss_per_epoch(epoch) = mean(info.test.cnn_IBP_loss)  ;
%% Move the CNN back to the CPU if it was trained on the GPU
if opts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end
toc;
  save(modelPath(epoch), 'net', 'info') ;
  figure(1) ; clf ;
  subplot(1,2,1) ;
  plot(1:epoch, info.train.loss_per_epoch, 'k') ; hold on ;
  xlabel('3d-diff-training epoch') ; ylabel('SAD_loss') ;
  grid on ;
  h=legend('diff-train') ;
  set(h,'color','none');
  title('training error on Middleburry database') ;

  subplot(1,2,2) ;
  plot(1:epoch,info.test.cnn_IBP_loss_per_epoch, '-g.') ;
  hleg=legend('Proposed_differCNN_IBP') ;
  set(hleg,'Location','best')
  xlabel('3d-diff-training epoch') ; ylabel('PSNR-dB') ;
  grid on ;
  title('testing error on Stereo database ') ;
  
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end
toc;



