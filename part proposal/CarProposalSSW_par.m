function  imdb = CarProposalSSW_par(imdb, cubDir)
% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
addpath('Dependencies');

fprintf('Demo of how to run the code for:\n');
fprintf('   J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   IJCV 2013\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
%     fprintf('   
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
sigma = 0.8;

% After segmentation, filter out boxes which have a width/height smaller
% than minBoxWidth (default = 20 pixels).
minBoxWidth = 0;%20

% Comment the following three lines for the 'quality' version
% colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
% simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
% ks = ks(1:2);

% Test the boxes
% load('GroundTruthVOC2007test.mat'); % Load ground truth boxes and images and image names
fprintf('After box extraction, boxes smaller than %d pixels will be removed\n', minBoxWidth);
fprintf('Obtaining boxes for Cub:\n');
totalTime = 0;

imsize = zeros(numel(imdb.images.name),3) ;
parfor i=1:length(imdb.images.name)
    fprintf('Selective Search for %d image\n', i);
    
    boxesT = [];
    priorityT = [];
    
    % VOCopts.img
    im = imread(fullfile(cubDir,imdb.images.name{i}));
    if size(im,3) == 1
        im = cat(3, im, im, im) ;
        imwrite(im, fullfile(cubDir,imdb.images.name{i})) ;
    end 
    imsize(i,:) = size(im) ;
    
    imageScale =1 ;% deal with large images
    if size(im,1) >=1000 || size(im,2) >=1000
        im = imresize(im,0.5) ;
        imageScale = 0.5 ;
    end
    
    idx = 1;
    for j=1:length(ks)
        k = ks(j); % Segmentation threshold k
        minSize = k; % We set minSize = k
        for n = 1:length(colorTypes)
            colorType = colorTypes{n};
            tic;
            [boxesTT blobIndIm blobBoxes hierarchy priorityTT] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
            totalTime = totalTime + toc;
            idx = idx + 1;
            
            boxesT = [boxesT ;boxesTT];
            priorityT = [priorityT ; priorityTT];
        end
    end
    
    priority = priorityT; % Concatenate priorities
    
    % Do pseudo random sorting as in paper
    priority = priority .* rand(size(priority));
    [priority sortIds] = sort(priority, 'ascend');
    boxesT = boxesT(sortIds,:);
    
    % add by Michael
    boxScores{i} = priority;
    boxes{i} =  boxesT/imageScale; % Concatenate boxes from all hierarchies
    
end
fprintf('\n');

%%
tic
for i=1:length(boxes)
    [boxes{i} boxScores{i}] = FilterBoxesWidth(boxes{i}, minBoxWidth, boxScores{i});
    [boxes{i} boxScores{i}]= BoxRemoveDuplicates(boxes{i}, boxScores{i});
end
totalTime = totalTime + toc;

imdb.images.boxes = boxes;
imdb.images.boxScores = boxScores;

imdb.images.size = imsize(:,1:2) ;

fprintf('Time per image: %.2f\nNow evaluating the boxes on Cub...\n', totalTime ./ length(imdb.images.name));

% %%
% [boxAbo boxMabo boScores avgNumBoxes] = BoxAverageBestOverlap(gtBoxes, gtImIds, boxes);
% 
% fprintf('Mean Average Best Overlap for the box-based locations: %.3f\n', boxMabo);