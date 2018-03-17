function imdb = cars_get_database_SSW(varargin)
% Modified from 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% used to prepare the car-196 dataset of imdb.mat for MatCovNet. 
% imdb.images.boxes stores the proposed rois.


carsDir = '/raid/L/Fine-grained Dataset/Cars-196';
useCropped = false;
ifval = true;

if useCropped
    imdb.imageDir = fullfile(carsDir, 'images_cropped') ;
else
    imdb.imageDir = fullfile(carsDir);
end

imdb.maskDir = fullfile(carsDir, 'masks'); % doesn't exist
imdb.sets = {'train', 'val', 'test'};

load(fullfile(carsDir, 'cars_annos'));

% Class names
imdb.classes.name = class_names';


N = numel(annotations);

imdb.images.name = cell(N, 1);
imdb.images.id = 1:N;
imdb.images.label = zeros(1,N);
imdb.images.bounds = zeros(4, N);
imdb.images.set = 3.*ones(1, N);
imdb.images.difficult = false(1, N) ; 

% Image names
for i=1:numel(annotations)

    imdb.images.name{i} = annotations(i).relative_im_path;

    % Class labels
    imdb.images.label(i) = annotations(i).class;

    % Bounding boxes
    
    imdb.images.bounds(:,i) = round([annotations(i).bbox_x1 annotations(i).bbox_y1 annotations(i).bbox_x2 annotations(i).bbox_y2]');

    % Image sets
    if(~annotations(i).test)
        imdb.images.set(i) = 1;
    end


end

% Class labels
% modified by Michael
classLabel = imdb.images.label ;
imdb.images.label = -ones(numel(imdb.classes.name),numel(classLabel));
for i=1:numel(classLabel)
    imdb.images.label(classLabel(i),i)=1;
end

% Image size, update  it in CubProposalSSW_par.m
imdb.images.size = [] ;

% Image size
imdb.images.size = [] ;

% add image files to imdb
% imdb.images.image = vl_imreadjpeg(strcat([imdb.imageDir filesep], imdb.images.name) , 'NumThreads', 8 );


if(ifval)

trainSize = numel(find(imdb.images.set==1));
validSize = round(trainSize/3);

trainIdx = find(imdb.images.set==1);

% set 1/3 of train set to validation
valIdx = trainIdx(randperm(trainSize, validSize));
imdb.images.set(valIdx) = 2;

end


imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;


% add by Michael
% calculate proposals using SSW
addpath('SelectiveSearchCodeIJCV');
addpath(fullfile('SelectiveSearchCodeIJCV', 'Dependencies'));
imdb = CarProposalSSW_par(imdb, carsDir) ;

save('data/Car/car_imdb.mat','-struct', 'imdb', '-v7.3');

