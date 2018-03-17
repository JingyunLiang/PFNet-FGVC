imdb = load('data/Car/car_imdb.mat') ;

% -------------------------------------------------------------------------

maxNum = 500 ;
for i=1:numel(imdb.images.name)
    bbox = imdb.images.boxes{i};% height width
    imsize = imdb.images.size(i,:) ;

    isGood = (bbox(:,3)-bbox(:,1))>20 & (bbox(:,4)-bbox(:,2))>20;
    bbox = bbox(isGood,:);
    
    % remove duplicate ones in 14*14
    [dummy, uniqueIdx] = unique(round(bbox/16), 'rows', 'first');
    uniqueIdx = sort(uniqueIdx);
    bbox = bbox(uniqueIdx,:);
    
    % limit number for training
    if 1%imdb.images.set(i)~=3
        nB = min(size(bbox,1),maxNum);
    else
        nB = size(bbox,1);
    end
    
    imdb.images.boxes{i} = bbox(1:nB,:);
    i
end

mkdir('car_ims') ;


parfor i = 1:numel(imdb.images.name)
    rois_ = imdb.images.boxes{i} ;% y1(vertical) x1(horizonal)  y2 x2
    rois = [zeros(size(rois_,1),1) rois_(:,2) rois_(:,1) rois_(:,4) rois_(:,3)] ;% input (x1,y1,x2,y2)
    dlmwrite(fullfile([imdb.images.name{i}(1:end-4) '.txt']),rois,' ') ;
    i
end
