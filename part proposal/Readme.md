## This directory is used to generate rois for PFNet. 


1, download [Selective Search Window](https://koen.me/research/selectivesearch/) and extract it here. It is a Matlab toolbox for SSW.

2, run `Car_get_database_SSW.m` to get a `car_imdb.mat` file for the dataset, which contains image data, rois and other metadata. Please make sure you have moved images to a suitbale directory. By the way, in fact, `CarProposalSSW_par.m` is called to generate rois.

3, `roisWarpperforPytorch_generatetxt.m` uses `car_imdb.mat` to generate `.txt` file for Pytorch. Generated rois of CUB-200-2011, Stanford Cars and FGVC-Aircraft are [provided](https://drive.google.com/open?id=18DWMrK2WVEMGzRdMpgqgNiRbWOTtRwnP). Here is an example:
```
0 2 2 1024 768
0 194 76 336 258
0 218 2 638 458
0 2 16 1024 454
0 638 466 792 580
0 2 318 1024 768
0 652 404 1024 768
```
Each line represents a proposed bounding box. `0 2 2 1024 768` are identifier, x1(horizonal), y1(vertical), x2 and y2 respectively.

