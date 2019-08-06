%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

for d = 1:1
     filename=strcat('E:\Project_Work_Doc\rar file\ICDAR 11,13\born digital image (web & email)\training set\image',num2str(d),'.png'); 
  outfile = strcat('E:\GENERATE_FEATURE\img700spComp20_' ,num2str(d),'.png');
    % create directory for each input image
    %mkdir('D:\H-DIBCO16 +maindir\LBP\cornerpoint\cluster_member_image\', num2str(d));
    %outfile1 = strcat('D:\H-DIBCO16 +maindir\LBP\cornerpoint\cluster_index\cluster_idx', num2str(d), '.txt');
    I = imread(filename);    
    I = rgb2gray(I);
    
    % detect corner points of input image
    points = detectMinEigenFeatures(I);
    imshow(I), hold on
    plot(points);
    
    % access the location of corner points
    points_loc = points.Location;
    
    % store x axis of corner points
    x_coordinate = points_loc(:,1);
    
    % store y axis of corner points
    y_coordinate= points_loc(:,2);
    
    % round off x_coordinate & y_coordinate
%     x_roundoff = round(x_coordinate);
%     y_roundoff = round(y_coordinate);
   
%     create two blank image of size I
%     [r,c] = size(I);
%     blank = zeros(r,c);
%     blank_plot = zeros(r,c);
%     idx = 1;
%   % plot the corner points into this blank image    
%     num = numel(x_roundoff);
%     for i = 1:num
%         a = x_roundoff(i);
%         b = y_roundoff(i);
%         blank(b,a) = 255;
%     end
    
 % initialize the file pointer to store cluster file for each image
%  fp = fopen(outfile1, 'w');
    %% Load Data
%  save('cornerpoints.mat',points_loc, '%f');
    
% data=load('cornerpoints');
% points_loc=data.points_loc;

% count number of elements of corner point array
num = numel(points_loc);
% as array has 2 coloumn x and y so divide by 2 
num = num/2;

%--------------------------------------------------
% % compute optimized epsilon for corner point set
% steps: 1. compute the distance between every pair of corner points
%        2. sort the distance values and draw a plot
%--------------------------------------------------
% % create a distance matrix to store distance values of corner points
% sz = (num*(num-1))/2;
% % sz = (num-1);
% dist_mat = zeros(sz, 1);
% count = 0;
% % compute pairwise distance
% for idx1 = 1:(num-1)
%     for idx2 = (idx1+1) : num
%         count = count +1;
%         dist_mat(count) = sqrt(((x_coordinate(idx1)- x_coordinate(idx2))^2) + ((y_coordinate(idx1)- y_coordinate(idx2))^2));
%     end 
% end
% 
% % compute the distance betwen pairwise points
% % for idx1 = 1:sz
% %     dist_mat(idx1) = sqrt(((x_coordinate(idx1+1)- x_coordinate(idx1))^2) + ((y_coordinate(idx1+1)- y_coordinate(idx1))^2));
% % end
% % sort the distance values
%  dist_mat_sort = sort(dist_mat);
% 
% % find index of distance matrix
% idx = find(dist_mat_sort);
% 
% sz_slope = sz-1;
% % initiliaze a array of slope
% slope_mat = zeros(sz_slope,1);
% 
% % compute slope of each points with its neighbor
%  for i = 1:sz_slope
%      x_idx = idx(i+1)-idx(i);
%      y_idx = dist_mat_sort(i+1) - dist_mat_sort(i);
%      slope = y_idx/x_idx;
%      slope_mat(i) = slope;
%  end
%  
%  % compute mean and standard deviation of non-zero slopes
%  nz = nonzeros(slope_mat);
%  mean_slope = mean(nz);
%  sd_slope = std(nz);
%  
%  % find the distance value correspond to slope which meet the criteria
%  for slope_idx = 1:sz_slope
%      if(slope_mat(slope_idx)>(mean_slope + sd_slope))
%          eps_slope = slope_mat(slope_idx);
%          break;
%      end
%  end
% compute KNN algorithm
% [knn_idx, knn_dist] = knnsearch(points_loc, points_loc, 'K', 7);
% 
% % sum of maximum distaince values
% sum_radius = sum(knn_dist(:,7));
% mean_radius = sum_radius/num;
% % convert knn distance matrix into 1-D array
% knn_dist_transpose = transpose(knn_dist);
% knn_plot = reshape(knn_dist_transpose, [(num*7), 1]);
% 
% % sort knn plot array
% knn_plot_sort = sort(knn_plot(knn_plot~=0));
% 
% % find the number of elements in k-dist plot
% Kdist_no = length(knn_plot_sort);
% 
% % find size of slope array
% sz_slope = Kdist_no - 1;
% 
% % Initialize the slope array 
% slope_mat = zeros(sz_slope, 1);
% 
% % compute the index of k-dist plot
% Kdist_idx = find(knn_plot_sort>=0);
% 
% % compute slope of each point with its neighbor in k-dist plot
% for i = 1:sz_slope
%      x_idx = Kdist_idx(i+1) - Kdist_idx(i);
%      y_idx = knn_plot_sort(i+1)- knn_plot_sort(i);
%      slope = y_idx/x_idx;
%      slope_mat(i) = slope;
% end
%  
%  % compute mean and standard deviation of non-zero slopes
%  nz = nonzeros(slope_mat);
%  mean_slope = mean(nz);
%  sd_slope = std(nz);
%  
%  % find the distance value correspond to slope which meet the criteria
%  for slope_idx = 1:sz_slope
%      if(slope_mat(slope_idx)>(mean_slope + (sd_slope)))
%          eps_slope = slope_mat(slope_idx);
%          break;
%      end
%  end
% 
%  epsilon_dbscan = find(slope_mat == eps_slope);
%% Run DBSCAN Clustering Algorithm

% determine mimimum number of points to each cluster
MinPts=ceil(log(num));

% compute KNN algorithm
[knn_idx, knn_dist] = knnsearch(points_loc, points_loc, 'K', MinPts);

% sum of maximum distance values from each points within its K-neighbors
clst_radius = sum(knn_dist(:,MinPts));
mean_radius = clst_radius/num;

% determine the epsilon
epsilon=mean_radius;

IDX=DBscan(points_loc,epsilon,MinPts);
% fprintf(fp, '%d\n', IDX);

% count number of distinct cluster from cluster index array
cluster_distinct = unique(nonzeros(IDX));

% find number of element from distinct cluster array (number of cluster)
no_clst = numel(cluster_distinct);

%find index of each datapoint cluster wise
 for i = 1:no_clst
     idx_clst = find(IDX == cluster_distinct(i));
     clst_member_x = x_coordinate(idx_clst);
     clst_member_y = y_coordinate(idx_clst);
     
     % create number of directory as number of cluster for each image
    % mkdir('D:\H-DIBCO16 +maindir\LBP\cornerpoint\cluster_member_image\1\', num2str(i));
     
     % store x and y coordinate of each points of each cluster
%      outfile_x = strcat('outfile\cluster_x', num2str(i), '.txt');
%      outfile_y = strcat('outfile\cluster_y', num2str(i), '.txt'); 

     outfile_x = fullfile(outfile, 'cluster_x');
     outfile_xx = strcat(outfile_x,num2str(i), '.txt');
     outfile_y = fullfile(outfile, 'cluster_y');
     outfile_yy = strcat(outfile_y,num2str(i), '.txt');
     % initialize file pointer
     fp_x = fopen(outfile_xx, 'a+');
     fprintf(fp_x, '%f\n', clst_member_x);
     fp_y = fopen(outfile_yy, 'a+');
     fprintf(fp_y, '%f\n', clst_member_y);
     
 end
       
%% Plot Results
 PlotClusterinResult(points_loc, IDX);
 title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);
 end
