for d =403:403
  filename=strcat('E:\Project_Work_Doc\rar file\ICDAR 11,13\born digital image (web & email)\training set\image',num2str(d),'.png'); 
  outfile = strcat('E:\GENERATE_FEATURE\img900spComp15' ,num2str(d),'.png');
  fid = fopen('E:\MatlabCode\feature_matrix1.txt', 'w');
    I = imread(filename);
    I = rgb2gray(I);
    [L,N] = superpixels(I,800,'NumIterations', 100, 'Compactness',20, 'IsInputLab', false);
    figure
    BW = MyBoundaryMask(L);
    OverBW=imoverlay(I, BW,'green');
    %imshow(OverBW,'InitialMagnification',90);
    SLICClusterOutDataImg=strcat('E:\MatlabCode\SLIC_Image\SLICCluster800Compactness20OutputImg',num2str(d),'.png');
    imwrite(OverBW,SLICClusterOutDataImg);
    
    f1 = getframe;
    [x1, Map] = frame2im(f1);
    %imwrite(x1, outfile);
    %compute dimension of input image
    [row,col] = size(I);
    
    %count number of distinct regions
    ROI = unique(L);
    %count number of element in ROI array
    no_ROI = numel(ROI);
    %initiliaze array to store mean of all ROI
    mean_ROI = zeros(no_ROI, 1);
    % initilaize array to stote the standard deviation of all ROI
    sd_ROI = zeros(no_ROI, 1);
    meanX=zeros(no_ROI,1);
    meanY=zeros(no_ROI,1);
    feature_matrix=zeros(N,4);
    for x =1:no_ROI
     %find indices of pixels within specific ROI
       count_ROI = find(L== ROI(x));
       count = numel(count_ROI);
       pix_array = zeros(count,1);
       %compute the mean intensity of pixels in ROI
        a = 0;
        for i = 1:row
            for j = 1:col
                if(L(i,j) == ROI(x)) 
                   a= a+1;
                   pix_array(a) = I(i,j);
                end
            end
        end
        total_intensity = sum(pix_array);
        mean_intensity = (total_intensity)/(count);
        mean_ROI(x)= mean_intensity;
        sd_ROI(x) = std(pix_array);
        [P,Q]=find(L==ROI(x));
         meanX(x,1)=sum(P)/count;
         meanY(x,1)=sum(Q)/count;
         fprintf(fid, '%f\t %f\t %f\t %f \n',mean_ROI(x), sd_ROI(x), meanX(x,1), meanY(x,1));
         feature_matrix(x,1)=mean_intensity;
         feature_matrix(x,2)=sd_ROI(x);
         feature_matrix(x,3)=meanX(x);
         feature_matrix(x,4)=meanY(x);
        
    end
   %imwrite(I, outfile);
   %K-MEANS implementation
   fclose(fid);
  [Cluster, ClCentres, sumd, D] = kmeans(feature_matrix,floor(N/4)); 
  [uvalues, ~, uid] = unique(Cluster(:));  
  count_appear = accumarray(uid, 1);   %easiest way to calculate the histogram of uvalues
  linindices = accumarray(uid, (1:numel(Cluster))', [], @(idx) {idx});  %split linear indices according to uid
  valwhere = [num2cell(uvalues), linindices];  %concatenate
  valwhere(count_appear == 0, :) = [] ;   %remove count of 1
  L1=zeros(row,col);
  [row_valwhere,col_valwhere]=size(valwhere);
  %Creating new level matix against new cluster from K-MEANS
  for i1=1:row_valwhere
      t=valwhere{i1,1};
      [row_cell,col_cell]=size(valwhere{i1,2});
      for i2=1:row_cell
                  t1=valwhere{i1,2}(i2);
                  [x_i,y_i]=find(L==t1);
                  for i3=1:numel(x_i)
                      for j3=1:numel(y_i)
                          L1(x_i(i3),y_i(j3))=t;
                      end
                  end
      end
  end
 BW_cluster = MyBoundaryMask(L1);
% imshow(imoverlay(I, BW_cluster, 'green'),'InitialMagnification', 90);
 KmeanClusterOutDataImg=strcat('E:\MatlabCode\kmeans_cluster_image\kmeansClusterOutputImg',num2str(d),'.png');
 imwrite(imoverlay(I,BW_cluster,'green'),KmeanClusterOutDataImg);
% assigning new variable name to feature contain
 points_loc = feature_matrix;
% store x axis of corner points
 dbmean_coordinate = points_loc(:,1);
    
 %  store y axis of corner points
 dbsd_coordinate= points_loc(:,2); 
 % count number of elements of corner point array
 num = numel(points_loc);
 % as array has 2 coloumn x and y so divide by 2 
 num = num/4;
 % determine mimimum number of points to each cluster
 MinPts=ceil(log(num));

% compute KNN algorithm
[knn_idx, knn_dist] = knnsearch(points_loc, points_loc, 'K', MinPts);
% sum of maximum distance values from each points within its K-neighbors
clst_radius = sum(knn_dist(:,MinPts));
mean_radius = clst_radius/num;

% determine the epsilon
  epsilon=mean_radius;
%  S=importdata("feature_matrix1.txt");
% Run DBSCAN Clustering Algorithm
%   epsilon=5;
%   MinPts=3;
 IDX=DBscan(points_loc,epsilon,MinPts);
%% Plot Results
  PlotClusterinResult(points_loc,IDX);
  title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);
  L2=zeros(row,col);
        for i4=1:numel(IDX)
            t1=IDX(i4,1);
            if t1>0
            x1_i=find(IDX==t1);
            for i5=1:numel(x1_i)
               [x1_ii,y1_ii]=find(L==x1_i(i5));
               for i6=1:numel(x1_ii)
                   for j6=1:numel(y1_ii)
                       L2(x1_ii(i6),y1_ii(j6))=t1;
                   end
               end
            end
            end
        end
         BW_cluster_dbscan = MyBoundaryMask(L2);
 %imshow(imoverlay(I, BW_cluster_dbscan, 'green'),'InitialMagnification', 90);
 DBClusterOutDataImg=strcat('E:\MatlabCode\DB_image\DBClusterOutputImg',num2str(d),'.png');
 imwrite(imoverlay(I,BW_cluster_dbscan,'green'),DBClusterOutDataImg);
end