# Image-Segmentation-Using-DBSCAN-And-Kmeans-Clustering-Algorithm

In this work we present a general unsupervised image segmentation  technique
based on earlier proposed super-pixel based segmentation  along with k-mean
clustering  and Density-based spatial clustering  of applications with noise
(DBSCAN) to get satisfactory result. However, methods that generate super-pixels results into the over-segmentation image. So, the new way arrangement
is required to get proper slicing. There is a need to subsequently merge super-
pixels  with the same visual content. In our case, we tried to extract
feature from each super-pixels like mean intensity, standard deviation, average
coordinate distance. Using these feature, generate new cluster with K-mean and
DBSCAN  to reduce the number of clusters which is comparatively less than
that of numbers of super-pixels generated before.




Text detection from images containing text is a nontrivial research problem. As,
text strokes are mostly homogeneous, text may be detected by identifying homo-
geneous regions followed by appropriate analysis. In this dissertation, we present
a novel image segmentation approach for homogeneous region detection by super-
pixel grouping. Firstly, superpixels are generated from an input color image using
simple linear iterative clustering (SLIC) algorithm. Then, features are extracted
from these superpixels and subsequently grouped into clusters using DBSCAN
algorithm. A cluster obtained in this way is basically a homogeneous connected
component that is considered as a potential candidate text region. From the
experimental results, it may be stated that this work is a small yet valuable ex-
tension to the study of image analysis using super-pixels and it may enhance
current art of work in some sense to liberate the eld of image informatics.





