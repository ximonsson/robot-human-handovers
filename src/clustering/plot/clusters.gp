set nohidden3d
set view 60,80
set autoscale
#set parametric

clusters = ARG1
centroids = ARG2
stats clusters

set multiplot layout 1,2 rowsfirst

# plot the clusters 3D and 2D with their centroids
splot for [i=0:STATS_blocks-1] clusters index i w points pt 1 title "Cluster [".(i+1)."]", \
	for [i=0:STATS_blocks-1] centroids index i w points pt 5 ps 2 title "Centroid [".(i+1)."]"

plot for [i=0:STATS_blocks-1] clusters index i u 1:2 w points pt 1 title "Cluster [".(i+1)."]", \
	for [i=0:STATS_blocks-1] centroids index i u 1:2 w points pt 5 ps 2 title "Centroid [".(i+1)."]"

unset multiplot
