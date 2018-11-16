reset

set output outputfile
set nohidden3d
set view 60,80
set autoscale
#set parametric

clusterfile = 'results/clustering/clusters_'
centroidsfile = 'results/clustering/centroids_'

clusters = clusterfile.ARG1.".dat"
centroids = centroidsfile.ARG1.".dat"
stats clusters nooutput

set multiplot layout 2,1

# plot the clusters 3D and 2D with their centroids
splot for [i=0:STATS_blocks-1] clusters index i w points pt 1 title "Cluster [".(i+1)."]", \
	for [i=0:STATS_blocks-1] centroids index i w points pt 5 ps 2 title "Centroid [".(i+1)."]"

plot for [i=0:STATS_blocks-1] clusters index i u 1:2 w points pt 1 title "Cluster [".(i+1)."]", \
	for [i=0:STATS_blocks-1] centroids index i u 1:2 w points pt 5 ps 2 title "Centroid [".(i+1)."]"

unset multiplot
