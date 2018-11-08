set nohidden3d
set autoscale

clusters = ARG1
centroids = ARG2
stats clusters

plot for [i=0:STATS_blocks-1] clusters index i u 1:2 w points pt 1 title "Cluster [".(i+1)."]", \
	for [i=0:STATS_blocks-1] centroids index i u 1:2 w points pt 5 ps 2 title "Centroid [".(i+1)."]"

