set nohidden3d
set view 60,80
set autoscale

clusters = ARG1
centroids = ARG2
stats clusters

splot for [i=0:STATS_blocks-1] clusters index i w points pt 1 title "Cluster [".(i+1)."]", \
	#for [i=0:STATS_blocks-1] centroids index i w points pt 5 ps 2 title "Centroid [".(i+1)."]"
