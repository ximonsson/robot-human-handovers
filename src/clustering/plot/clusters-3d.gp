reset
set output outputfile
set nohidden3d
set view 30,75
set autoscale

clusterfile = 'results/clustering/clusters_'
centroidsfile = 'results/clustering/centroids_'

clusters = clusterfile.ARG1.".dat"
centroids = centroidsfile.ARG1.".dat"
stats clusters nooutput

splot for [i=0:STATS_blocks-1] clusters index i w points pt 1 notitle, \
	for [i=0:STATS_blocks-1] centroids index i w points pt 5 ps 2 lc rgb "black" notitle
