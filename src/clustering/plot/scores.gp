set autoscale
set style data linespoints
#set pointtype 7 pointsize 1.5

set multiplot layout 1,2 rowsfirst

#set label "Opposite of the value of X on the K-means objective"
plot "results/clustering/scores.dat" notitle

#set label "Mean silhouette coefficient"
plot "results/clustering/silhouette.dat" notitle

unset multiplot
