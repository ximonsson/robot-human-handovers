set autoscale
set style data linespoints
set style line 1 pt 7 ps 1.5

set multiplot layout 1,2 rowsfirst

set xlabel "Number of clusters"

set ylabel "Sum of squared distances of samples to their closests cluster center"
plot "results/clustering/scores.dat" w linespoints ls 1 notitle

set ylabel "Mean silhouette coefficient"
plot "results/clustering/silhouette.dat" w linespoints ls 1 notitle

unset multiplot
