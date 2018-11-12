reset

set output outputfile
set autoscale
set style data linespoints
set style line 1 pt 7 ps 1.5
set grid

stats 'results/clustering/silhouette.dat' name 'A' nooutput

set multiplot layout 2,1 rowsfirst

set xlabel "Number of clusters"

#set label "Mean silhouette coefficient"
set arrow 1 from A_pos_max_y, graph 0.5 to A_pos_max_y, A_max_y fill
set label 1 at A_pos_max_y, graph 0.5 "max" center offset 0,-1
plot "results/clustering/silhouette.dat" w linespoints ls 1 notitle

#set label "Sum of squared distances of samples to their closests cluster center"
set arrow 1 from A_pos_max_y, graph 0.5 to A_pos_max_y, 3000 fill
set label 1 at A_pos_max_y, graph 0.5 "best fit" center offset 0,1
plot "results/clustering/scores.dat" w linespoints ls 1 notitle

unset multiplot
