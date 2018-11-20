reset

stats 'results/clustering/silhouette.dat' name 'A' nooutput

set output outputfile
set autoscale
set style data linespoints
set style line 1 pt 7 ps 1.5
set grid
set arrow 1 from A_pos_max_y, graph 0.5 to A_pos_max_y, A_max_y fill
set label 1 at A_pos_max_y, graph 0.5 "max" center offset 0,-1
set xlabel "Number of clusters"
set ylabel "Mean silhouette coefficient"

plot "results/clustering/silhouette.dat" w linespoints ls 1 notitle
