reset

stats 'results/clustering/silhouette.dat' name 'A' nooutput

set output outputfile
set autoscale
set style data linespoints
set style line 1 pt 7 ps 1.5
set grid
set xlabel "Number of clusters"
set ylabel "Sum of squared distances"

set arrow 1 from A_pos_max_y, graph 0.5 to A_pos_max_y, 3000 fill
set label 1 at A_pos_max_y, graph 0.5 "best fit" center offset 0,1
plot "results/clustering/scores.dat" w linespoints ls 1 notitle
