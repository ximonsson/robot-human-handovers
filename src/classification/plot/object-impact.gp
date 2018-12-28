reset
set output outputfile
set autoscale
set border 3
set grid
set xtics nomirror
set ytics nomirror
set xlabel "Number of objects"
set ylabel "Accuracy"
set key right bottom

set style data linespoints
set style line 1 pt 7 ps 1.5
set style line 2 pt 7 ps 1.5

plot "results/classification/pairwise_acc.dat" index 0 w linespoints ls 1 t "test accuracy", \
	"results/classification/pairwise_acc.dat" index 1 w linespoints ls 2 t "validation accuracy"
