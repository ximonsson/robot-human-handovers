reset

set output outputfile
set autoscale
set grid
set border 3
set style histogram gap 1
set style line 1 pt 7 ps 1.5
set style fill solid 0.5 border 0

set xtics ("1" 0, "2" 1, "3" 2) nomirror
set ytics nomirror

set xlabel "Principal component"
set ylabel "Percentage of variance"

plot "results/clustering/pca.dat" u ($2 * 100) w histogram lc "blue" notitle, \
	"" u ($2 * 100) w linespoints ls 1 notitle
