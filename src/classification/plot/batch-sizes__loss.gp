reset

set output outputfile
set grid
set xtics nomirror
set ytics nomirror
set border 3

set xrange [0:200]
set ylabel "Cross entropy"
plot \
	"results/classification/LR-1e-05__EP-10__BS-128__K-5/loss.dat" w lines t "128", \
	"results/classification/LR-1e-05__EP-10__BS-64__K-5/loss.dat" w lines t "64", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/loss.dat" w lines t "32", \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/loss.dat" w lines t "16", \
	"results/classification/LR-1e-05__EP-10__BS-8__K-5/loss.dat" w lines t "8"
