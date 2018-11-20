reset

set output outputfile
set grid
set xtics nomirror
set ytics nomirror
set border 3

set autoscale
set xrange[0:200]
set ylabel "Accuracy"
set key right bottom

plot \
	"results/classification/LR-1e-05__EP-10__BS-128__K-5/acc_test.dat" w lines t "128", \
	"results/classification/LR-1e-05__EP-10__BS-64__K-5/acc_test.dat" w lines t "64", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/acc_test.dat" w lines t "32", \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/acc_test.dat" w lines t "16", \
	"results/classification/LR-1e-05__EP-10__BS-8__K-5/acc_test.dat" w lines t "8"
