reset

set output outputfile
set border 3
set xtics nomirror
set ytics nomirror
set grid
set autoscale
set key r c

set ylabel "Accuracy"
plot \
	"results/classification/LR-0.1__EP-20__BS-32__K-5/acc_test.dat" index 0 w lines t "0.1", \
	"results/classification/LR-0.001__EP-20__BS-32__K-5/acc_test.dat" index 0 w lines t "0.001", \
	"results/classification/LR-0.0001__EP-20__BS-32__K-5/acc_test.dat" index 0 w lines t "0.0001", \
	"results/classification/LR-1e-05__EP-20__BS-32__K-5/acc_test.dat" index 0 w lines t "1e-05"
	#"results/classification/LR-0.01__EP-20__BS-32__K-5/acc_test.dat" index 0 w lines t "0.01", \
