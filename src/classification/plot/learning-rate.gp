reset

set output "tex/plot_lr.tex"
set border 2
set xtics nomirror
set ytics nomirror
set grid
set xrange [0:200]
set title "Loss, validation and test accuracy for different learning rates and batch size 32"

set multiplot layout 3,1

set yrange [0:2]

set ylabel "Cross entropy"
plot \
	"results/classification/LR-0.1__EP-10__BS-32__K-5/loss.dat" w lines t "0.1", \
	"results/classification/LR-0.01__EP-10__BS-32__K-5/loss.dat" w lines t "0.01", \
	"results/classification/LR-0.001__EP-10__BS-32__K-5/loss.dat" w lines t "0.001", \
	"results/classification/LR-0.0001__EP-10__BS-32__K-5/loss.dat" w lines t "0.0001", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/loss.dat" w lines t "1e-05"

set autoscale

unset title
set ylabel "Validation accuracy"
plot \
	"results/classification/LR-0.1__EP-10__BS-32__K-5/acc_val.dat" w lines t "0.1", \
	"results/classification/LR-0.01__EP-10__BS-32__K-5/acc_val.dat" w lines t "0.01", \
	"results/classification/LR-0.001__EP-10__BS-32__K-5/acc_val.dat" w lines t "0.001", \
	"results/classification/LR-0.0001__EP-10__BS-32__K-5/acc_val.dat" w lines t "0.0001", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/acc_val.dat" w lines t "1e-05"

set ylabel "Test accuracy"
plot \
	"results/classification/LR-0.1__EP-10__BS-32__K-5/acc_test.dat" w lines t "0.1", \
	"results/classification/LR-0.01__EP-10__BS-32__K-5/acc_test.dat" w lines t "0.01", \
	"results/classification/LR-0.001__EP-10__BS-32__K-5/acc_test.dat" w lines t "0.001", \
	"results/classification/LR-0.0001__EP-10__BS-32__K-5/acc_test.dat" w lines t "0.0001", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/acc_test.dat" w lines t "1e-05"

unset multiplot
