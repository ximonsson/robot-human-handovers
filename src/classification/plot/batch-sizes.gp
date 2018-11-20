reset

set output "tex/plot_bs.tex"
set border 2
set xtics nomirror
set ytics nomirror
set grid

set title "Loss, validation and test accuracy for different batch sizes and learning rate 1e-5"

set xrange [0:200]

set multiplot layout 3,1


set ylabel "Cross entropy"
plot \
	"results/classification/LR-1e-05__EP-10__BS-128__K-5/loss.dat" w lines t "128", \
	"results/classification/LR-1e-05__EP-10__BS-64__K-5/loss.dat" w lines t "64", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/loss.dat" w lines t "32", \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/loss.dat" w lines t "16", \
	"results/classification/LR-1e-05__EP-10__BS-8__K-5/loss.dat" w lines t "8"

set autoscale

unset title
set ylabel "Validation accuracy"
plot \
	"results/classification/LR-1e-05__EP-10__BS-128__K-5/acc_val.dat" w lines t "128", \
	"results/classification/LR-1e-05__EP-10__BS-64__K-5/acc_val.dat" w lines t "64", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/acc_val.dat" w lines t "32", \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/acc_val.dat" w lines t "16", \
	"results/classification/LR-1e-05__EP-10__BS-8__K-5/acc_val.dat" w lines t "8"

set ylabel "Test accuracy"
plot \
	"results/classification/LR-1e-05__EP-10__BS-128__K-5/acc_test.dat" w lines t "128", \
	"results/classification/LR-1e-05__EP-10__BS-64__K-5/acc_test.dat" w lines t "64", \
	"results/classification/LR-1e-05__EP-10__BS-32__K-5/acc_test.dat" w lines t "32", \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/acc_test.dat" w lines t "16", \
	"results/classification/LR-1e-05__EP-10__BS-8__K-5/acc_test.dat" w lines t "8"

unset multiplot
