reset

set grid
set xtics nomirror
set ytics nomirror
set border 2

set autoscale
#set xrange[0:200]
set key right bottom

set multiplot layout 3,1 rowsfirst

f = "results/classification/LR-0.0001__EP-20__BS-128__K-5__D-0.5_w2-t-0/"

set ylabel "Loss"
set title "Loss"
plot for [i=0:4] f . "loss.dat" index i w lines notitle

set ylabel "Accuracy"
set title "Accuracy Validation"
plot for [i=0:4] f . "acc_val.dat" index i w lines notitle

set title "Accuracy Test"
plot for [i=0:4] f . "acc_test.dat" index i w lines notitle

unset multiplot
