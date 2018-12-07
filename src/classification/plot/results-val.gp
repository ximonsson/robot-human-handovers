reset

set output outputfile
set grid
unset xtics
set ytics nomirror
set border 2

set autoscale
set key right bottom

f = "results/classification/LR-0.0001__EP-20__BS-128__K-5__D-0.5_w2-t-0/"

set ylabel "Accuracy"
plot for [i=0:4] f . "acc_val.dat" index i w lines t "k = ".(i + 1)
