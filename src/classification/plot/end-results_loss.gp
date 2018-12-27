reset

set grid
set xtics nomirror
set ytics nomirror
set border 2

set autoscale
set key right bottom

f = "results/classification/LR-0.0001__EP-20__BS-128__K-5__D-0.5_w2-t-0/loss.dat"

set ylabel "Loss"
set title "Loss"
plot for [i=0:4] f index i w lines notitle
