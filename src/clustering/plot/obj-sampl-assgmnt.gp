reset

set output outputfile
#set palette rgbformula -7,2,-7

set ylabel "Object"
set xlabel "Cluster"

f='results/clustering/object-sample-assignments_6.dat'

set view map
plot \
	f matrix columnheaders rowheaders u 1:2:3 with image notitle, \
	f matrix columnheaders rowheaders u 1:2:($3 == 0 ? "" : sprintf("%g",$3)) w labels notitle
