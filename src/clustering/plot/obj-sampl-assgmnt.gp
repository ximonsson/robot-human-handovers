reset

set output outputfile

set ylabel "Object"
set xlabel "Cluster"

f='results/clustering/object-sample-assignments_6_norm.dat'

set view map
plot \
	f matrix columnheaders rowheaders u 1:2:3 with image notitle, \
	f matrix columnheaders rowheaders u 1:2:($3 == 0 ? "" : sprintf("%.2f",$3)) w labels notitle
