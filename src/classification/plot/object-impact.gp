reset
set autoscale
set output outputfile

f = "results/classification/pairwise_acc.dat"
set view map

plot \
	f matrix rowheaders columnheaders u 1:2:3 with image notitle, \
	f matrix rowheaders columnheaders u 1:2:($3 == 0 ? "" : sprintf("%.3f",$3/100)) w labels notitle

