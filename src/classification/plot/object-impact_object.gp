reset
set autoscale
set output outputfile

f = "results/classification/pairwise_obj_acc.dat"
set view map
set ytics (\
	"beerglass" 0,\
	"bottle" 1,\
	"carrafe" 2,\
	"cup" 3,\
	"fork" 4,\
	"glass" 5,\
	"knife" 6,\
	"scissors" 7,\
	"spatula" 8,\
	"spoon" 9,\
	"wineglass" 10,\
	"woodenspoon" 11)
set xtics ("2" 0, "4" 1, "6" 2, "8" 3, "10" 4)

set xlabel "Training set size"

plot \
	f matrix u 2:1:($3) with image notitle, \
	f matrix u 2:1:($3 == 0 ? "" : sprintf("%.3f",$3/100)) w labels notitle
