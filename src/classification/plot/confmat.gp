reset

set output outputfile
set palette rgbformula -7,2,-7

set ylabel "True value"
set xlabel "Predicted value"

f = "results/classification/LR-1e-05__EP-20__BS-16__K-5/confusion_matrix.dat"

set view map
plot \
	f index 0 matrix columnheaders rowheaders u 1:2:3 with image notitle, \
	f index 0 matrix columnheaders rowheaders u 1:2:($3 == 0 ? "" : sprintf("%g",$3)) w labels notitle
