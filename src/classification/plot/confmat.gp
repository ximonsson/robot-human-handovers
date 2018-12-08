reset

set output outputfile
set palette defined (0 "#ffffff", 500 "#3388cc")
set ylabel "True value"
set xlabel "Predicted value"

f = "results/classification/LR-0.0001__EP-20__BS-128__K-5__D-0.5_w2-t-0/confusion_matrix__best.dat"

set view map
plot \
	f index 0 matrix rowheaders columnheaders u 1:2:3 with image notitle, \
	f index 0 matrix rowheaders columnheaders u 1:2:($3 == 0 ? "" : sprintf("%g",$3)) w labels notitle
