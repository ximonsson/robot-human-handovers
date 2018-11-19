reset

set palette rgbformula -7,2,-7
#set cbrange [0:5]
#set cblabel "Score"

set ylabel "Predicted value"
set xlabel "True value"

set view map
plot \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/confusion_matrix.dat" matrix columnheaders rowheaders u 1:2:3 with image notitle, \
	"results/classification/LR-1e-05__EP-10__BS-16__K-5/confusion_matrix.dat" matrix columnheaders rowheaders u 1:2:($3 == 0 ? "" : sprintf("%g",$3)) w labels notitle

