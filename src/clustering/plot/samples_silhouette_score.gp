reset

set style data histogram
set style histogram clustered gap 5
set style fill solid

stats ARG1 nooutput

#set yrange [-0.25:1]
set xtics nomirror
set ytics nomirror
set ylabel "Silhouette sample score"

#plot newhistogram "one", ARG1 index 0 title "hej",
	#ARG1 index 1 title "tja"

plot ARG1 index 0 w histogram t col

# =========================================================================

#unset xtics

#set tmargin 3
#set bmargin 3
#set rmargin 0

#set multiplot layout 1,STATS_blocks rowsfirst

#set ytics nomirror
#set ylabel "Silhouette sample score"

#plot ARG1 index 0 title "Cluster [1]"
#unset ytics
#unset ylabel
#set lmargin 0
#
#do for [i=1:STATS_blocks-1] {
	#plot ARG1 index i title "Cluster [".(i+1)."]"
#}

#unset multiplot
