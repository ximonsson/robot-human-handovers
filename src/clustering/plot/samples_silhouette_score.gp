reset

set autoscale
unset xtics
set ytics nomirror
set style fill solid
set border 2
set yrange [-0.25:1.0]

stats ARG1 nooutput

M = STATS_mean
indices = STATS_blocks

set multiplot layout 1,indices rowsfirst
do for [i=0:indices-1] {
	stats ARG1 index i
	set title "Cluster [".(i+1)."]"
	plot ARG1 index i w boxes notitle, STATS_mean w line lw 3 t "Cluster Mean", M w line lw 3 t "Sample Mean"
}
unset multiplot
