reset

set autoscale
unset xtics
set ytics nomirror
set style fill solid
set border 2
set yrange [-0.25:1.0]

set style line 2 lc rgb "red" lt 4 lw 1 dt 2
set style line 3 lc rgb "green" lt 4 lw 1 dt 2

stats ARG1 nooutput

M = STATS_mean
indices = STATS_blocks

set multiplot layout 1,indices rowsfirst

stats ARG1 index 0
set title "Cluster [1]"
plot \
	ARG1 index 0 w boxes lc 0 notitle, \
	STATS_mean w lines ls 2 t "Cluster Mean", \
	M w lines ls 3 t "Sample Mean"

do for [i=1:indices-1] {
	stats ARG1 index i
	set title "Cluster [".(i+1)."]"
	plot \
		ARG1 index i w boxes lc i notitle, \
		STATS_mean w lines ls 2 notitle, \
		M w lines ls 3 notitle
}
unset multiplot
