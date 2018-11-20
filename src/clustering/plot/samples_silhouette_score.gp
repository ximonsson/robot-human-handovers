reset

set output outputfile
set autoscale
unset xtics
set ytics nomirror
set style fill solid
set border 2
set yrange [-0.25:1.0]
set ylabel "Silhouette coefficient per sample"

set style line 2 lc rgb "#FF0A0A" lt 4 lw 1 dt 2
set style line 3 lc rgb "#005A05" lt 4 lw 1 dt 2

stats ARG1 nooutput
M = STATS_mean
indices = STATS_blocks

array sizes[indices]
sizes[1] = 0
do for [i=2:indices] {
	stats ARG1 index (i-2) nooutput
	sizes[i] = STATS_records + sizes[i-1] + 100
}

xcoord(n, i) = n + sizes[i+1]
plot \
	for [i=0:indices-1] ARG1 index i u (xcoord($0, i)):1 w boxes t "Cluster [".(i+1)."]", \
	M w lines ls 2 t "Mean"
