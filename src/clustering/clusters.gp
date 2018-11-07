set nohidden3d
set view 30,60
set autoscale
#set parametric

filename = ARG1
stats filename

set multiplot layout 1,2 rowsfirst

#set xdata 'X'
#set ydata 'Y'
#set zdata 'Z'

# plot the clusters 3D and 2D
splot for [i=0:STATS_blocks-1] filename index i w points pt 1 title "Cluster [".(i+1)."]"
plot for [i=0:STATS_blocks-1] filename index i u 1:2 w points pt 1 title "Cluster [".(i+1)."]"

# plot centroids

unset multiplot
