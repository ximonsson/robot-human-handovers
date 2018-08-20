export LD_LIBRARY_PATH=/usr/local/lib
RECORDINGS=data/recordings
BIN=./bin/extract


function ex
{
	$BIN --image $1 --flip
}

if [ "$#" -gt 0 ]
then
	ex $1
	exit
fi


# loop over all recording directories
for REC in $RECORDINGS/*
do
	echo $REC
	# loop over all RGB frames
	for FRAME in $REC/rgb/*.jpg
	do
		# extract data
		# if we didn't find any tag we continue to the next frame
		OUT=`ex $FRAME`
		if [ $? -ne 0 ]
		then
			continue
		fi

		IFS=':' read -ra TAG <<< "$OUT"
		ID="${TAG[0]}"
		echo "in frame $FRAME we found TAG#$ID"
	done
done


