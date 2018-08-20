export LD_LIBRARY_PATH=/usr/local/lib
RECORDINGS=data/recordings
BIN=./bin/extract
OUTFILE=

function ex
{
	$BIN --image $1 --flip
}

# if an input argument was supplied only run the extraction on that file
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
		OUT="$(ex $FRAME 2>&1)"
		if [ $? -ne 0 ]
		then
			continue
		fi

		echo $FRAME
		IFS=$'\n'
		for LINE in $OUT
		do
			echo " > $LINE"
		done

		#IFS=':' read -ra TAG <<< "$OUT"
		#ID="${TAG[0]}"
		#echo "in frame $FRAME we found TAG#$ID"
	done
done
