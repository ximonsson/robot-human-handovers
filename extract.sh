DATA=data
RECORDINGS=$DATA/recordings
TRAINING=$DATA/training
BIN=./bin/extract_handover
DATAFILE="$TRAINING/raw"

function ex
{
	LD_LIBRARY_PATH=/usr/local/lib $BIN --image $1 --flip
}

function progress
{
	DIR=$1
	FRAME=$2
	TOT=$3

	PROGRESS=$(echo "scale=2; $FRAME/$TOT*100" | bc)
	printf "\r> Processing [ $DIR ] ... $PROGRESS%%"
}

# if an input argument was supplied only run the extraction on that file
if [ "$#" -gt 0 ]
then
	ex $1
	exit
fi

# create directory if it does not exist and remove all old training data
if [ ! -d $TRAINING ]
then
	mkdir -p $TRAINING
elif [ -f $DATAFILE ]
then
	rm $DATAFILE
fi

# loop over all recording directories
for REC in $RECORDINGS/*
do
	IDS=() # item ids that we find in the frames
	FRAMES=0 # total amount of frames with handovers in the recording
	FRAMES_TOTAL=`ls -1 $REC/rgb/*.jpg | wc -l`

	# loop over all RGB frames
	i=0
	for FRAME in $REC/rgb/*.jpg
	do
		# search data in frame
		OUT="$(ex $FRAME 2>&1)"
		RET=$?
		((i++))
		progress $REC $i $FRAMES_TOTAL

		# if no handover was found continue over to the next one
		if [ $RET -ne 0 ]
		then
			continue
		else
			FRAMES=$((FRAMES + 1))
		fi

		# parse the output of the application
		IFS=$'\n' read -rd '' -a LINES <<< $"$OUT"

		# parse ID of the tag
		IFS=':' read -ra TAG <<< "${LINES[0]}"
		IDS+=("${TAG[0]}")

		# write data to file
		echo "#$FRAME" >> $DATAFILE
		echo "${LINES[0]}" >> $DATAFILE
		echo "${LINES[1]}" >> $DATAFILE
		echo "${LINES[2]}" >> $DATAFILE
	done

	IDS=`printf "%s\n" ${IDS[@]} | sort -u`
	IFS=$'\n' read -rd '' -a IDS <<< $"$IDS"
	echo ": $FRAMES frames with a total of ${#IDS[@]} objects found"
done
