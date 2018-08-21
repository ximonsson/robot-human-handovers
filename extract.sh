DATA=data
RECORDINGS=$DATA/recordings
TRAINING=$DATA/training
BIN=./bin/extract_handover

function ex
{
	LD_LIBRARY_PATH=/usr/local/lib $BIN --image $1 --flip
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
elif [ "$(ls -A $TRAINING)" ]
then
	rm $TRAINING/*
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

		# parse the output of the application
		echo $FRAME
		IFS=$'\n' read -rd '' -a LINES <<< $"$OUT"

		# parse ID of the tag
		IFS=':' read -ra TAG <<< "${LINES[0]}"
		ID="${TAG[0]}"

		# write data to file
		DATAFILE="$TRAINING/$ID"
		echo "#$FRAME" >> $DATAFILE
		echo "${LINES[0]}" >> $DATAFILE
		echo "${LINES[1]}" >> $DATAFILE
		echo "${LINES[2]}" >> $DATAFILE
	done
done
