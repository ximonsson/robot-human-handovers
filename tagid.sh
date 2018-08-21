DATA=data
OBJECTS=$DATA/objects
BIN=bin/extract_tagid

for IM in $OBJECTS/rgb/*.jpg
do
	# extract data
	# if we didn't find any tag we continue to the next frame
	OUT="$($BIN --image $IM --flip 2>&1)"
	if [ $? -ne 0 ]
	then
		continue
	fi

	# parse ID of the tag
	IFS=':' read -ra TAG <<< "$OUT"
	ID="${TAG[0]}"

	echo "$IM => $ID"
done
