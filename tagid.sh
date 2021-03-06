DIR=$1
BIN=bin/extract_tagid
DST=objects.db

if [ -f $DST ]
then
	rm $DST
fi

echo "Looping over RGB images in $DIR/rgb ... "
for IM in $DIR/*.jpg
do
	# extract data
	# if we didn't find any tag we continue to the next frame
	OUT="$(LD_LIBRARY_PATH=/usr/local/lib $BIN --image $IM --flip 2>&1)"
	if [ $? -ne 0 ]
	then
		echo "No tag detected in $IM"
		continue
	fi

	# parse ID of the tag
	IFS=':' read -ra TAG <<< "$OUT"
	ID="${TAG[0]}"

	echo "$IM => $ID"
	echo "$IM:$OUT" >> $DST
done
