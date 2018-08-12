RGB=rgb
DEPTH=depth
REGISTERED=registered
DATA=data/recordings
OUT=$DATA/$1

# delete any old recordings in this name
if [ -e $OUT ]
then
	rm -r $OUT
fi

# create output directories
mkdir -p $OUT/$RGB
mkdir -p $OUT/$DEPTH
mkdir -p $OUT/$REGISTERED

# start recording
BIN=bin/rec
$BIN $OUT
