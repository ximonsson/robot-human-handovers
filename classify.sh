
# RUN SETTINGS

. ./env # load environment

PKG=classification
TRAINING_DATA=data/classification/images-train
TEST_DATA=data/classification/images-test

function classify
{
	LR=$1
	BS=$2
	OBJECTS="$3"
	TEST_OBJECTS="$4"
	LOG_SUFFIX=$5
	K=$6

	echo "python -m $PKG \
		--learning-rate=$LR \
		--batch-size=$BS \
		--objects="$OBJECTS" \
		--test-objects="$TEST_OBJECTS" \
		--data="$TRAINING_DATA" \
		--test-data="$TEST_DATA" \
		--logdir-suffix="$LOG_SUFFIX" \
		--k=$K"
}


#declare -A objects = (
	#["ball"]="None",
	#["bottle"]="21",
	#["box"]="16",
	#["brush"]="22",
	#["can"]="15",
	#["cutters"]="17",
	#["glass"]="20",
	#["hammer"]="2",
	#["knife"]="12",
	#["cup"]="23",
	#["pen"]="3",
	#["pitcher"]="19",
	#["scalpel"]="4",
	#["scissors"]="5",
	#["screwdriver"]="14",
	#["tube"]="18",
	#)

#declare -A test_objects = (
#
#)

objects=(
	#"ball",
	"bottle"
	"box"
	"cutters"
	"glass"
	"knife"
	"scissors"
	"brush"
	"scalpel"
	"can"
	"screwdriver"
	"pitcher"
	"hammer"
	"pen"
	"cup"
	#"tube",
	)

test_objects=(
	"new-bottle"
	"new-can"
	"new-cheeseknife"
	"new-cup"
	"new-fork"
	"new-glass"
	"new-jar"
	"new-knive"
	"new-pliers"
	"new-scissors"
	"new-screwdriver"
	"new-spoon"
	"new-wineglass"
	)

#
# TESTS
#


# LEARNING RATES
LRs=("0.1" "0.01" "0.001" "0.0001" "0.00001")
BS=128
for LR in "${LRs[@]}"
do
	classify $LR $BS "${objects[@]}" "${test_objects[@]}" "lr_$LR""__bs_$BS" 5
done


# BATCH SIZES



# OBJECTS


