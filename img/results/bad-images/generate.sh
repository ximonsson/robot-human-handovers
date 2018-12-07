filename="img/results/bad-images.dat"
while read -r line
do
	line=`echo $line | xargs`
	(cd img/results/bad-images && ln -s -T all/$line $line)
done < "$filename"
