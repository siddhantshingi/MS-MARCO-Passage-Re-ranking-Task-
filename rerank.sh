if [[ $1 = "--query" && $3 = "--top1000" && $5 = "--collection" && $7 = "--output" ]];
then
	python3 process_eval_data.py $2 $4 $6
	if [ $? -eq 0 ]; then
		echo OK
	else
		echo FAIL
		exit 0
	fi
	python3 rerank.py $8
else
	echo "input structure is wrong"
fi