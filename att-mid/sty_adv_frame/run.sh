for ((i=0; i<4; i+=1));
do
	CUDA_VISIBLE_DEVICES=$(($i%3)) nohup python -u main.py --start_index=$i > ../runlogs/log_$i.txt &
	#echo $i*10
done
