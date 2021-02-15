for ((i=14; i<20; i+=1));
do
	 CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --start_index=$i > ../runlogs/log_$i.txt &
	#echo $i*10
done
