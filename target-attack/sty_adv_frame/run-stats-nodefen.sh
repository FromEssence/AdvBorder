batch_size=1000
for ((i=0; i<5; i+=1));
do
	CUDA_VISIBLE_DEVICES=$(($i%3)) nohup python resnet50_attack_stats.py --defense=0 --start_index=$i --batch_size=$batch_size > ../results/attack-nodefen-stats_$i.txt &
	# echo $(($i*10))
	sleep 10s
done
