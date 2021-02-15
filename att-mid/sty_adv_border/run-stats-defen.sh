batch_size=1000
for ((i=0; i<4; i+=1));
do
	CUDA_VISIBLE_DEVICES=$(($i%3)) nohup python resnet50_attack_stats.py --defense=1 --start_index=$i --batch_size=$batch_size > ../results/defen50/attack-defen-50-stats_$i.txt &
	#echo $(($i*10))
	sleep 5s
done
