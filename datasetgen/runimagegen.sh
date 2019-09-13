#!/bin/bash

#SBATCH -N 1
#SBATCH -p juliet
#SBATCH -t 120:30:00
#SBATCH -J datagen
#SBATCH -o datagen.o%j

#srun -p juliet -N 1 -n 1 -c 1

for i in {1..5}
do
	python clusterdatagen.py ./inputnetworks/network_2.mtx $i
	python clusterdatagen.py ./inputnetworks/network_3.mtx $i
	python clusterdatagen.py ./inputnetworks/network_4.mtx $i
	python clusterdatagen.py ./inputnetworks/network_5.mtx $i
	python clusterdatagen.py ./inputnetworks/network_6.mtx $i
	python clusterdatagen.py ./inputnetworks/network_7.mtx $i
	python clusterdatagen.py ./inputnetworks/network_8.mtx $i
	python clusterdatagen.py ./inputnetworks/network_9.mtx $i
	python clusterdatagen.py ./inputnetworks/network_10.mtx $i
	python clusterdatagen.py ./inputnetworks/network_11.mtx $i
	python clusterdatagen.py ./inputnetworks/network_12.mtx $i
	python clusterdatagen.py ./inputnetworks/network_13.mtx $i
	python clusterdatagen.py ./inputnetworks/network_14.mtx $i
	python clusterdatagen.py ./inputnetworks/network_15.mtx $i
	python clusterdatagen.py ./inputnetworks/network_16.mtx $i
	python clusterdatagen.py ./inputnetworks/network_17.mtx $i
	python clusterdatagen.py ./inputnetworks/network_18.mtx $i
	python clusterdatagen.py ./inputnetworks/network_19.mtx $i
	python clusterdatagen.py ./inputnetworks/network_20.mtx $i
	python clusterdatagen.py ./inputnetworks/network_21.mtx $i
	python clusterdatagen.py ./inputnetworks/network_22.mtx $i
	python clusterdatagen.py ./inputnetworks/network_23.mtx $i
	python clusterdatagen.py ./inputnetworks/network_24.mtx $i
	python clusterdatagen.py ./inputnetworks/network_25.mtx $i
	python clusterdatagen.py ./inputnetworks/network_26.mtx $i
	python clusterdatagen.py ./inputnetworks/network_27.mtx $i
	python clusterdatagen.py ./inputnetworks/network_28.mtx $i
	python clusterdatagen.py ./inputnetworks/network_29.mtx $i
	python clusterdatagen.py ./inputnetworks/network_30.mtx $i
done
