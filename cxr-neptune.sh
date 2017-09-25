declare -r path_train=/home/dnr/Documents/data/NeRDD/h5
#declare -r path_test=/home/dnr/Documents/data/NeRDD/testing
declare -r path_val=/home/dnr/Documents/data/NeRDD/validation
#declare -r path_inf=/home/dnr/Documents/data/NeRDD/inference

declare -r name=cxr-class-50k-bs32

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

rm $path_log

python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --name $name --bConf 1 --nGPU 4 --bs 128 --ep 50000 --nClass 3 --lr 0.01 --do 0.3
