declare -r path_train=/media/dnr/Documents/data/NeRDD/h5
#declare -r path_test=/home/dnr/Documents/data/NeRDD/testing
declare -r path_val=/media/dnr/Documents/data/NeRDD/validation
#declare -r path_inf=/home/dnr/Documents/data/NeRDD/inference

declare -r name=cxr-class-all

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/cxr

rm $path_log

python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --pVis $path_vis --name $name --nGPU 4 --bConf 1 --bs 32 --ep 500 --nClass 3 --lr 0.01 --do 0.5
