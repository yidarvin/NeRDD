declare -r path_train=/home/dnr/Documents/data/NeRDD/h5
#declare -r path_test=/home/dnr/Documents/data/NeRDD/testing
declare -r path_val=/home/dnr/Documents/data/NeRDD/validation
#declare -r path_inf=/home/dnr/Documents/data/NeRDD/inference

declare -r name=cxr-class-10000

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/cxr

rm $path_log

python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pModel $path_model --pLog $path_log --pVis $path_vis --name $name --nGPU 4 --bs 128 --ep 500 --nClass 3 --lr 0.001 --do 0.5
