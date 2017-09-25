#Author: Archana Shenoy
#based on Darvin Yi's First Aid package on github
#Training a resnet model to output cross entropy loss

# CXR224
declare -r path_train=/media/dnr/Documents/data/NeRDD/CXR224/training
declare -r path_test=/media/dnr/Documents/data/NeRDD/CXR224/testing
declare -r path_val=/media/dnr/Documents/data/NeRDD/CXR224/validation

# Experiment Name
declare -r name=CXR224-AS-ResNet

# Model path, Log path, figure save path.
declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/$name
mkdir $path_vis
rm $path_log

# Network
declare -r network=Res

# Hyperparameters
declare -r learning_rate=0.001
declare -r learning_rate_decay=0.99 #1.0 is no LR decay
declare -r dropout_keep_prob=0.5
declare -r L2_reg=0.00000001 #lambda for ridge regularization
declare -r L1_reg=0.0 #lambda for lasso regularization
declare -r batch_size=256
declare -r epoch_limit=100

# Load previous model (boolean switch)
declare -r boolean_switch_load=0 #1 means you load

# Train models
python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pTest $path_test --name $name --pModel $path_model --pLog $path_log --pVis $path_vis --net $network --lr $learning_rate --dec $learning_rate_decay --do $dropout_keep_prob --l2 $L2_reg --l1 $L1_reg --bs $batch_size --ep $epoch_limit --bLo $boolean_switch_load --nGPU 4 --bConf 1 --nClass 3 --bDisp 0
