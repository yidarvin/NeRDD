#Author: Archana Shenoy
#Based on Darvin Yi's First Aid package instructions
#This program runs a previously saved model on a new dataset and outputs probabilities of all classes


# file path for validation dataset
declare -r path_inf=/media/dnr/Documents/data/NeRDD/CXRsmall/testing

#  model
declare -r name=CXRsmall-example-CT

# Model path, Log path, figure save path.
declare -r path_model=/home/dnr/modelState/$name.ckpt
#declare -r path_log=/home/dnr/logs/$name.txt
#declare -r path_vis=/home/dnr/visualizations/$name

#mkdir $path_vis
#rm $path_log


# Hyperparameters
declare -r learning_rate=0.001
declare -r learning_rate_decay=0.99 #1.0 is no LR decay
declare -r dropout_keep_prob=0.5
declare -r L2_reg=0.00000001 #lambda for ridge regularization
declare -r L1_reg=0.0 #lambda for lasso regularization
declare -r batch_size=256
declare -r epoch_limit=100

# Load Previous Model (Boolean Switch)
declare -r boolean_switch_load=1

# Change the stuff above.  Do not change the stuff below.  Or do it.  I'm not your supervisor.
python /home/dnr/FirstAid/train_CNNclassification.py --pInf $path_inf --name $name --pModel $path_model --lr $learning_rate --dec $learning_rate_decay --do $dropout_keep_prob --l2 $L2_reg --l1 $L1_reg --bs $batch_size --ep $epoch_limit --bLo $boolean_switch_load --nGPU 4 --bConf 1 --nClass 3 --bDisp 0 
