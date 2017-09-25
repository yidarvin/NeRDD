# Change CXRsmall to CXR224 or CXR512 for other categories.
declare -r path_train=/media/dnr/Documents/data/NeRDD/CXR224/training
declare -r path_test=/media/dnr/Documents/data/NeRDD/CXR224/testing
declare -r path_val=/media/dnr/Documents/data/NeRDD/CXR224/validation

# Change the name.  This decides what the model is called.  Very Important.
# To not bother other people or write over your previous model in a
# previous experiment, CHANGE THIS.
declare -r name=CXR224-AM-Inception

# Model path, Log path, figure save path.
declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt
declare -r path_vis=/home/dnr/visualizations/$name
mkdir $path_vis
rm $path_log

# Change for network.  Allowed:
# - Alex
# - VGG16
# - GoogLe
# - Inception
# - Res
declare -r network=Inception

# Hyperparameters
declare -r learning_rate=0.001
declare -r learning_rate_decay=0.99 #1.0 is no LR decay
declare -r dropout_keep_prob=0.5
declare -r L2_reg=0.00000001 #lambda for ridge regularization
declare -r L1_reg=0.0 #lambda for lasso regularization
declare -r batch_size=256
declare -r epoch_limit=500

# Load Previous Model
# This is a boolean switch.  If you make this 1, it will load a previously
# trained model (with the same experiment name (e.g. CXRsmall-example).
# If you didn't run an experiment with the same name before, the model will
# not exist, and you can't load anything in, and the program will fail.
declare -r boolean_switch_load=0 #1 means you load

# Change the stuff above.  Do not change the stuff below.  Or do it.  I'm not your supervisor.
python /home/dnr/FirstAid/train_CNNclassification.py --pTrain $path_train --pVal $path_val --pTest $path_test --name $name --pModel $path_model --pLog $path_log --pVis $path_vis --net $network --lr $learning_rate --dec $learning_rate_decay --do $dropout_keep_prob --l2 $L2_reg --l1 $L1_reg --bs $batch_size --ep $epoch_limit --bLo $boolean_switch_load --nGPU 4 --bConf 1 --nClass 3 --bDisp 0
