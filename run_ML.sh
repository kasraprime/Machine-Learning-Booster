exp_name=$1 # softmax, baseline, frozen, develop
labels_dim=$2 # 51, 110, 261, 350
features_dim=$3 # 100 , 150 , 300, 550 This is the latent dimension
attention=$4 # attention , no-attention
eval_mode=$5 # train , train-test , test
ablation=$6  # Ours, random (baseline 1), residual
random_seed=$7 # 42, 10
activation=$8 # softplus, relu, tanh

results_dir='results/'$exp_name'-'$ablation'-'$sampling'-'$attention'-'$full_R'/seed-'$random_seed'/'
train_result_dir=$results_dir'/train'
valid_result_dir=$results_dir'/valid'
test_result_dir=$results_dir'/test'
all_results='results/all/'

mkdir -p $results_dir
mkdir -p $train_result_dir
mkdir -p $valid_result_dir
mkdir -p $test_result_dir
mkdir -p $all_results

epoch=100
pred=0.50

track=1 # 0 to turn off wandb and tensorboard tracking
gpu_num=1
lr=0.01

python ML.py --wandb_track $track --experiment_name $exp_name --epochs $epoch\
--random_seed $random_seed --label_dim $labels_dim --feature_dim $features_dim \
--prediction_thresh $pred  --results_dir $results_dir \
--learning_rate $lr --ablation $ablation \
--activation $activation --gpu_num $gpu_num --attention $attention --eval_mode $eval_mode
