if [ $1  ]; then
  MODEL=$1
else
  MODEL='fast_st_ps'
fi

for i in $(seq 0 6);
do
  python pretrain_model.py --training_mode overfit --model $MODEL --nb_epoch 100 --nb_res_layer $i
  python pretrain_model.py --training_mode identity --model $MODEL --nb_epoch 100 --nb_res_layer $i
done