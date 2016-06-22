for i in $(seq 0 6);
do
  python pretrain_model.py --training_mode overfit --nb_epoch 100 --nb_res_layer $i
  python pretrain_model.py --training_mode identity --nb_epoch 100 --nb_res_layer $i
done