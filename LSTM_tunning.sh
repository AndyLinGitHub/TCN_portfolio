model="LSTM"
epoch=1024
save_dir="LSTM_result"

for input_period in 240;
do
  for lr in 0.001 0.0001;
  do
    for optimization_target in sharpe;
    do
      for hidden_size in 16 32 64;
      do
        for num_layers in 1 2 4;
        do
          for dropout in 0 0.1 0.2;
          do
          echo === HP: $model $input_period $lr $hidden_size $num_layers $dropout ===
          python main.py  --setting.save_dir $save_dir \
                          --portfolio_config.model $model \
                          --portfolio_config.input_period $input_period \
                          --hyperparameters_config.epoch $epoch \
                          --hyperparameters_config.lr $lr \
                          --hyperparameters_config.optimization_target $optimization_target \
                          --model_structure_config.$model.hidden_size $hidden_size\
                          --model_structure_config.$model.num_layers $num_layers\
                          --model_structure_config.$model.dropout $dropout
          done
        done
      done
    done
  done
done