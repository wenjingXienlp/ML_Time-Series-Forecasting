if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
output_len=96
model_name=Transformer
data_path="/opt/data/private/xwj/ML/data"
#   --root_path /opt/data/private/xwj/ML/data \
#   --data_path train_set.csv \

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$output_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $output_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --freq 't' \
  --batch_size 32 \
  --gpu 0 \
  --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$output_len.log