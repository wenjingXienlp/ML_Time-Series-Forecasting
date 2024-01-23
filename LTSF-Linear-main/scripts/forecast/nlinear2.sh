export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
output_len=336
model_name=Transformer
data_path="/opt/data/private/xwj/ML/data"
checkpoints="/opt/data/private/xwj/ML/LTSF-Linear-main/NLinear_cpkts"
#   --root_path /opt/data/private/xwj/ML/data \
#   --data_path train_set.csv \

python -u run_longExp.py \
  --is_training 1 \
  --checkpoints $checkpoints \
  --root_path /opt/data/private/wtc_beifen/time_prediction/Autoformer/dataset/ETT-small \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$output_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $output_len \
  --enc_in 7 \
  --e_layers 7 \
  --des 'Exp' \
  --itr 5 \
  --batch_size 64 \
  --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$output_len.log
