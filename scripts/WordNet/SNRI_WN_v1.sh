DATA_DIR=dataset

MODEL_NAME=SNRI
DATASET_NAME=WN18RR_v1
DATA_PATH=$DATA_DIR/$DATASET_NAME
DB_PATH=$DATA_DIR/WN18RR_v1_subgraph
PK_PATH=$DATA_DIR/WN18RR_v1.pkl
TRAIN_SAMPLER_CLASS=SNRISampler
VALID_SAMPLER_CLASS=ValidSNRISampler
TEST_SAMPLER_CLASS=TestSNRISampler
LITMODEL_NAME=indGNNLitModel
LOSS=Margin_Loss
MAX_EPOCHS=50
EMB_DIM=32
TRAIN_BS=16
EVAL_BS=16
TEST_BS=1
MARGIN=10.0
LR=1e-4
CHECK_PER_EPOCH=3
EARLY_STOP_PATIENCE=20
NUM_WORKERS=20
DROPOUT=0
CALC_HITS=1,5,10
GPU=1
CHECKPOINT_DIR=/home/lli/NeuralKG-ind-NeuralKG-ind/output/link_prediction/WN18RR_v1/SNRI/epoch\=29-Eval\|aoc\=0.974.ckpt
TEST_METRIC=hit
TEST_DB_PATH=$DATA_DIR/WN18RR_v1_ind/test_subgraphs

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --db_path $DB_PATH \
    --pk_path $PK_PATH \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --valid_sampler_class $VALID_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --litmodel_name $LITMODEL_NAME \
    --loss $LOSS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --test_bs $TEST_BS \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --num_workers $NUM_WORKERS \
    --dropout $DROPOUT \
    --calc_hits $CALC_HITS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --test_metric $TEST_METRIC \
    --test_db_path $TEST_DB_PATH \
    --test_only