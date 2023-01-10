DATA_DIR=dataset

MODEL_NAME=RMPI
DATASET_NAME=WN18RR_v1
DATA_PATH=$DATA_DIR/$DATASET_NAME
DB_PATH=$DATA_DIR/WN18RR_v1_RMPI_subgraph
PK_PATH=$DATA_DIR/WN18RR_v1.pkl
TRAIN_SAMPLER_CLASS=RMPISampler
VALID_SAMPLER_CLASS=ValidRMPISampler
TEST_SAMPLER_CLASS=DglSampler_P
LITMODEL_NAME=indGNNLitModel
LOSS=Margin_Loss
MAX_EPOCHS=20
EMB_DIM=32
TRAIN_BS=16
EVAL_BS=16
TEST_BS=1
MARGIN=10.0
LR=1e-3
DROPOUT=0
CHECK_PER_EPOCH=2
EARLY_STOP_PATIENCE=20
NUM_WORKERS=20
CALC_HITS=1,5,10
GPU=1
NUM_GCN_LAYERS=2
HOP=2
ENCLOSING_SUB_GRAPH=False
ABLATION=1
L2=5e-4

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
    --num_gcn_layers $NUM_GCN_LAYERS \
    --hop $HOP \
    --enclosing_sub_graph $ENCLOSING_SUB_GRAPH \
    --ablation $ABLATION \
    --l2 $L2

