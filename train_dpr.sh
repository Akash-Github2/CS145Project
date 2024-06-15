# python train_dense_encoder.py \
#     train_datasets=[stackex_qa_train] \
#     dev_datasets=[stackex_qa_valid]
#     train=biencoder_local \
#     output_dir=output/dpr/

python train_roberta_encoder.py \
    train_datasets=[stackex_qa_train] \
    dev_datasets=[stackex_qa_valid]
    train=biencoder_local \
    output_dir=output/dpr/