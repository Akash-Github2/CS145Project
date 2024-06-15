model_dir=/home/akashp/145/OAG-AQA/outputs/2024-06-14/10-45-49/output_dpr
epoch=25
python dense_retriever.py \
	model_file=$model_dir/dpr_biencoder.$epoch \
	qa_dataset=stackex_qa_test \
	ctx_datatsets=[dpr_stackex_qa] \
	encoded_ctx_files=[/home/akashp/145/OAG-AQA/outputs/2024-06-14/10-45-49/output_dpr/ctx_encoder_temp.pkl_0] \
	out_file=$model_dir/bert_out_$epoch.txt
