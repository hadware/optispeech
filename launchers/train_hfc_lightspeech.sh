python3 -m optispeech.train \
    experiment="hfc_female-en_us-lightspeech" \
    data.train_filelist_path="preprocessed/hfc_female-en_us/train.txt" \
    data.valid_filelist_path="preprocessed/hfc_female-en_us/val.txt" \
    data.batch_size=64 \
    data.num_workers=2 \
    callbacks.model_checkpoint.every_n_epochs=500 \
    paths.log_dir="logs/"