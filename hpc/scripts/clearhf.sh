find ~/.cache/huggingface/datasets/ -mindepth 1 -not -path "*evalset*" -exec rm -rf {} \;
