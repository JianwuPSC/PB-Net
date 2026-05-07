
python topsy_turvy/embedding.py --seqs=Sac_unipro_no-uniq_protein.fasta -o Sac_unipro.h5 -d 1

python train_yeast.py \
  --train-tsv yeast_train/train.txt \
  --valid-tsv yeast_train/valid.txt \
  --test-tsv yeast_train/test_all.txt \
  --embedding Sac_unipro.h5 \
  --save-prefix yeast_train/yeast_topsy.pt \
  --num-epochs 100 --batch-size 32 --lr 0.001 \
  --max-seq-len 2000 \
  --device 0 \
  --use_glider

python test_yeast.py \
  --model yeast_train/yeast_topsy.pt_epoch029.sav \
  --test-tsv yeast_train/test_all.txt \
  --embedding Sac_unipro.h5 \
  --max-seq-len 2000 \
  --manual-threshold 0.5 \
  --outprefix yeast_topsy \
  --plot-curves \
  --device 0

python evaluate_topsy_turvy.py \
  --model Pretrained-Models/topsy_turvy_best_model.sav \
  --test-tsv yeast_train/test_all.txt \
  --embeddings Sac_unipro.h5 \
  --max-seq-len 2000 \
  --outfile human_topsy \
  --plot-curves \
  --device -1
