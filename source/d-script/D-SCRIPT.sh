python embed.py --seqs=Sac_unipro_no-uniq_protein.fasta -o Sac_unipro.h5

python train_yeast.py \
  --train-csv yeast_train/train.txt \
  --valid-csv yeast_train/valid.txt \
  --test-csv yeast_train/test_all.txt \
  --embedding Sac_unipro.h5 \
  --save-prefix yeast_train/yeast_train.pt \
  --num-epochs 100 --batch-size 32 --lr 0.001 \
  --pool-width 9 \
  --kernel-width 7 \
  --do-pool \
  --max-seq-len 2000 \
  --device 0 \
  --no-w

python test_yeast.py \
  --model yeast_train/yeast_train.pt_epoch045.sav \
  --test yeast_train/test_all.txt \
  --embedding Sac_unipro.h5 \
  --max-seq-len 2000 \
  --manual-threshold 0.5 \
  --device 0 \
  -o yeast_dscript

python eval.py --model=dscript-data/models/dscript_human_v2.pt --test=yeast_train/test_all.txt --embedding=Sac_unipro.h5 -o=human_dscript
