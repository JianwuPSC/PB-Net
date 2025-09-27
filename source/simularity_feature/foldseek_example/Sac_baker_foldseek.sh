##### creat db
~/data/tools/foldseek/bin/foldseek createdb ./AFDB_Human Human_foldseek_DB/Human_foldseek_DB

~/data/tools/foldseek/bin/foldseek easy-cluster Entry/ res tmp -c 0.9 

##### structure similarity Entry_foldseek.txt
~/data/tools/foldseek/bin/foldseek easy-search Entry.pdb Human_foldseek_DB/Human_foldseek_DB Entry_foldseek.txt Entry_tmp --num-iterations 3 --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,prob,lddt,alntmscore,qtmscore,ttmscore,u,t --threads 10

