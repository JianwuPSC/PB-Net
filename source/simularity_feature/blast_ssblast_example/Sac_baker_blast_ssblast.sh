
###### get binder dssp from pdb, Proteome pdb to dssp

python ~/dataprocess/pdb2dssp.py Entry.pdb Entry.dssp Entry

# get sequence blast out and second struc blast out
# example.list, from foldseek output

#binder	target	binder_start	binder_end target_start	target_end
#P00546_8-295_0_dldesign_0	O13529	7	32	3	28
#P00546_8-295_0_dldesign_0	O13549	1	32	66	98

##### sequence similarity and second structure similarity
#binder fasta, binder dssp, tagret fasta, target dssp in the feature_blast_ssblast.py
python ~/dataprocess/feature_blast_ssblast.py example.list binder.fa binder.dssp target.fa target.dssp blast.out ssblast.out
