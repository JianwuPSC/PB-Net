
##### Rosettafold a3m

cd-hit -i euk_species.fasta -o euk_species_id90.fasta -c 0.9 -T 0.4 -n 5 -m 100000
mmseqs createdb euk_species_id90.fasta ~/euk_species_id90
mmseqs createindex ~/euk_species_id90 ~/euk_species_id90_index_prefix

mmseqs createdb Entry.fa Sac_eku_msa/Entry

mmseqs search Sac_eku_msa/Entry ~/euk_species_id90 Sac_eku_msa/Entry_result Sac_eku_msa/Entry_tmp

mmseqs result2msa Sac_eku_msa/Entry ~/euk_species_id90 Sac_eku_msa/Entry_result Sac_eku_msa/Entry_result_msa

mmseqs unpackdb Sac_eku_msa/Entry_result_msa Sac_eku_msa/Entry_pack

mv Sac_eku_msa/Entry_pack/0 Sac_eku_msa/Entry.a3m

#####  colabfold a3m

colabfold_batch Entry_colab.fa out_dir --msa-only

colabfold_batch --num-recycle 2 --model-type alphafold2_multimer_v3 Entry_colab.fa colabfold_txt/Entry --num-models 1

#####  a3m merge
# length: proteinA length
python ~/dataprocess/a3m_merge.py proteinA.a3m proteinB.a3m msa_merge_out/protA_protB.a3m

#####  rosetta 2track score

python ~/RoseTTAFold/network_2track/predict_msa.py -m ~/RoseTTAFold/weights -msa msa_merge_out/protA_protB.a3m -npz msa_merge_out/protA_protB.npz -L1 length

python ~/dataprocess/rosetta_2track_hot_test.py msa_merge_out/protA_protB.npz msa_merge_out/protA_protB_2track.txt protA_protB

#####  DCA
# Entry Q12446_Q9ZZW2	633 (Q12446 length)
python ~/dataprocess/dca.py Entry.list msa_merge_out/ pydca/Entry
