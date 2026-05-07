# PB-Net
A Novel Deep Learning Approach for Within-Species and Cross-Species Protein Interaction Discovery and Prediction


<img width="577" height="705" alt="未标题-1" src="https://github.com/user-attachments/assets/240f2a81-b5f5-42f8-932b-0403fd1e868a" />

# 1 Install Model to inference
# Install RFdiffusion
https://github.com/RosettaCommons/RFdiffusion
# Install dl_binder_design (ProteinMPNN, AF2)
https://github.com/nrbennet/dl_binder_design
# Install foldseek
https://github.com/steineggerlab/foldseek
# Install ESM,Saprot
https://github.com/evolutionaryscale/esm?tab=readme-ov-file
https://github.com/evolutionaryscale/esm?tab=readme-ov-file

# 2 Binder design and identification framework
# (1) binder design use list
    bach Sac_baker_rfdiff.sh --input example.list --output out_file --rfdiff_path ~/run_inference.py --pdb_path pdb_file
# or single example
    python /home/wuj/data/tools/RFdiffusion/scripts/run_inference.py inference.output_prefix=rfdiffusion_out inference.input_pdb=target_pdb 'contigmap.contigs=[Astart-end}/0 50-70]'  inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0
# (2) binder sequence 
    mkdir -p dl_mpnn_out/Entry; cd dl_mpnn_out/Entry
    CUDA_VISIBLE_DEVICES=1 ~/silent_tools/silentfrompdbs rfdiffusion_out/*/*pdb > dl_mpnn_out/Entry/Entry.silent
    CUDA_VISIBLE_DEVICES=1 ~/dl_binder_design/mpnn_fr/dl_interface_design.py -silent dl_mpnn_out/Entry/Entry.silent -relax_cycles 0 -seqs_per_struct 6 -outsilent dl_mpnn_out/Entry/Entry_dlmpnn.silent
    CUDA_VISIBLE_DEVICES=1 ~/silent_tools/silentextract dl_mpnn_out/Entry/Entry_dlmpnn.silent
    grep 'ANNOTATED_SEQUENCE' dl_mpnn_out/Entry/Entry_dlmpnn.silent|awk '{print $3"\t"$2}'|sed 's/\[/\t/g'|awk '{print ">"$1"\n"$2}' > dl_mpnn_out/Entry/Entry_dlmpnn.fa
    python dataprocess/mpnnseq_rename.py dl_mpnn_out/Entry/Entry_dlmpnn.fa dl_mpnn_out/Entry/Entry_rename_dlmpnn.fa Entry_
# (3) binder quality varify 
    mkdir -p dl_af2_out/Entry;cd dl_af2_out/Entry
    CUDA_VISIBLE_DEVICES=1 ~/dl_binder_design/af2_initial_guess/predict.py -silent dl_mpnn_out/Entry/Entry_dlmpnn.silent -paramdir AF2/Reduced_dbs -model_names model_1_ptm -outsilent dl_af2_out/Entry/Entry_af2.silent -scorefilename dl_af2_out/Entry/Entry_af2.score
    ~/silent_tools/silentextract dl_af2_out/Entry/Entry_af2.silent
    python dl_design_example/pdb_contact_map_12A_list.py dl_af2_out/Entry dl_af2_out/Entry/Entry_af2_contact.txt Entry
    sed '/description/d' dl_af2_out/Entry/Entry_af2.score|awk '{print $11"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9}'|awk 'NR==FNR{a[$1]=$0;next}{if(a[$1]){print a[$1]"\t"$0}}' - dl_af2_out/Entry/Entry_af2_contact.txt|awk '{print "Entry_"$1"\t"$11"\t"$13"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9}' > dl_af2_out/Entry/Entry_af2_quality.txt
# (4) foldseek search and filter
binder quality filter : Total PLDDT > 70 & Interface PAE < 25

    foldseek createdb AFDB_species species_foldseek_DB/species_foldseek_DB
    foldseek easy-cluster example/ res tmp -c 0.9
    foldseek easy-search Entry_1-50_12_dldesign_2_af2pred.pdb ~/species_foldseek_DB/species_foldseek_DB foldseek_out/Entry_foldseek.txt foldseek_out/Entry_tmp --num-iterations 3 --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,prob,lddt,alntmscore,qtmscore,ttmscore,u,t --threads 10
    
foldseek result filter : bits scores > 60
# 2 Item Filter Model
# (1) protein embedding (proteinA, Binder, proteinB)
item_sample.txt : like,  Q9LFV3_120-378_1_dldesign_2_18-58_A0A0A7EPL0_59-98	 null,  output including : proteinA esm embedding, Binder esm embedding, proteinB esm embedding, proteinA saprot embedding, Binder saprot embedding, proteinB saprot embedding

    CUDA_LAUNCH_BLOCKING=2 python dataprocess/PLM_embedding_to_csv.py genome_unipro_no-uniq_protein.fasta Binder_seqs.fasta AFDB_species/ Binder_PDB_chainA/ iterm_sample.txt 
    iterm_foldseek_embedding/ cuda:2
# (2) PPI prediction
    # training
    
inference  class 0 : PPI true, class 1 : PPI false, Possibility=0.5, means, values greater than 0.5 are set as class 0. ​A higher predicted score (ranging from 0 to 1) indicates a greater likelihood that the pair interacts.​

    python PLM_MLP_test.py protA_esm_embedding.csv binder_esm_embedding.csv protB_esm_embedding.csv protA_saprot_embedding.csv binder_saprot_embedding.csv protB_saprot_embedding.csv model.pt out_file
