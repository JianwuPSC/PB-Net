
# Entry=Unipro_ID

target_pdb=AFDB_yeast/AF-${Entry}-F1-model_v4.pdb
rfdiffusion_out=rfdiffusion/sample/${Entry}
dl_mpnn_out=dl_binder_design/proteinmpnn
dl_af2_out=dl_binder_design/af2


#conda activate proteinmpnn_binder_design
mkdir -p $dl_mpnn_out/${Entry}
cd $dl_mpnn_out/${Entry}

CUDA_VISIBLE_DEVICES=1 ~/silent_tools/silentfrompdbs ${rfdiffusion_out}/*/*pdb > $dl_mpnn_out/${Entry}/${Entry}.silent 

CUDA_VISIBLE_DEVICES=1 ~/dl_binder_design/mpnn_fr/dl_interface_design.py -silent $dl_mpnn_out/${Entry}/${Entry}.silent -relax_cycles 0 -seqs_per_struct 6 -outsilent $dl_mpnn_out/${Entry}/${Entry}_dlmpnn.silent

CUDA_VISIBLE_DEVICES=1 ~/silent_tools/silentextract $dl_mpnn_out/${Entry}/${Entry}_dlmpnn.silent
grep 'ANNOTATED_SEQUENCE' $dl_mpnn_out/${Entry}/${Entry}_dlmpnn.silent|awk '{print $3"\t"$2}'|sed 's/\[/\t/g'|awk '{print ">"$1"\n"$2}' > $dl_mpnn_out/${Entry}/${Entry}_dlmpnn.fa

python /home/wuj/data/project/ppi_predict/dataprocess/mpnnseq_rename.py $dl_mpnn_out/${Entry}/${Entry}_dlmpnn.fa $dl_mpnn_out/${Entry}/${Entry}_rename_dlmpnn.fa ${Entry}_

#conda activate af2_binder_design
mkdir -p $dl_af2_out/${Entry}
cd $dl_af2_out/${Entry}

CUDA_VISIBLE_DEVICES=1 /home/wuj/data/tools/dl_binder_design/af2_initial_guess/predict.py -silent $dl_mpnn_out/${Entry}/${Entry}_dlmpnn.silent -paramdir /home/wuj/data/tools/AF2/AF2/Reduced_dbs -model_names model_1_ptm -outsilent $dl_af2_out/${Entry}/${Entry}_af2.silent -scorefilename $dl_af2_out/${Entry}/${Entry}_af2.score
/home/wuj/data/tools/silent_tools/silentextract $dl_af2_out/${Entry}/${Entry}_af2.silent

python dl_design_example/pdb_contact_map_12A_list.py dl_af2_out/Entry dl_af2_out/Entry/Entry_af2_contact.txt Entry

##### proteinA/Binder quality output
sed '/description/d' $dl_af2_out/${Entry}/${Entry}_af2.score|awk '{print $11"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9}'|awk 'NR==FNR{a[$1]=$0;next}{if(a[$1]){print a[$1]"\t"$0}}' - $dl_af2_out/${Entry}/${Entry}_af2_contact.txt|awk '{print "'${Entry}_'"$1"\t"$11"\t"$13"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9}' > $dl_af2_out/${Entry}/${Entry}_af2_quality.txt
