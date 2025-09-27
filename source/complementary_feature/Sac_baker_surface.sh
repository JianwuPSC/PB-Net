###################################### Surface

### masif surface finger feature

pdb2pqr30 --ff=AMBER --with-ph=7.0 Entry.pdb Entry.pqr

x_neg=`sort -k 6 -n Entry.pqr|sed -n '1p'|awk '{print $6}'|awk '{if ($1<0) print $1*(-1);else print $1}'`
y_neg=`sort -k 7 -n Entry.pqr|sed -n '1p'|awk '{print $7}'|awk '{if ($1<0) print $1*(-1);else print $1}'`
z_neg=`sort -k 8 -n Entry.pqr|sed -n '1p'|awk '{print $8}'|awk '{if ($1<0) print $1*(-1);else print $1}'`
x_posi=`sort -k 6 -n -r Entry.pqr|sed -n '1p'|awk '{print $6}'|awk '{if ($1<0) print $1*(-1);else print $1}'`
y_posi=`sort -k 7 -n -r Entry.pqr|sed -n '1p'|awk '{print $7}'|awk '{if ($1<0) print $1*(-1);else print $1}'`
z_posi=`sort -k 8 -n -r Entry.pqr|sed -n '1p'|awk '{print $8}'|awk '{if ($1<0) print $1*(-1);else print $1}'`
fn=`tail Entry.pqr|sed -n '1p'|awk '{print $5/20+10}'`
cg=`tail Entry.pqr|sed -n '1p'|awk '{print $5/20+20}'`
aa_x=`echo "scale=2; ${x_neg} + ${x_posi} + ${cg}"|bc`
aa_y=`echo "scale=2; ${y_neg} + ${y_posi} + ${cg}"|bc`
aa_z=`echo "scale=2; ${z_neg} + ${z_posi} + ${cg}"|bc`
bb_x=`echo "scale=2; ${x_neg} + ${x_posi} + ${fn}"|bc`
bb_y=`echo "scale=2; ${y_neg} + ${y_posi} + ${fn}"|bc`
bb_z=`echo "scale=2; ${z_neg} + ${z_posi} + ${fn}"|bc`

sed 's/protein/Entry/g' apbs.in|sed 's/aa_x/'${aa_x}'/'|sed 's/aa_y/'${aa_y}'/'|sed 's/aa_z/'${aa_z}'/'|sed 's/bb_x/'${bb_x}'/'|sed 's/bb_y/'${bb_y}'/'|sed 's/bb_z/'${bb_z}'/' > apbs1.in

apbs apbs1.in

protonated_file=Entry_prot.pdb
protonated_file_chain_pdb=Entry_chain.pdb
protonated_file_name=Entry

##### output Entry.ply

python /home/wuj/data/project/ppi_predict/dataprocess/surface_finger.py $source_pdb $protonated_file $protonated_file_chain_pdb $protonated_file_name

##### proteinA/binder_finger

# Entry_prot.pdb Entry_chain.pdb Entry.ply in AFBS
# proteinA/binder pdb in af2_pdb/Entry

python ~/dataprocess/surface_finger_contact_feature.py af2_pdb/Entry Entry.pdb ~/APBS/ ~/protA_binder_finger/

##### binder/proteinB_finger

mkdssp -i Entry.pdb -o mkdssp.txt
awk -F ',' '{print $1}' mkdssp.txt |awk '{print $(NF-1)}'|grep -E '^[0-9]+(\.[0-9]+)?$' > Entry.acc

#Entry_target.list # Q06411	156	206  (binder target range)

python ~/dataprocess/surface_finger_acc_feature.py Entry_target.list Entry.acc Entry.pdb ~/APBS/ ~/binder_protB_finger/

#####
# item.txt P00546_8-295_11_dldesign_0     Q06411  1       31      156     206
python ~/dataprocess/surface_source_target_compare_score.py item.txt surface_out.txt ~/protA_binder_finger/ ~/binder_protB_finger/

