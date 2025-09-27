#!/bin/bash

verbose=false
input_file=""
output_file="output.txt"
rfdiff_path=""
pdb_path=""

TEMP=$(getopt -o f:o:vh --long file:,output:,verbose,help -n "$0" -- "$@")

if [ $? != 0 ]; then
    echo "error" >&2
    exit 1
fi

eval set -- "$TEMP"

while true; do
    case "$1" in
        -i|--input) input_file="$2"; shift 2 ;;
        -o|--output) output_file="$2"; shift 2 ;;
        -pdb|--pdb_path) pdb_path="$2"; shift 2 ;;
        -rf|--rfdiff_path) rfdiff_path="$2"; shift 2 ;;
        -v|--verbose) verbose=true; shift ;;
        -h|--help) 
            echo "need: $0 [--verbose] [--input input file] [--output output file] [--rfdiff_path rfdiffusion inference.py] [--pdb_path pdb file]"
            exit 0
            ;;
        --) shift; break ;;
        *) echo ""; exit 1 ;;
    esac
done

############### use list

# Entry,length,Coiled_coil_count,Compositional_bias_count,Domain_count,Region_count,Repeat,other
# A0A0B7P3V8,1104,48-112,None,619-786,381-501 539-599,None,None

SRR_list=$input_file
last_index=`wc -l ${SRR_list}|awk '{print $1}' -`

#conda activate SE3nv

for id in `seq 1 $last_index`
do

export Entry=`awk 'BEGIN{FS=","}NR=="'$id'"{print $1}' $SRR_list`
export length=`awk 'BEGIN{FS=","}NR=="'$id'"{print $2}' $SRR_list`
export Coiled_coil=`awk 'BEGIN{FS=","}NR=="'$id'"{print $3}' $SRR_list`
export Compositional_bias=`awk 'BEGIN{FS=","}NR=="'$id'"{print $4}' $SRR_list`
export Domain=`awk 'BEGIN{FS=","}NR=="'$id'"{print $5}' $SRR_list`
export Region=`awk 'BEGIN{FS=","}NR=="'$id'"{print $6}' $SRR_list`
export Repeat=`awk 'BEGIN{FS=","}NR=="'$id'"{print $7}' $SRR_list`
export other=`awk 'BEGIN{FS=","}NR=="'$id'"{print $8}' $SRR_list`

target_pdb=$pdb_path

if [ -f $target_pdb ];then

rfdiffusion_out=$output_file/${Entry}

Coiled_coil_out=${rfdiffusion_out}/Coiled_coil
Compositional_bias_out=${rfdiffusion_out}/Compositional_bias
Domain_out=${rfdiffusion_out}/Domain
Region_out=${rfdiffusion_out}/Region
Repeat_out=${rfdiffusion_out}/Repeat
other_out=${rfdiffusion_out}/other

range_sub=10

############ Coiled_coil_out

if [[ $Coiled_coil != 'None' ]];then

#mkdir -p $Coiled_coil_out
Coiled_coil_list=(${Coiled_coil// / })
for list1 in ${Coiled_coil_list[@]}

do 
    start_=${list1%-*}
    end_=${list1#*-}
    range_start=`expr ${list1#*-} - ${list1%-*} - $range_sub`
    range_end=`expr ${list1#*-} - ${list1%-*} + $range_sub`
    Coiled_coil_final_out=$Coiled_coil_out/$list1
    
    python ${rfdiff_path} inference.output_prefix=$Coiled_coil_final_out inference.input_pdb=$target_pdb 'contigmap.contigs=[A'${start_}'-'${end_}'/0 50-70]'  inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

done

fi

############# Compositional_bias

if [[ $Compositional_bias != 'None' ]];then

#mkdir -p $Compositional_bias_out
Compositional_bias_list=(${Compositional_bias// / })
for list1 in ${Compositional_bias_list[@]}

do
    start_=${list1%-*}
    end_=${list1#*-}
    range_start=`expr ${list1#*-} - ${list1%-*} - $range_sub`
    range_end=`expr ${list1#*-} - ${list1%-*} + $range_sub`
    Compositional_bias_final_out=$Compositional_bias_out/$list1

    python ${rfdiff_path} inference.output_prefix=$Compositional_bias_final_out inference.input_pdb=$target_pdb 'contigmap.contigs=[A'${start_}'-'${end_}'/0 50-70]'  inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

done
#wait
fi

############# domain

if [[ $Domain != 'None' ]];then

#mkdir -p $Domain_out
Domain_list=(${Domain// / })
for list1 in ${Domain_list[@]}
do
    start_=${list1%-*}
    end_=${list1#*-}
    range_start=`expr ${list1#*-} - ${list1%-*} - $range_sub`
    range_end=`expr ${list1#*-} - ${list1%-*} + $range_sub`
    Domain_final_out=$Domain_out/$list1

    python ${rfdiff_path} inference.output_prefix=$Domain_final_out inference.input_pdb=$target_pdb 'contigmap.contigs=[A'${start_}'-'${end_}'/0 '50'-'70']'  inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

done
#wait
fi

############# range

if [[ $Region != 'None' ]];then

Region_list=(${Region// / })
for list1 in ${Region_list[@]}

do
    start_=${list1%-*}
    end_=${list1#*-}
    range_start=`expr ${list1#*-} - ${list1%-*} - $range_sub`
    range_end=`expr ${list1#*-} - ${list1%-*} + $range_sub`
    Region_final_out=$Region_out/$list1

    python ${rfdiff_path} inference.output_prefix=$Region_final_out inference.input_pdb=$target_pdb 'contigmap.contigs=[A'${start_}'-'${end_}'/0 '50'-'70']'  inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

done
#wait
fi

############## repeat

if [[ $Repeat != 'None' ]];then

#mkdir -p $Repeat_out
Repeat_list=(${Repeat// / })
for list1 in ${Repeat_list[@]}
do
    start_=${list1%-*}
    end_=${list1#*-}
    range_start=`expr ${list1#*-} - ${list1%-*} - $range_sub`
    range_end=`expr ${list1#*-} - ${list1%-*} + $range_sub`
    Repeat_final_out=$Repeat_out/$list1

    python ${rfdiff_path} inference.output_prefix=$Repeat_final_out inference.input_pdb=$target_pdb 'contigmap.contigs=[A'${start_}'-'${end_}'/0 '50'-'70']'  inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

done
#wait
fi

############# other

if [[ $other != 'None' ]];then
if [ $length -gt 30 ];then

#mkdir -p $other_out
other_list=(${other// / })
for list1 in ${other_list[@]}
do
    start_=${list1%-*}
    end_=${list1#*-}
    range_start=`expr ${list1#*-} - ${list1%-*} - $range_sub`
    range_end=`expr ${list1#*-} - ${list1%-*} + $range_sub`
    other_final_out=$other_out/$list1

    python ${rfdiff_path} inference.output_prefix=$other_final_out inference.input_pdb=$target_pdb 'contigmap.contigs=[A1-'${length}'/0 '50'-'70']'  inference.num_designs=20 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

done
fi
fi

fi
done

