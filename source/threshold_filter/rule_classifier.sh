python rule_based_item.py --test_file total.txt --rules "blast_score>30,f5>0.5" --logic AND --output_prefix qua_sim_com_item
python rule_based_item.py --test_file total.txt --rules "blast_score>30" --logic AND --output_prefix qua_sim_item

python rule_based_pair.py --test_file total.txt --rules "blast_score>30" --logic AND --output_prefix qua_sim_pair
python rule_based_pair.py --test_file total.txt --rules "blast_score>30,f5>0.5" --logic AND --output_prefix qua_sim_com_item_pair

python rule_based_item.py --test_file test_all_sample73_feature.txt --rules "blast_score>30,f5>0.5" --logic AND --output_prefix test_all_qua_sim_com_item
python rule_based_item.py --test_file test_all_sample73_feature.txt --rules "blast_score>30" --logic AND --output_prefix test_all_qua_sim_item

python rule_based_pair.py --test_file test_all_sample73_feature.txt --rules "blast_score>30" --logic AND --output_prefix test_all_qua_sim_pair
python rule_based_pair.py --test_file test_all_sample73_feature.txt --rules "blast_score>30,f5>0.5" --logic AND --output_prefix test_all_qua_sim_com_item_pair

