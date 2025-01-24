echo "Training models"
source ../../../env/bin/activate

# echo "Chinese GPT 3.5 Zero Shot Name and Def"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_culture --output_file culture.csv --category Culture --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_positivity --output_file positivity.csv --category Positivity --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_specialty --output_file specialty.csv --category Specialty --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_name --output_file name.csv --category Personal_Name --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_ambiance --output_file ambiance.csv --category Ambiance --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_creative --output_file creative.csv --category Pun_Creative --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_location --output_file location.csv --category Location --model gpt-3.5-turbo-0125 --prompt_type 3.5/zero_shot/name_def/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"

# echo "Chinese GPT 3.5 Few Shot Lexicon Based"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_culture --output_file culture.csv --category Culture --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_positivity --output_file positivity.csv --category Positivity --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_specialty --output_file specialty.csv --category Specialty --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_name --output_file name.csv --category Personal_Name --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_ambiance --output_file ambiance.csv --category Ambiance --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_creative --output_file creative.csv --category Pun_Creative --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_location --output_file location.csv --category Location --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/lexicon_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"

echo "Chinese GPT 3.5 def + examples"
### check PROMPT BEFORE YOU RUN THE CODE!!!!!!!!!!!
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_culture --output_file culture.csv --category Culture --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_positivity --output_file positivity.csv --category Positivity --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_specialty --output_file specialty.csv --category Specialty --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_name --output_file name.csv --category Personal_Name --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_ambiance --output_file ambiance.csv --category Ambiance --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_creative --output_file creative.csv --category Pun_Creative --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_location --output_file location.csv --category Location --model gpt-3.5-turbo-0125 --prompt_type 3.5/few_shot/rule_based/ --input_file ../data_cleaning/output/validation_cn.csv --output_folder ./outputs/validation/Chinese --language cn"

# echo "Chinese GPT 4 Zero Shot Name and Def"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_culture --output_file culture.csv --category Culture --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_positivity --output_file positivity.csv --category Positivity --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_specialty --output_file specialty.csv --category Specialty --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_name --output_file name.csv --category Personal_Name --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_ambiance --output_file ambiance.csv --category Ambiance --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_creative --output_file creative.csv --category Pun_Creative --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_location --output_file location.csv --category Location --prompt_type 4/zero_shot/name_def/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"

# echo "Chinese GPT 4 Lexicon Based"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_culture --output_file culture.csv --category Culture --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_positivity --output_file positivity.csv --category Positivity --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_specialty --output_file specialty.csv --category Specialty --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_name --output_file name.csv --category Personal_Name --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_ambiance --output_file 4_lexicon_ambiance.csv --category Ambiance --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese/best --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_creative --output_file creative.csv --category Pun_Creative --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/test_cn.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_location --output_file 4_lexicon_location.csv --category Location --prompt_type 4/few_shot/lexicon_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese/best --language cn"

# echo "Chinese GPT 4 Definiton and Example"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_culture --output_file culture.csv --category Culture --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_positivity --output_file positivity.csv --category Positivity --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_specialty --output_file 4_full_specialty.csv --category Specialty --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese/best --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_name --output_file 4_full_name.csv --category Personal_Name --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese/best --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_ambiance --output_file ambiance.csv --category Ambiance --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_creative --output_file 4_full_creative.csv --category Pun_Creative --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese/best --language cn"
# screen -dm bash -c "python3 -m classifier --prompt_file_path prompt_cn_location --output_file location.csv --category Location --prompt_type 4/few_shot/rule_based/ --input_file ./training_validation/testing.csv --output_folder ./outputs/test/Chinese --language cn"

# echo "Chinese llama Zero Shot Name and Definition"
# screen -dm bash -c "python3 -m classifier_llama --prompt_file_path prompt_en_culture --output_file culture.csv --category Culture --prompt_type llama/zero_shot/name_def/ --output_folder ./outputs/validation/Chinese --input_file ../data_cleaning/output/validation_cn.csv"