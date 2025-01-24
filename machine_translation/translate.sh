echo "Translating names"
source ../../../env/bin/activate

echo "running chatGPT"
# screen -dm bash -c "python3 -m translator_gpt --model gpt-3.5-turbo-0125 --output_file gpt35.csv --col_name gpt35_translation"

### GPT4 #####
# screen -dm bash -c "python3 -m translator_gpt --output_file gpt4_1.csv --col_name gpt4_1"
# screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_7.txt --output_file gpt4_7_new.csv --col_name gpt4_7"

screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_0.txt --output_file gpt4_0_new.csv --col_name gpt4_0_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_1.txt --output_file gpt4_1_new.csv --col_name gpt4_1_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_2.txt --output_file gpt4_2_new.csv --col_name gpt4_2_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_3.txt --output_file gpt4_3_new.csv --col_name gpt4_3_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_4.txt --output_file gpt4_4_new.csv --col_name gpt4_4_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_5.txt --output_file gpt4_5_new.csv --col_name gpt4_5_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_6.txt --output_file gpt4_6_new.csv --col_name gpt4_6_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_8.txt --output_file gpt4_8_new.csv --col_name gpt4_8_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_9.txt --output_file gpt4_9_new.csv --col_name gpt4_9_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_10.txt --output_file gpt4_10_new.csv --col_name gpt4_10_new"
screen -dm bash -c "python3 -m translator_gpt --model gpt-4-turbo-2024-04-09 --prompt_file_path prompts/GPT4/prompt_11.txt --output_file gpt4_11_new.csv --col_name gpt4_11_new"
