

######################################### Won't run without full data set ##########################################################################
# # Preprocessing
# python replace_country_mentions.py --article_glob ~/corpora/russian/Izvestiia/*/*/txt.tok  --outpath ../outputs/Izvestiia_processed

# # Runs that reproduce correlations in paper:
# python count_all_countries.py --timestep monthly --econ_file ../data/econ_files/russian_rtsi_rub.csv --word_level > ../outputs/monthly_word.txt
# python count_all_countries.py --timestep monthly --econ_file ../data/econ_files/russian_rtsi_rub.csv > ../outputs/monthly_article.txt

# python count_all_countries.py --timestep quarterly --word_level --econ_file ../data/econ_files/russian_quarterly_gdp.csv > ../outputs/quarterly_word.txt
# python count_all_countries.py --timestep quarterly --econ_file ../data/econ_files/russian_quarterly_gdp.csv > ../outputs/quarterly_article.txt

# python count_all_countries.py --timestep yearly --word_level --econ_file ../data/econ_files/russian_yearly_gdp.csv > ../outputs/yearly_word.txt
# python count_all_countries.py --timestep yearly --econ_file ../data/econ_files/russian_yearly_gdp.csv > ../outputs/yearly_article.txt

###################################################################################################################################################

# Granger casaulity

python do_granger.py --input_file ../outputs/monthly_word.txt --econ_file ../data/econ_files/russian_rtsi_usd.csv
python do_granger.py --input_file ../outputs/monthly_article.txt --econ_file ../data/econ_files/russian_rtsi_usd.csv