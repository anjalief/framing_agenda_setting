# framing_agenda_setting
This repository contains code for https://arxiv.org/abs/1808.09386 " Framing and Agenda-setting in Russian News: a Computational Analysis of Intricate Political Strategies"

Its contents, by directory are:

data:
   country_subs.txt: file containing list of all countries keywords
   econ_files: contains monthly, quarterly, yearly GDP data and monthly RTSI
   usa.txt: file containing "USA", used as input to count_all_countries

src:
   replace_country_mentions.py: used for preprocessing, i.e. collapse "America" and "United States" to USA
   count_all_countries.py: the main file for counting how many times a word is
                           mentioned in a particular time slice
          Output:
             default output:
                     [date (start of range)] [# articles containing keywords at least 2 times] [total # of articles in time slice] [# articles with keywords / # total articles]
             if you specify the parameter "--word_level", then the output is:
                     [date (start of range)] [# of keywords in timeslice] [total # of words in timeslice] [# keywords / # total words]
             if you provide an econ file, it will also print correlations
          Example run:
             python count_all_countries.py --timestep yearly --input_path "../data/Izvestiia_processed/" --subs_file "../data/usa.txt"
   do_granger.py: Run bsaic granger casaulity test (note: paper results were computed in R, not using this script)

