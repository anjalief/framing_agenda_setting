# Run granger casaulity test
# Input files should be the output of replace_country_mentions
from econ_utils import make_percent_change, load_econ_file, do_granger_test
import argparse
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--econ_file")
    parser.add_argument('--timestep', type=str, help="specify what time increment to use for aggregating articles",
                        default='monthly',
                        choices=['monthly', 'quarterly', 'semi', 'yearly'])
    args = parser.parse_args()

    date_seq = []
    freq_seq = []
    for line in open(args.input_file).readlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        date = datetime.datetime.strptime(parts[0], "%Y-%m-%d").date()
        date_seq.append(date)
        freq_seq.append(float(parts[3]))

    # cut first date because we take % change

    freq_seq = make_percent_change(freq_seq)

    econ_seq = load_econ_file(args.econ_file, args.timestep, date_seq)

    econ_seq = make_percent_change(econ_seq)

    # This is the python version, cleaner but less informative
    # print(do_granger_test(freq_seq, econ_seq))

    # R script gives the breakdown for each variable
    fp = open("./granger_text_input.txt", "w")
    for x in freq_seq:
        fp.write("%.10f\n" % x)
    fp.close()

    fp = open("./granger_econ_input.txt", "w")
    for x in econ_seq:
        fp.write("%.10f\n" % x)
    fp.close()

    import subprocess, os
    subprocess.call(["Rscript", "./do_granger.R"])

    # clean up
    os.remove("./granger_text_input.txt")
    os.remove("./granger_econ_input.txt")

if __name__ == "__main__":
    main()
