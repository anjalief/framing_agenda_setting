from datetime import date
import datetime
from article_utils import *

RTSI_OPEN_IDX=0
RTSI_MIN_IDX=1
RTSI_MAX_IDX=2
RTSI_CLOSE_IDX=3

# if you've provided a date_to_idx mapping, return
# idx_to_val, where idx corresponds to date_to_idx map
def load_monthly_gdp(filename, date_to_idx = None):
    idx_to_val = {}
    for line in open(filename).readlines():
        splits = line.split(",")
        if not splits[0]: # we have some blank lines
            continue
        val = float(splits[1].strip())
        date_split = splits[0].split("-")
        # assumes 1st of the month
        d = date(int(date_split[0]), int(date_split[1]), 1)

        if date_to_idx:
            # don't need econ data for dates we don't have articles for
            if d in date_to_idx:
                idx = date_to_idx[d]
                idx_to_val[idx] = val
        else:
            idx_to_val[d] = val

    return idx_to_val

# file format is YYYY-QQ,value (1995-Q2,xxx)
def load_quarterly_gdp(filename, date_to_idx = None):
    idx_to_val = {}
    for line in open(filename).readlines():
        splits = line.split(",")
        if not splits[0]: # we have some blank lines
            continue
        val = float(splits[1].strip())
        date_split = splits[0].split("-")
        # assumes 1st of the month, assign quarter to 1st month
        quarter_month = quarter_to_month(int(date_split[1][1]))
        d = date(int(date_split[0]), quarter_month, 1)

        if date_to_idx:
            # don't need econ data for dates we don't have articles for
            if d in date_to_idx:
                idx = date_to_idx[d]
                idx_to_val[idx] = val
        else:
            idx_to_val[d] = val
    return idx_to_val

def load_yearly_gdp(filename):
    year_to_val = {}
    for line in open(filename).readlines():
        splits = line.split(",")
        d = date(int(splits[0]), 1, 1)
        year_to_val[d] = float(splits[1])
    return year_to_val

def load_rtsi(filename, date_to_idx = None):
    idx_to_val = {}
    for line in open(filename).readlines():
        splits = line.split(",")
        if not splits[0]:
            continue
        vals = [float(x) for x in splits[1:]]

        # order in file is day.month.year
        date_split = splits[0].split(".")
        d = date(int(date_split[2]), int(date_split[1]), int(date_split[0]))

        if date_to_idx:
            if d in date_to_idx:
                idx = date_to_idx[d]
                idx_to_val[idx] = vals
        else:
            idx_to_val[d] = vals
    return idx_to_val

# For now, we take last month in quarter as data for year
def load_quarterly_rtsi(filename):
    monthly_data = load_rtsi(filename)
    quarter_to_vals = {}
    for d in monthly_data:
        if d.month in [3, 6, 9, 12]:
            quarter_start = date(d.year, int(d.month) - 2, d.day)
            quarter_to_vals[quarter_start] = monthly_data[d]
    return quarter_to_vals

def load_semi_rtsi(filename):
    monthly_data = load_rtsi(filename)
    semi_to_vals = {}
    for d in monthly_data:
        if d.month in [6, 12]:
            semi_start = date(d.year, int(d.month) - 5, d.day)
            semi_to_vals[semi_start] = monthly_data[d]
    return semi_to_vals

# For now, we take last month in year  as data for year
def load_yearly_rtsi(filename):
    monthly_data = load_rtsi(filename)
    year_to_vals = {}
    for d in monthly_data:
        if d.month in [12]:
            year_start = date(d.year, 1, 1)
            year_to_vals[year_start] = monthly_data[d]
    return year_to_vals

def load_econ_file(econ_file, timestep, date_seq):
    if "gdp" in econ_file:
        if timestep == "monthly":
            date_to_gdp = load_monthly_gdp(econ_file)
        elif timestep == "quarterly":
            date_to_gdp = load_quarterly_gdp(econ_file)
        elif timestep == "yearly":
            date_to_gdp = load_yearly_gdp(econ_file)
        econ_seq = [date_to_gdp[d] for d in date_seq]
    elif "rtsi" in econ_file:
        if timestep == "monthly":
            date_to_rtsi = load_rtsi(econ_file)
        elif timestep == "quarterly":
            date_to_rtsi = load_quarterly_rtsi(econ_file)
        elif timestep == "semi":
            date_to_rtsi = load_semi_rtsi(econ_file)
        elif timestep == "yearly":
            date_to_rtsi = load_yearly_rtsi(econ_file)
        else:
            assert False, "Invalid timestep"
        econ_seq = [date_to_rtsi[d][RTSI_CLOSE_IDX] for d in date_seq]
    return econ_seq

def load_percent_change(filename):
    date_to_val = {}
    for line in open(filename).readlines():
        splits = line.split(",")
        d = datetime.datetime.strptime(splits[0], "%Y-%m-%d").date()
        date_to_val[d] = float(splits[1])
    return date_to_val

# create a differenced series
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        diff.append(dataset[i] - dataset[i - interval])
    return diff

# take % change of series
def make_percent_change(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        diff.append((dataset[i] / dataset[i - interval]) - 1)
    return diff

# average pooling (to smooth curves)
def average_interval(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset) - interval):
        diff.append(sum(dataset[i-interval:i+interval+1]) / (interval * 2 + 1))
    return diff

def get_corr(x, y):
    # this returns a matrix where 0,1 is the correlation between seq 0 and seq 1 (I think)
    # since we're only going to give in 1D lists, we can grab the top right element
    return numpy.corrcoef(x, y)[0, 1]


from statsmodels.tsa.stattools import grangercausalitytests
import numpy
def do_granger_test(series1, series2):
    usa_freq = numpy.array(series1)
    rtsi = numpy.array(series2)
    input = numpy.stack((usa_freq, rtsi), 1)

    return grangercausalitytests(input, maxlag=4)
