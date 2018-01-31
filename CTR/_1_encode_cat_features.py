import pandas as pd
import numpy as np
import sys
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state
import pylab
import time
import xgboost as xgb
from joblib import dump, load, Parallel, delayed
import ctrUtils

raw_data_path = ctrUtils.raw_data_path
tmp_data_path = ctrUtils.tmp_data_path

t0org0 = pd.read_csv(open(raw_data_path + "train", "ra"))
h0org = pd.read_csv(open(raw_data_path + "test", "ra"))

if ctrUtils.sample_pct < 1.0:
    np.random.seed(999)
    r1 = np.random.uniform(0,1,t0org0.shape[0])
    print("testing with small sample of training data, ", t0org0.shape)


h0org['click'] = 0
t0org = pd.concat([t0org0,h0org])
print("finished loading raw data, ", t0org.shape)
print("to add some basic features ...")
t0org['day']=np.round(t0org.hour % 10000 / 100)
t0org['hour1'] = np.round(t0org.hour % 100)
t0org['day_hour']=(t0org.day.values - 21) * 24 + t0org.hour1.values
t0org['day_hour_pre'] = t0org['day_hour'] - 1
t0org['day_hour_next'] = t0org['day_hour'] + 1
t0org['app_or_web'] = 0
t0org.ix[t0org.app_id.values=='ecad2386','app_or_web'] = 1

t0 = t0org
t0['app_site_id'] = np.add(t0.app_id.values, t0.site_id.values)
print("to encode categorical features using mean responses from earlier days -- univariate")
sys.stdout.flush()

calc_exptv(t0, ['app_or_web'])
calc_exptv(t0, exptv_vn_list)

calc_exptv(t0, ['app_site_id'], add_count=True)