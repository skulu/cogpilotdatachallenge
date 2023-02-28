from datetime import datetime, timedelta
# import heartpy as hp
import numpy as np
import os
import pandas as pd
import scipy as sp
from sktime.transformations.panel.padder import PaddingTransformer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Gets csv directories in original folder structure
# cap_ID format:'cp003', level format:'01B', sensor format:'lslshimmertorsoacc'
def get_dirs_to_csv(data_dir, cap_ID, level, sensor):
    # get the file direcotries for target cap_ID
    all_cap_dirs = os.listdir(data_dir)
    target_cap_dir = [d for d in all_cap_dirs if cap_ID in d]
    # get the session direcotry for target cap_ID
    target_cap_session_dir_name = [d for d in 
                                   os.listdir(os.path.join(data_dir, target_cap_dir[0])) 
                                   if os.path.isdir(os.path.join(data_dir, target_cap_dir[0],d))]
    target_cap_session_dir = os.path.join(data_dir, target_cap_dir[0], target_cap_session_dir_name[0])
    
    if level is None:
        target_cap_level_dir_names = [d for d in 
                                    os.listdir(target_cap_session_dir)]
    else:
        # get 3 directories of target level for target cap_ID
        target_cap_level_dir_names = [d for d in 
                                    os.listdir(target_cap_session_dir) if level in d]
    
    target_cap_level_dirs = [os.path.join(target_cap_session_dir, d) for d in target_cap_level_dir_names]
    # get 3 data files that contain sensor names from 3 level directories
    keywords = [sensor, 'dat']
    matched_files = []
    for d in target_cap_level_dirs:
        for f in os.listdir(d):
            if all(keyword in f for keyword in keywords) and os.path.isfile(os.path.join(d, f)):
                matched_files.append(os.path.join(d, f))
    return matched_files


# Gets effective frequency of data
def get_csv_freq(data_csv_dir):
    header_dir=data_csv_dir[:-7]+'hea.csv'
    df = pd.read_csv(header_dir)
    sr = df['Fs_Hz_effective'][0]
    sr = int(sr) + (sr % 1 > 0)
    return sr


def get_head_tail_time_to_remove(csv_dir):
    folder_dir = os.path.split(csv_dir)[0]
    xp11_name = [d for d in os.listdir(folder_dir) if 'lslxp11xpcac' in d and 'dat' in d]
    xp11_dir = os.path.join(folder_dir, xp11_name[0])
    if os.path.isfile(xp11_dir):
        df = pd.read_csv(xp11_dir)
        # take only the sensor columns, drop the time column
        df1 = df.drop(df.columns[0], axis=1)
        row_sum = df1.sum(axis=1)
        # get the length of stride of 1st values and stride of last values
        occ_head = row_sum.value_counts()[row_sum[0]]
        occ_tail = row_sum.value_counts()[row_sum.tail(1).values[0]]
        # get the time points: t0 - inital time, t1 first real data, t2 last real data time, t3 last time
        if occ_head > 1: # if more stride exist
            t0 = df.iloc[0,0]
            t0_real = datetime.fromordinal(int(t0)) + timedelta(days=t0%1) - timedelta(days = 366)
            t1 = df.iloc[occ_head,0]
            t1_real = datetime.fromordinal(int(t1)) + timedelta(days=t1%1) - timedelta(days = 366)
            delta = t1_real - t0_real
            head = delta.total_seconds() 
        else: head = 0
        if occ_tail > 1: 
            t2 = df.iloc[-occ_tail,0]
            t2_real = datetime.fromordinal(int(t2)) + timedelta(days=t2%1) - timedelta(days = 366)
            t3 = df.iloc[-1,0]
            t3_real = datetime.fromordinal(int(t3)) + timedelta(days=t3%1) - timedelta(days = 366)
            delta = t3_real - t2_real
            tail = delta.total_seconds() 
        else: tail = 0
    else: 
        head, tail = (0,0) # if xp11 file do not exist, assume 0,0
    return (head, tail)


# List of CSV filenames
def get_all_data_csv_filenames(data_dir, sensor = '', subject = '', level = ''): 
    csv_files = []
    # Recursively search for CSV files in the subdirectories of the current directory
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv') and (sensor in f) and (subject in f) and (level in f) and os.path.isfile(os.path.join(root, f)):
                csv_files.append(f)
    return csv_files


# Generate df_runs, which contains the subject, difficulty, run and runtime data for all test runs of interest
def get_df_runs(data_dir, sensor, subject, level, printing=True, eval=False):
    if type(subject) is list:
        fnames = []
        for sub in subject:
            fnames.extend(get_all_data_csv_filenames(data_dir, sensor, sub, level))
    else:
        fnames = get_all_data_csv_filenames(data_dir, sensor, subject, level)

    subject_list, difficulty, run, date, time = [], [], [], [], []
    for f in fnames:
        split_name = f.split('_')
        if eval:
            subject_list.append(split_name[1])
            difficulty.append('unknown')
            
        else:
            subject_list.append(split_name[2])
            difficulty.append(split_name[1])

        run.append(split_name[-1][0:2])
        date.append(split_name[-2])
        df = pd.read_csv(data_dir + '\\' + f)
        time.append(df.iloc[-1,0])
        # print(f'{len(run)}, {len(subject_list)}, {len(difficulty)}, {len(date)}, {len(time)}')
        # print()
    df_runs = pd.DataFrame({'subject': subject_list, 'difficulty': difficulty, 'run': run, 'date': date, 'time': time})
    if printing: print(f'Number of runs detected: {df_runs.shape[0]}')

    return df_runs


def get_df_runs_htceye(data_dir, sensor, subject, level, printing=True, eval=False):
    if type(subject) is list:
        fnames = []
        for sub in subject:
            fnames.extend(get_all_data_csv_filenames(data_dir, sensor, sub, level))
    else:
        fnames = get_all_data_csv_filenames(data_dir, sensor, subject, level)

    subject_list, difficulty, run, date, length = [], [], [], [], []
    for f in fnames:
        split_name = f.split('_')
        # print(split_name)
        if eval:
            difficulty.append('unknown')
        else:
            difficulty.append(split_name[4])
        
        run.append(split_name[-1][0:2])
        date.append(split_name[-2])
        subject_list.append(split_name[-3])
        df = pd.read_csv(data_dir + '\\' + f)
        length.append(df.shape[0])
        

    df_runs = pd.DataFrame({'subject': subject_list, 'difficulty': difficulty, 'run': run, 'date': date, 'length': length})
    if printing: print(f'Number of runs detected: {df_runs.shape[0]}')

    return df_runs


def scale_and_pad(data_dir, scaler, pad_length, df_runs, sensor, eval=False, eye=False):
    pad = PaddingTransformer(pad_length = pad_length)

    # Get how many columns of data each file has
    sub = df_runs.iloc[0,0]
    dif = df_runs.iloc[0,1]
    run = df_runs.iloc[0,2]
    date = df_runs.iloc[0,3]
    if eval:
        path = os.path.join(data_dir, sensor + '_' + sub + '_' + date + '_' + run + '.csv')
    else:
        path = os.path.join(data_dir, sensor + '_' + dif + '_' + sub + '_' + date + '_' + run + '.csv')
    df_tmp = pd.read_csv(path)

    if not eye:
        X = np.zeros((df_runs.shape[0], df_tmp.shape[1]-1, pad_length))
    else:
        X = np.zeros((df_runs.shape[0], df_tmp.shape[1], pad_length))
    y = []

    for i, row in df_runs.iterrows():
        sub = row['subject']
        dif = row['difficulty']
        run = row['run']
        date = row['date']
        if eval:
            path = os.path.join(data_dir, sensor + '_' + sub + '_' + date + '_' + run + '.csv')
        else:
            path = os.path.join(data_dir, sensor + '_' + dif + '_' + sub + '_' + date + '_' + run + '.csv')
        
        df_tmp = pd.read_csv(path)
        if scaler and not eye:
            df_tmp.iloc[:,1:] = scaler.fit_transform(df_tmp.iloc[:,1:])
        if scaler and eye:
            df_tmp = scaler.fit_transform(df_tmp)
        
        if not eye:
            X[i] = pad.fit_transform(df_tmp.iloc[:,1:]).T.values
        else:
            X[i] = pad.fit_transform(df_tmp).T

        dif_dict = {'01B': 0, '02B': 1, '03B': 2, '04B': 3}
        if not eval: 
            y.append(dif_dict[dif])
    
    if not eval: 
        y = np.array(y)
    return X, y


# Combine each sensor's data to one x numpy file and associated y numpy file
def generate_ml_data(data_dir, target_dir, file_suffix, scaler, pad_length, sensor = '', subject = '', level = '', eval=False, eye=False):
    # df_runs contains the subject, difficulty, run and runtime data for all test runs of interest
    
    if not eye:
        df_runs = get_df_runs(data_dir, sensor, subject, level, True, eval)
    else:
        df_runs = get_df_runs_htceye(data_dir, sensor, subject, level, True, eval)

    X, y = scale_and_pad(data_dir, scaler, pad_length, df_runs, sensor, eval, eye)
    print(f'X shape: {X.shape}')
    if not eval: print(f'y shape: {y.shape}')

    # Save
    np.save(os.path.join(target_dir, 'X_' + file_suffix + '.npy'), X)
    if not eval:
        np.save(os.path.join(target_dir, 'y_' + file_suffix + '.npy'), y)
        print('Saved files:\n\t' + 'X_' + file_suffix + '.npy' + '\n\t' + 'y_' + file_suffix + '.npy')
    else:
        print('Saved files:\n\t' + 'X_' + file_suffix + '.npy')


def find_peaks(df, col):
    x = df.iloc[:, col]
    # define quantile points
    Q3 = np.quantile(x, 0.75)
    Q2 = np.quantile(x, 0.5)
    Q1 = np.quantile(x, 0.25)
    IQR = Q3-Q1
    # find peak locations - peak should be above Q2, distance of 0.5s peak height above IQR
    peaks,_ = sp.signal.find_peaks(x, height=Q2, distance=30, prominence = IQR)
    # for plotting peaks
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    # plt.show()
    return peaks


def find_median_heart_rate(peaks):
    if len(peaks) > 5:
        intervals = np.diff(peaks)
        median_interval = np.median(intervals)
        # data freq 60Hz (lpf cutoff freq 30Hz), heart rate per sec: 1/(n/f) per min = 60f/n
        heart_rate = 3600/median_interval
    else: heart_rate = np.NaN
    return heart_rate


## -- RESULT LOGGER -- ##
def get_class(class_list, prob_list):
    idx = list(prob_list).index(max(prob_list))
    return class_list[idx]

def log_result(classifier, class_list, y_test, y_pred_proba, model_result = None):
    if model_result is None:
        model_result = {
            "classifier":[],
            "accuracy_score":[],
            "AUC_score":[],
            "F1_score":[]
        }

    y_pred = []
    for y_list in y_pred_proba:
        y_pred.append(get_class(class_list, y_list))
    acc = accuracy_score(y_test, y_pred)
  
    # On which average to use: https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average = 'macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    model_result["classifier"].append(classifier)
    model_result["accuracy_score"].append(acc)
    model_result["AUC_score"].append(auc)
    model_result["F1_score"].append(f1)

    # display(pd.DataFrame(model_result))
    return model_result