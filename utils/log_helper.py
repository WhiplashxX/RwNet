import pickle
import os

import pandas as pd


def save_obj(obj, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(dir, name):
    with open(dir + name + '.pkl', 'rb') as f:
        data = pickle.load(f)
        data.fillna(0, inplace=True)
        columns_to_convert = ['nevents', 'explored', 'grade_reqs', 'nforum_posts', 'course_length', 'ndays_act']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        return data


def save_args(args):
    args_dict = args.args_to_dict
    save_obj(args_dict, args.log_dir, 'args_dict')

    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
        args = ['{} : {}'.format(key, args_dict[key]) for key in args_dict]
        f.write('\n'.join(args))   
