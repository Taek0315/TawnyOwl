# main_logic/cat_process_ele.py (예시)
from main_logic.score import ele as score_module
from main_logic.CET_logic_ele_row import process_single_person
import pandas as pd

def cet_process(dict_data):
    row = pd.Series(dict_data)

    score_dicts = {
        'interest_score_dict': score_module.interest_score_dict,
        'apti_score_dict': score_module.apti_score_dict,
        'job_score_dict': score_module.job_score_dict,
        'env_p_score_dict': score_module.env_p_score_dict,
        'env_t_score_dict': score_module.env_t_score_dict,
        'env_d_score_dict': score_module.env_d_score_dict,
        'env_i_score_dict': score_module.env_i_score_dict
    }

    result = process_single_person(
        row,
        score_dicts['interest_score_dict'],
        score_dicts['apti_score_dict'],
        score_dicts['job_score_dict'],
        score_dicts['env_p_score_dict'],
        score_dicts['env_t_score_dict'],
        score_dicts['env_d_score_dict'],
        score_dicts['env_i_score_dict'],
        score_module.major_df,
        score_module.joblist_df,
        score_module.description_df
    )
    return result
