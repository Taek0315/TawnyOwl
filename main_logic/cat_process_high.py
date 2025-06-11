import pandas as pd
import numpy as np
import warnings

# score 모듈 import
from main_logic.score import high as score_module

# 기존 process_single_person import
from main_logic.CET_logic_high_row import process_single_person

# 메인 함수
def cet_process(dict_data):
    """
    dict_data: 1명 응답 데이터 (dict 형태)
    return: 결과 dict
    """

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # dict → pd.Series 변환
    row = pd.Series(dict_data)

    # score_dicts 내부 구성
    score_dicts = {
        'interest_score_dict': score_module.interest_score_dict,
        'person_score_dict': score_module.person_score_dict,
        'apti_score_dict': score_module.apti_score_dict,
        'job_score_dict': score_module.job_score_dict,
        'env_p_t_score_dict': score_module.env_p_t_score_dict,
        'env_d_i_score_dict': score_module.env_d_i_score_dict
    }

    # process_single_person 호출
    result = process_single_person(
        row,
        score_dicts['interest_score_dict'],
        score_dicts['person_score_dict'],
        score_dicts['apti_score_dict'],
        score_dicts['job_score_dict'],
        score_dicts['env_p_t_score_dict'],
        score_dicts['env_d_i_score_dict'],
        score_module.major_df,
        score_module.joblist_df,
        score_module.description_df
    )

    return result
