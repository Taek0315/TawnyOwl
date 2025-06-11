import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
score_path = os.path.join(base_dir, "score_high.xlsx")

# 규준표 로드
interest_score = pd.read_excel(score_path, sheet_name="interest")
person_score   = pd.read_excel(score_path, sheet_name="person")
apti_score     = pd.read_excel(score_path, sheet_name="apti")
job_score      = pd.read_excel(score_path, sheet_name="job")
env_p_t_score  = pd.read_excel(score_path, sheet_name="env(p_t)")
env_d_i_score  = pd.read_excel(score_path, sheet_name="env(d_i)")
major_df       = pd.read_excel(score_path, sheet_name="major")
joblist_df     = pd.read_excel(score_path, sheet_name="joblist")
description_df = pd.read_excel(score_path, sheet_name="description")

# dict 준비 (미리 준비 → process_single_person에 그대로 넘김)
interest_score_dict = {
    'T_score_m': dict(zip(interest_score['score'], interest_score['T_score_m'])),
    'T_score_f': dict(zip(interest_score['score'], interest_score['T_score_f']))
}
person_score_dict = {
    'T_score_m': dict(zip(person_score['score'], person_score['T_score_m'])),
    'T_score_f': dict(zip(person_score['score'], person_score['T_score_f']))
}
apti_score_dict = {
    'T_score_m': dict(zip(apti_score['score'], apti_score['T_score_m'])),
    'T_score_f': dict(zip(apti_score['score'], apti_score['T_score_f']))
}
job_score_dict = {
    'T_score_m': dict(zip(job_score['score'], job_score['T_score_m'])),
    'T_score_f': dict(zip(job_score['score'], job_score['T_score_f']))
}
env_p_t_score_dict = {
    'T_score_m': dict(zip(env_p_t_score['score'], env_p_t_score['T_score_m'])),
    'T_score_f': dict(zip(env_p_t_score['score'], env_p_t_score['T_score_f']))
}
env_d_i_score_dict = {
    'T_score_m': dict(zip(env_d_i_score['score'], env_d_i_score['T_score_m'])),
    'T_score_f': dict(zip(env_d_i_score['score'], env_d_i_score['T_score_f']))
}
