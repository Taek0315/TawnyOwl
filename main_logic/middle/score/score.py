import pandas as pd
import os

# 현재 파일 위치 기준 경로 구성
base_dir = os.path.dirname(os.path.abspath(__file__))
score_path = os.path.join(base_dir, "score_middle.xlsx")

# 규준표 로드 및 dict 변환

# 흥미
interest_score = pd.read_excel(score_path, sheet_name="interest")
interest_score_dict = {
    'T_score_m': dict(zip(interest_score['score'], interest_score['T_score_m'])),
    'T_score_f': dict(zip(interest_score['score'], interest_score['T_score_f']))
}

# 성격
person_score = pd.read_excel(score_path, sheet_name="person")
person_score_dict = {
    'T_score_m': dict(zip(person_score['score'], person_score['T_score_m'])),
    'T_score_f': dict(zip(person_score['score'], person_score['T_score_f']))
}

# 적성
apti_score = pd.read_excel(score_path, sheet_name="apti")
apti_score_dict = {
    'T_score_m': dict(zip(apti_score['score'], apti_score['T_score_m'])),
    'T_score_f': dict(zip(apti_score['score'], apti_score['T_score_f']))
}

# 선호직업
job_score = pd.read_excel(score_path, sheet_name="job")
job_score_dict = {
    'T_score_m': dict(zip(job_score['score'], job_score['T_score_m'])),
    'T_score_f': dict(zip(job_score['score'], job_score['T_score_f']))
}

# 직업환경 (env_p_t)
env_p_t_score = pd.read_excel(score_path, sheet_name="env(p_t)")
env_p_t_score_dict = {
    'T_score_m': dict(zip(env_p_t_score['score'], env_p_t_score['T_score_m'])),
    'T_score_f': dict(zip(env_p_t_score['score'], env_p_t_score['T_score_f']))
}

# 직업환경 (env_d_i)
env_d_i_score = pd.read_excel(score_path, sheet_name="env(d_i)")
env_d_i_score_dict = {
    'T_score_m': dict(zip(env_d_i_score['score'], env_d_i_score['T_score_m'])),
    'T_score_f': dict(zip(env_d_i_score['score'], env_d_i_score['T_score_f']))
}

# 학과/직업/설명
major_df = pd.read_excel(score_path, sheet_name="major")
joblist_df = pd.read_excel(score_path, sheet_name="joblist")
description_df = pd.read_excel(score_path, sheet_name="description")
