# loader_univ.py

import json
import os

# 현재 파일 위치 기준
base_dir = os.path.dirname(os.path.abspath(__file__))

# helper 함수
def load_score_json(filename):
    with open(os.path.join(base_dir, filename), "r", encoding="utf-8") as f:
        return json.load(f)

# 규준별 로드
interest_score_dict = load_score_json("score_univ_interest.json")
person_score_dict   = load_score_json("score_univ_person.json")
apti_score_dict     = load_score_json("score_univ_apti.json")
job_score_dict      = load_score_json("score_univ_job.json")
env_p_t_score_dict  = load_score_json("score_univ_envp_t.json")
env_d_i_score_dict  = load_score_json("score_univ_envd_i.json")

# major/joblist/description → 여기는 기존처럼 엑셀에서 df 로 로드
import pandas as pd

score_path = os.path.join(base_dir, "score_univ.xlsx")

major_df       = pd.read_excel(score_path, sheet_name="major")
joblist_df     = pd.read_excel(score_path, sheet_name="joblist")
description_df = pd.read_excel(score_path, sheet_name="description")
