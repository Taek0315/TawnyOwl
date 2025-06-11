import pandas as pd
import os

# 현재 파일 위치 기준 경로 구성
base_dir = os.path.dirname(os.path.abspath(__file__))
score_path = os.path.join(base_dir, "score_ele.xlsx")

# 규준표 로딩
# 흥미(interest)
interest_score_dict = {
    'S': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="interest_S")['score'],
                              pd.read_excel(score_path, sheet_name="interest_S")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="interest_S")['score'],
                              pd.read_excel(score_path, sheet_name="interest_S")['T_score_f']))
    },
    'E': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="interest_E")['score'],
                              pd.read_excel(score_path, sheet_name="interest_E")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="interest_E")['score'],
                              pd.read_excel(score_path, sheet_name="interest_E")['T_score_f']))
    },
    'C': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="interest_C")['score'],
                              pd.read_excel(score_path, sheet_name="interest_C")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="interest_C")['score'],
                              pd.read_excel(score_path, sheet_name="interest_C")['T_score_f']))
    },
    'R': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="interest_R")['score'],
                              pd.read_excel(score_path, sheet_name="interest_R")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="interest_R")['score'],
                              pd.read_excel(score_path, sheet_name="interest_R")['T_score_f']))
    },
    'I': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="interest_I")['score'],
                              pd.read_excel(score_path, sheet_name="interest_I")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="interest_I")['score'],
                              pd.read_excel(score_path, sheet_name="interest_I")['T_score_f']))
    },
    'A': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="interest_A")['score'],
                              pd.read_excel(score_path, sheet_name="interest_A")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="interest_A")['score'],
                              pd.read_excel(score_path, sheet_name="interest_A")['T_score_f']))
    },
}

# 적성(apti)
apti_score_dict = {
    'S': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="apti_S")['score'],
                              pd.read_excel(score_path, sheet_name="apti_S")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="apti_S")['score'],
                              pd.read_excel(score_path, sheet_name="apti_S")['T_score_f']))
    },
    'E': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="apti_E")['score'],
                              pd.read_excel(score_path, sheet_name="apti_E")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="apti_E")['score'],
                              pd.read_excel(score_path, sheet_name="apti_E")['T_score_f']))
    },
    'C': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="apti_C")['score'],
                              pd.read_excel(score_path, sheet_name="apti_C")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="apti_C")['score'],
                              pd.read_excel(score_path, sheet_name="apti_C")['T_score_f']))
    },
    'R': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="apti_R")['score'],
                              pd.read_excel(score_path, sheet_name="apti_R")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="apti_R")['score'],
                              pd.read_excel(score_path, sheet_name="apti_R")['T_score_f']))
    },
    'I': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="apti_I")['score'],
                              pd.read_excel(score_path, sheet_name="apti_I")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="apti_I")['score'],
                              pd.read_excel(score_path, sheet_name="apti_I")['T_score_f']))
    },
    'A': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="apti_A")['score'],
                              pd.read_excel(score_path, sheet_name="apti_A")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="apti_A")['score'],
                              pd.read_excel(score_path, sheet_name="apti_A")['T_score_f']))
    },
}

# 직업선호(job)
job_score_dict = {
    'S': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="job_S")['score'],
                              pd.read_excel(score_path, sheet_name="job_S")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="job_S")['score'],
                              pd.read_excel(score_path, sheet_name="job_S")['T_score_f']))
    },
    'E': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="job_E")['score'],
                              pd.read_excel(score_path, sheet_name="job_E")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="job_E")['score'],
                              pd.read_excel(score_path, sheet_name="job_E")['T_score_f']))
    },
    'C': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="job_C")['score'],
                              pd.read_excel(score_path, sheet_name="job_C")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="job_C")['score'],
                              pd.read_excel(score_path, sheet_name="job_C")['T_score_f']))
    },
    'R': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="job_R")['score'],
                              pd.read_excel(score_path, sheet_name="job_R")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="job_R")['score'],
                              pd.read_excel(score_path, sheet_name="job_R")['T_score_f']))
    },
    'I': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="job_I")['score'],
                              pd.read_excel(score_path, sheet_name="job_I")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="job_I")['score'],
                              pd.read_excel(score_path, sheet_name="job_I")['T_score_f']))
    },
    'A': {
        'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="job_A")['score'],
                              pd.read_excel(score_path, sheet_name="job_A")['T_score_m'])),
        'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="job_A")['score'],
                              pd.read_excel(score_path, sheet_name="job_A")['T_score_f']))
    },
}

# 직업환경(env)
env_p_score_dict = {
    'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="env_P")['score'],
                          pd.read_excel(score_path, sheet_name="env_P")['T_score_m'])),
    'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="env_P")['score'],
                          pd.read_excel(score_path, sheet_name="env_P")['T_score_f']))
}

env_t_score_dict = {
    'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="env_T")['score'],
                          pd.read_excel(score_path, sheet_name="env_T")['T_score_m'])),
    'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="env_T")['score'],
                          pd.read_excel(score_path, sheet_name="env_T")['T_score_f']))
}

env_d_score_dict = {
    'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="env_D")['score'],
                          pd.read_excel(score_path, sheet_name="env_D")['T_score_m'])),
    'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="env_D")['score'],
                          pd.read_excel(score_path, sheet_name="env_D")['T_score_f']))
}

env_i_score_dict = {
    'T_score_m': dict(zip(pd.read_excel(score_path, sheet_name="env_I")['score'],
                          pd.read_excel(score_path, sheet_name="env_I")['T_score_m'])),
    'T_score_f': dict(zip(pd.read_excel(score_path, sheet_name="env_I")['score'],
                          pd.read_excel(score_path, sheet_name="env_I")['T_score_f']))
}

# 학과/직업/설명
major_df = pd.read_excel(score_path, sheet_name="major")
joblist_df = pd.read_excel(score_path, sheet_name="joblist")
description_df = pd.read_excel(score_path, sheet_name="description")
