import pandas as pd
import numpy as np
import warnings

########################################
# 1) 한 행(응답자)에 대한 모든 로직
########################################
def process_single_person(row_data,
                          # 규준표(점수표)들
                          interest_S_score, interest_E_score, interest_C_score, interest_R_score, interest_I_score, interest_A_score,
                          apti_S_score, apti_E_score, apti_C_score, apti_R_score, apti_I_score, apti_A_score,
                          job_S_score, job_E_score, job_C_score, job_R_score, job_I_score, job_A_score,
                          env_p_score, env_t_score, env_d_score, env_i_score,
                          major_df, joblist_df, description_df
                         ):
    """
    초등용 CET 로직을 '한 행(응답자 단위)'로 수행하여 결과를 dict로 반환하는 함수.
    row_data: pd.Series (해당 응답자의 데이터)
    """
    ############################
    # (A) 흥미/적성/직업환경/선호직업 합산 (문항별)
    ############################

    def safe_sum(cols):
        """row_data에서 주어진 컬럼명을 합산. NaN이면 0으로 간주."""
        s = 0
        for c in cols:
            val = row_data.get(c, 0)
            if pd.isna(val):
                val = 0
            s += val
        return s

    # 흥미
    interest_S = safe_sum(['Int_1','Int_7','Int_13','Int_19','Int_25','Int_31','Int_37','Int_43'])
    interest_E = safe_sum(['Int_2','Int_8','Int_14','Int_20','Int_26','Int_32','Int_38','Int_44'])
    interest_C = safe_sum(['Int_3','Int_9','Int_15','Int_21','Int_27','Int_33','Int_39','Int_45'])
    interest_R = safe_sum(['Int_4','Int_10','Int_16','Int_22','Int_28','Int_34','Int_40','Int_46'])
    interest_I = safe_sum(['Int_5','Int_11','Int_17','Int_23','Int_29','Int_35','Int_41','Int_47'])
    interest_A = safe_sum(['Int_6','Int_12','Int_18','Int_24','Int_30','Int_36','Int_42','Int_48'])

    # 적성
    apti_S = safe_sum(['Apt_1','Apt_7','Apt_13','Apt_19','Apt_25','Apt_31','Apt_37','Apt_43'])
    apti_E = safe_sum(['Apt_2','Apt_8','Apt_14','Apt_20','Apt_26','Apt_32','Apt_38','Apt_44'])
    apti_C = safe_sum(['Apt_3','Apt_9','Apt_15','Apt_21','Apt_27','Apt_33','Apt_39','Apt_45'])
    apti_R = safe_sum(['Apt_4','Apt_10','Apt_16','Apt_22','Apt_28','Apt_34','Apt_40','Apt_46'])
    apti_I = safe_sum(['Apt_5','Apt_11','Apt_17','Apt_23','Apt_29','Apt_35','Apt_41','Apt_47'])
    apti_A = safe_sum(['Apt_6','Apt_12','Apt_18','Apt_24','Apt_30','Apt_36','Apt_42','Apt_48'])

    # 직업환경(EJ)
    # EJn-1 => (row_data['EJ_{n}']==1)?1 else0
    # EJn-2 => (row_data['EJ_{n}']==2)?1 else0
    # 하지만 스크립트상 df['EJ2-2','EJ4-1'...]로 sum => 행단위: we'll define small helpers:
    def EJ1(n):
        # EJn-1
        val = row_data.get(f'EJ_{n}', np.nan)
        return 1 if (val==1) else 0
    def EJ2(n):
        # EJn-2
        val = row_data.get(f'EJ_{n}', np.nan)
        return 1 if (val==2) else 0

    env_D = ( EJ2(2)+ EJ1(4)+ EJ2(6)+ EJ1(8)+ EJ1(10)+ EJ2(12)+ EJ1(14)+ EJ2(16)+
              EJ2(18)+ EJ1(20)+ EJ1(22)+ EJ1(24)+ EJ1(26)+ EJ1(28)+ EJ2(30) )
    env_I = ( EJ1(2)+ EJ2(4)+ EJ1(6)+ EJ2(8)+ EJ2(10)+ EJ1(12)+ EJ2(14)+ EJ1(16)+
              EJ1(18)+ EJ2(20)+ EJ2(22)+ EJ2(24)+ EJ2(26)+ EJ2(28)+ EJ1(30) )
    env_P = ( EJ2(1)+ EJ1(3)+ EJ2(5)+ EJ1(7)+ EJ2(9)+ EJ1(11)+ EJ2(13)+ EJ1(15)+
              EJ2(17)+ EJ1(19)+ EJ2(21)+ EJ1(23)+ EJ2(25)+ EJ1(27)+ EJ2(29) )
    env_T = ( EJ1(1)+ EJ2(3)+ EJ1(5)+ EJ2(7)+ EJ1(9)+ EJ2(11)+ EJ1(13)+ EJ2(15)+
              EJ1(17)+ EJ2(19)+ EJ1(21)+ EJ2(23)+ EJ1(25)+ EJ2(27)+ EJ1(29) )

    # 선호직업
    job_S = safe_sum(['pre_1','pre_7','pre_13','pre_19','pre_25','pre_31','pre_37','pre_43'])
    job_E = safe_sum(['pre_2','pre_8','pre_14','pre_20','pre_26','pre_32','pre_38','pre_44'])
    job_C = safe_sum(['pre_3','pre_9','pre_15','pre_21','pre_27','pre_33','pre_39','pre_45'])
    job_R = safe_sum(['pre_4','pre_10','pre_16','pre_22','pre_28','pre_34','pre_40','pre_46'])
    job_I = safe_sum(['pre_5','pre_11','pre_17','pre_23','pre_29','pre_35','pre_41','pre_47'])
    job_A = safe_sum(['pre_6','pre_12','pre_18','pre_24','pre_30','pre_36','pre_42','pre_48'])

    ############################
    # (B) T점수 매핑
    ############################
    gender = row_data.get('성별', np.nan)  # 1=남, else=여?

    def lookup_Tscore(raw_score, score_df_m, score_df_f):
        """raw_score(합산점수)에 대해 남/여별 T점수 lookup."""
        if pd.isna(gender):
            return None
        if gender == 1:
            return score_df_m.get(raw_score, None)
        else:
            return score_df_f.get(raw_score, None)

    # 먼저, score_df => map
    # 예: interest_S_score['score'] vs interest_S_score['T_score_m']
    # 만약 row not found => None
    map_intS_m = dict(zip(interest_S_score['score'], interest_S_score['T_score_m']))
    map_intS_f = dict(zip(interest_S_score['score'], interest_S_score['T_score_f']))
    interest_S_T = lookup_Tscore(interest_S, map_intS_m, map_intS_f)

    map_intE_m = dict(zip(interest_E_score['score'], interest_E_score['T_score_m']))
    map_intE_f = dict(zip(interest_E_score['score'], interest_E_score['T_score_f']))
    interest_E_T = lookup_Tscore(interest_E, map_intE_m, map_intE_f)

    map_intC_m = dict(zip(interest_C_score['score'], interest_C_score['T_score_m']))
    map_intC_f = dict(zip(interest_C_score['score'], interest_C_score['T_score_f']))
    interest_C_T = lookup_Tscore(interest_C, map_intC_m, map_intC_f)

    map_intR_m = dict(zip(interest_R_score['score'], interest_R_score['T_score_m']))
    map_intR_f = dict(zip(interest_R_score['score'], interest_R_score['T_score_f']))
    interest_R_T = lookup_Tscore(interest_R, map_intR_m, map_intR_f)

    map_intI_m = dict(zip(interest_I_score['score'], interest_I_score['T_score_m']))
    map_intI_f = dict(zip(interest_I_score['score'], interest_I_score['T_score_f']))
    interest_I_T = lookup_Tscore(interest_I, map_intI_m, map_intI_f)

    map_intA_m = dict(zip(interest_A_score['score'], interest_A_score['T_score_m']))
    map_intA_f = dict(zip(interest_A_score['score'], interest_A_score['T_score_f']))
    interest_A_T = lookup_Tscore(interest_A, map_intA_m, map_intA_f)

    # 적성
    map_aptiS_m = dict(zip(apti_S_score['score'], apti_S_score['T_score_m']))
    map_aptiS_f = dict(zip(apti_S_score['score'], apti_S_score['T_score_f']))
    apti_S_T = lookup_Tscore(apti_S, map_aptiS_m, map_aptiS_f)

    map_aptiE_m = dict(zip(apti_E_score['score'], apti_E_score['T_score_m']))
    map_aptiE_f = dict(zip(apti_E_score['score'], apti_E_score['T_score_f']))
    apti_E_T = lookup_Tscore(apti_E, map_aptiE_m, map_aptiE_f)

    map_aptiC_m = dict(zip(apti_C_score['score'], apti_C_score['T_score_m']))
    map_aptiC_f = dict(zip(apti_C_score['score'], apti_C_score['T_score_f']))
    apti_C_T = lookup_Tscore(apti_C, map_aptiC_m, map_aptiC_f)

    map_aptiR_m = dict(zip(apti_R_score['score'], apti_R_score['T_score_m']))
    map_aptiR_f = dict(zip(apti_R_score['score'], apti_R_score['T_score_f']))
    apti_R_T = lookup_Tscore(apti_R, map_aptiR_m, map_aptiR_f)

    map_aptiI_m = dict(zip(apti_I_score['score'], apti_I_score['T_score_m']))
    map_aptiI_f = dict(zip(apti_I_score['score'], apti_I_score['T_score_f']))
    apti_I_T = lookup_Tscore(apti_I, map_aptiI_m, map_aptiI_f)

    map_aptiA_m = dict(zip(apti_A_score['score'], apti_A_score['T_score_m']))
    map_aptiA_f = dict(zip(apti_A_score['score'], apti_A_score['T_score_f']))
    apti_A_T = lookup_Tscore(apti_A, map_aptiA_m, map_aptiA_f)

    # job
    map_jobS_m = dict(zip(job_S_score['score'], job_S_score['T_score_m']))
    map_jobS_f = dict(zip(job_S_score['score'], job_S_score['T_score_f']))
    job_S_T = lookup_Tscore(job_S, map_jobS_m, map_jobS_f)

    map_jobE_m = dict(zip(job_E_score['score'], job_E_score['T_score_m']))
    map_jobE_f = dict(zip(job_E_score['score'], job_E_score['T_score_f']))
    job_E_T = lookup_Tscore(job_E, map_jobE_m, map_jobE_f)

    map_jobC_m = dict(zip(job_C_score['score'], job_C_score['T_score_m']))
    map_jobC_f = dict(zip(job_C_score['score'], job_C_score['T_score_f']))
    job_C_T = lookup_Tscore(job_C, map_jobC_m, map_jobC_f)

    map_jobR_m = dict(zip(job_R_score['score'], job_R_score['T_score_m']))
    map_jobR_f = dict(zip(job_R_score['score'], job_R_score['T_score_f']))
    job_R_T = lookup_Tscore(job_R, map_jobR_m, map_jobR_f)

    map_jobI_m = dict(zip(job_I_score['score'], job_I_score['T_score_m']))
    map_jobI_f = dict(zip(job_I_score['score'], job_I_score['T_score_f']))
    job_I_T = lookup_Tscore(job_I, map_jobI_m, map_jobI_f)

    map_jobA_m = dict(zip(job_A_score['score'], job_A_score['T_score_m']))
    map_jobA_f = dict(zip(job_A_score['score'], job_A_score['T_score_f']))
    job_A_T = lookup_Tscore(job_A, map_jobA_m, map_jobA_f)

    # env
    map_envD_m = dict(zip(env_d_score['score'], env_d_score['T_score_m']))
    map_envD_f = dict(zip(env_d_score['score'], env_d_score['T_score_f']))
    env_D_T = lookup_Tscore(env_D, map_envD_m, map_envD_f)

    map_envI_m = dict(zip(env_i_score['score'], env_i_score['T_score_m']))
    map_envI_f = dict(zip(env_i_score['score'], env_i_score['T_score_f']))
    env_I_T = lookup_Tscore(env_I, map_envI_m, map_envI_f)

    map_envP_m = dict(zip(env_p_score['score'], env_p_score['T_score_m']))
    map_envP_f = dict(zip(env_p_score['score'], env_p_score['T_score_f']))
    env_P_T = lookup_Tscore(env_P, map_envP_m, map_envP_f)

    map_envT_m = dict(zip(env_t_score['score'], env_t_score['T_score_m']))
    map_envT_f = dict(zip(env_t_score['score'], env_t_score['T_score_f']))
    env_T_T = lookup_Tscore(env_T, map_envT_m, map_envT_f)

    ############################
    # (C) factor 계산
    ############################
    # factor_S = ((interest_S_T*2) + (apti_S_T*2) + job_S_T) / 5
    # etc
    def safe_val(x):
        return x if x is not None else 0

    fS = ( safe_val(interest_S_T)*2 + safe_val(apti_S_T)*2 + safe_val(job_S_T) )/5
    fE = ( safe_val(interest_E_T)*2 + safe_val(apti_E_T)*2 + safe_val(job_E_T) )/5
    fC = ( safe_val(interest_C_T)*2 + safe_val(apti_C_T)*2 + safe_val(job_C_T) )/5
    fR = ( safe_val(interest_R_T)*2 + safe_val(apti_R_T)*2 + safe_val(job_R_T) )/5
    fI = ( safe_val(interest_I_T)*2 + safe_val(apti_I_T)*2 + safe_val(job_I_T) )/5
    fA = ( safe_val(interest_A_T)*2 + safe_val(apti_A_T)*2 + safe_val(job_A_T) )/5

    ############################
    # (D) 최종 순위(1~6) with tie-break
    ############################
    # Holland 육각형 거리
    distance_map = {
        ('S','S'):0, ('S','E'):1, ('S','C'):2, ('S','R'):3, ('S','I'):2, ('S','A'):1,
        ('E','S'):1, ('E','E'):0, ('E','C'):1, ('E','R'):2, ('E','I'):3, ('E','A'):2,
        ('C','S'):2, ('C','E'):1, ('C','C'):0, ('C','R'):1, ('C','I'):2, ('C','A'):3,
        ('R','S'):3, ('R','E'):2, ('R','C'):1, ('R','R'):0, ('R','I'):1, ('R','A'):2,
        ('I','S'):2, ('I','E'):3, ('I','C'):2, ('I','R'):1, ('I','I'):0, ('I','A'):1,
        ('A','S'):1, ('A','E'):2, ('A','C'):3, ('A','R'):2, ('A','I'):1, ('A','A'):0,
    }
    def get_distance(a,b):
        return distance_map[(a,b)]

    # 6 factor
    factor_dict = {
        'S': fS,
        'E': fE,
        'C': fC,
        'R': fR,
        'I': fI,
        'A': fA,
    }
    # 내림차순
    sorted_items = sorted(factor_dict.items(), key=lambda x: (-x[1], x[0]))

    # tie-break 함수 (간단화)
    def tie_break_top(sorted_items):
        # 생략 or 원본
        return sorted_items

    sorted_items = tie_break_top(sorted_items)
    sorted_labels = [x[0] for x in sorted_items]
    sorted_scores = [x[1] for x in sorted_items]

    rank_codes = {}
    rank_points= {}
    for i in range(6):
        rank_codes[i+1] = sorted_labels[i]
        rank_points[i+1] = round(sorted_scores[i],2)

    ############################
    # (E) 흥미/적성/직업선호 단순 순위(동점처리 X)
    ############################
    # 예: interest_S_T..interest_A_T => array => np.argsort
    def none0(x): return x if x is not None else 0
    interest_ts = [none0(interest_S_T), none0(interest_E_T), none0(interest_C_T),
                   none0(interest_R_T), none0(interest_I_T), none0(interest_A_T)]
    labels_secRIA = ['S','E','C','R','I','A']
    idxs = np.argsort(-np.array(interest_ts))
    int_lbls = [labels_secRIA[i] for i in idxs]
    int_vals= [round(interest_ts[i],2) for i in idxs]

    # 적성
    apti_ts = [none0(apti_S_T), none0(apti_E_T), none0(apti_C_T),
               none0(apti_R_T), none0(apti_I_T), none0(apti_A_T)]
    idxs2 = np.argsort(-np.array(apti_ts))
    apti_lbls = [labels_secRIA[i] for i in idxs2]
    apti_vals = [round(apti_ts[i],2) for i in idxs2]

    # 직업선호
    job_ts = [none0(job_S_T), none0(job_E_T), none0(job_C_T),
              none0(job_R_T), none0(job_I_T), none0(job_A_T)]
    idxs3= np.argsort(-np.array(job_ts))
    job_lbls=[labels_secRIA[i] for i in idxs3]
    job_vals=[round(job_ts[i],2) for i in idxs3]

    ############################
    # (F) 매칭개수
    ############################
    # 1~6위 => rank_codes[1] etc, i번째 => int_lbls[i-1], apti_lbls[i-1], job_lbls[i-1]
    matching_counts={}
    for i in range(1,7):
        code_i = rank_codes[i]  # 1순위= rank_codes[1]
        mc=0
        # 흥미 i번째
        if code_i == int_lbls[i-1]:
            mc+=1
        # 적성 i번째
        if code_i == apti_lbls[i-1]:
            mc+=1
        # 직업 i번째
        if code_i == job_lbls[i-1]:
            mc+=1
        matching_counts[i]=mc
    total_match = sum(matching_counts.values())
    reaction_fit_ratio = round((total_match/24)*100,2)

    ############################
    # (G) 반응적합, 변별도
    ############################
    # df_count = df.iloc[:,8:182] => 174개 문항 => row 단위
    # row_data: Int_1..48(48), Apt_1..48(48), EJ_1..30(30), pre_1..48(48) => 총 174
    # answered= # of non-NaN
    qcols=[]
    for i in range(1,49):
        qcols.append(f'Int_{i}')
    for i in range(1,49):
        qcols.append(f'Apt_{i}')
    for i in range(1,31):
        qcols.append(f'EJ_{i}')
    for i in range(1,49):
        qcols.append(f'pre_{i}')
    answered=0
    for c in qcols:
        val=row_data.get(c, np.nan)
        if pd.notna(val):
            answered+=1
    response_fit = round((answered/174)*100,2)

    # 변별도 = 1순위_점수 - ((2순위_점수 + 4순위_점수)/2)
    disc=None
    if all(x in rank_points for x in [1,2,4]):
        disc = round(rank_points[1] - ((rank_points[2]+rank_points[4])/2),2)

    ############################
    # (H) Code1, Code2
    ############################
    code1=None
    if response_fit<=70:
        code1='분류불능a'
    elif disc is not None and disc<1:
        code1='분류불능b'
    elif (fS<30 and fE<30 and fC<30 and fR<30 and fI<30 and fA<30):
        code1='분류불능c'
    elif pd.isna(gender):
        code1=None
    else:
        code1= rank_codes[1]+rank_codes[2]

    if code1 in ['분류불능a','분류불능b','분류불능c']:
        code2='분류불능'
    else:
        sc2=rank_points.get(2,0)
        sc3=rank_points.get(3,0)
        if (sc2 - sc3)>=3:
            code2 = rank_codes[2]+rank_codes[1]
        else:
            code2 = rank_codes[1]+rank_codes[3]

    # 1순위/2순위 코드 넘버
    priority_mapping = {'S':1,'E':2,'C':3,'R':4,'I':5,'A':6}
    fc_num = priority_mapping.get(rank_codes[1],None)
    sc_num = priority_mapping.get(rank_codes[2],None)

    ############################
    # (I) 직업환경 Code
    ############################
    # env_D_T, env_I_T, env_P_T, env_T_T
    # if all<30 => '분류불능d'
    # else => jobenv_code2 + jobenv_code1
    jobenv_code1='T'
    if env_P_T is not None and env_T_T is not None:
        if env_P_T>env_T_T:
            jobenv_code1='P'
        elif env_P_T<env_T_T:
            jobenv_code1='T'
        else:
            # ==
            if fc_num in [1,2,3]:
                jobenv_code1='P'
            else:
                jobenv_code1='T'
    jobenv_code2='I'
    if env_D_T is not None and env_I_T is not None:
        if env_D_T>env_I_T:
            jobenv_code2='D'
        elif env_D_T<env_I_T:
            jobenv_code2='I'
        else:
            if fc_num in [1,2,3]:
                jobenv_code2='D'
            else:
                jobenv_code2='I'

    jobenv_final=None
    if (env_D_T is not None and env_I_T is not None and
        env_P_T is not None and env_T_T is not None):
        if (env_D_T<30 and env_I_T<30 and env_P_T<30 and env_T_T<30):
            jobenv_final='분류불능d'
        else:
            jobenv_final=jobenv_code2+jobenv_code1

    final_code = None
    if code1 == '분류불능a':
        final_code = '분류불능A'
    elif code1 == '분류불능b':
        final_code = '분류불능B'
    elif code1 == '분류불능c':
        final_code = '분류불능C'
    elif jobenv_final == '분류불능d':
        final_code = code1
    else:
        final_code = code1 + jobenv_final

    ############################
    # (J) 학과/직업 리스트 매핑
    ############################
    def map_major(c):
        if c in ['분류불능a','분류불능b','분류불능c','분류불능']:
            return '분류불능'
        if c is None:
            return None
        return dict(zip(major_df['code'],major_df['major'])).get(c,None)

    def map_joblist(c):
        if c in ['분류불능a','분류불능b','분류불능c','분류불능']:
            return '분류불능'
        if c is None:
            return None
        return dict(zip(joblist_df['code'],joblist_df['job'])).get(c,None)
    
    description_map = dict(zip(description_df['code'],description_df['description']))

    def map_description(code_val):
        return description_map.get(code_val, None)

    major1 = map_major(code1)
    joblist1= map_joblist(code1)
    major2 = map_major(code2)
    joblist2= map_joblist(code2)
    description_1 = map_description(final_code)

    ############################
    # (K) 최종 결과 dict
    ############################
    result = {
       # 흥미 합산값
       'interest_S': interest_S, 'interest_E': interest_E, 'interest_C': interest_C,
       'interest_R': interest_R, 'interest_I': interest_I, 'interest_A': interest_A,
       'interest_S_T': interest_S_T, 'interest_E_T': interest_E_T,
       'interest_C_T': interest_C_T, 'interest_R_T': interest_R_T,
       'interest_I_T': interest_I_T, 'interest_A_T': interest_A_T,

       # 적성
       'apti_S': apti_S,'apti_E': apti_E,'apti_C': apti_C,
       'apti_R': apti_R,'apti_I': apti_I,'apti_A': apti_A,
       'apti_S_T': apti_S_T,'apti_E_T': apti_E_T,'apti_C_T': apti_C_T,
       'apti_R_T': apti_R_T,'apti_I_T': apti_I_T,'apti_A_T': apti_A_T,

       # 선호직업
       'job_S': job_S,'job_E': job_E,'job_C': job_C,
       'job_R': job_R,'job_I': job_I,'job_A': job_A,
       'job_S_T': job_S_T,'job_E_T': job_E_T,'job_C_T': job_C_T,
       'job_R_T': job_R_T,'job_I_T': job_I_T,'job_A_T': job_A_T,

       # 직업환경
       'env_D': env_D,'env_I': env_I,'env_P': env_P,'env_T': env_T,
       'env_D_T': env_D_T,'env_I_T': env_I_T,'env_P_T': env_P_T,'env_T_T': env_T_T,

       # factor
       'factor_S': fS,'factor_E': fE,'factor_C': fC,
       'factor_R': fR,'factor_I': fI,'factor_A': fA,
    }

    # 1~6위
    for i in range(1,7):
        result[f'{i}순위'] = rank_codes[i]
        result[f'{i}순위_점수'] = rank_points[i]

    # 흥미/적성/직업 순위(간단 정렬)
    for i in range(1,7):
        result[f'Int{i}순위'] = int_lbls[i-1]
        result[f'Int{i}순위_점수'] = int_vals[i-1]
        result[f'Apti{i}순위'] = apti_lbls[i-1]
        result[f'Apti{i}순위_점수'] = apti_vals[i-1]
        result[f'jobpre{i}순위'] = job_lbls[i-1]
        result[f'jobpre{i}순위_점수'] = job_vals[i-1]
        result[f'{i}순위_매칭개수'] = matching_counts[i]

    result['반응적합도_비율'] = reaction_fit_ratio
    result['반응적합'] = response_fit
    result['변별도'] = disc

    # Code1, Code2
    result['Code1'] = code1
    result['Code2'] = code2

    # 1,2순위 코드 넘버
    result['1순위 코드 넘버'] = fc_num
    result['2순위 코드 넘버'] = sc_num

    # 직업환경 코드
    result['직환code1'] = jobenv_code1
    result['직환code2'] = jobenv_code2
    result['직업환경 Code'] = jobenv_final

    # 학과, 직업 리스트
    result['major'] = major1
    result['joblist'] = joblist1
    result['major2'] = major2
    result['joblist2'] = joblist2

    result['final_code'] = final_code
    result['description'] = description_1

    return result


########################################
# 2) 메인 함수: iterrows() + 위 로직
########################################
def cet_process(input_path, score_path, output_path):
    """
    행 단위로 CET 초등용 로직을 수행한 뒤, 최종 결과를 output_path에 저장.
    """
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # (1) 입력 로딩
    df_input = pd.read_excel(input_path)

    # (2) 규준표 로딩
    interest_S_score = pd.read_excel(score_path, sheet_name="interest_S")
    interest_E_score = pd.read_excel(score_path, sheet_name="interest_E")
    interest_C_score = pd.read_excel(score_path, sheet_name="interest_C")
    interest_R_score = pd.read_excel(score_path, sheet_name="interest_R")
    interest_I_score = pd.read_excel(score_path, sheet_name="interest_I")
    interest_A_score = pd.read_excel(score_path, sheet_name="interest_A")

    apti_S_score = pd.read_excel(score_path, sheet_name="apti_S")
    apti_E_score = pd.read_excel(score_path, sheet_name="apti_E")
    apti_C_score = pd.read_excel(score_path, sheet_name="apti_C")
    apti_R_score = pd.read_excel(score_path, sheet_name="apti_R")
    apti_I_score = pd.read_excel(score_path, sheet_name="apti_I")
    apti_A_score = pd.read_excel(score_path, sheet_name="apti_A")

    job_S_score = pd.read_excel(score_path, sheet_name="job_S")
    job_E_score = pd.read_excel(score_path, sheet_name="job_E")
    job_C_score = pd.read_excel(score_path, sheet_name="job_C")
    job_R_score = pd.read_excel(score_path, sheet_name="job_R")
    job_I_score = pd.read_excel(score_path, sheet_name="job_I")
    job_A_score = pd.read_excel(score_path, sheet_name="job_A")

    env_p_score = pd.read_excel(score_path, sheet_name="env_P")
    env_t_score = pd.read_excel(score_path, sheet_name="env_T")
    env_d_score = pd.read_excel(score_path, sheet_name="env_D")
    env_i_score = pd.read_excel(score_path, sheet_name="env_I")

    major_df = pd.read_excel(score_path, sheet_name="major")
    joblist_df= pd.read_excel(score_path, sheet_name="joblist")
    description_df  = pd.read_excel(score_path, sheet_name="description")

    # (2) 각 행 처리
    results=[]
    for idx, row in df_input.iterrows():
        row_result = process_single_person(
            row,
            # 규준표들...
            interest_S_score, interest_E_score, interest_C_score, interest_R_score, interest_I_score, interest_A_score,
            apti_S_score, apti_E_score, apti_C_score, apti_R_score, apti_I_score, apti_A_score,
            job_S_score, job_E_score, job_C_score, job_R_score, job_I_score, job_A_score,
            env_p_score, env_t_score, env_d_score, env_i_score,
            major_df, joblist_df, description_df
        )
        
        results.append(row_result)

    # (3) 결과 -> DF -> 저장
    df_output = pd.DataFrame(results)


    df_input_reset = df_input.reset_index(drop=True)
    df_output_reset = df_output.reset_index(drop=True)

    # (4) 좌: df_input, 우: df_output, 즉 컬럼을 가로로 이어붙이기
    df_merged = pd.concat([df_input_reset, df_output_reset], axis=1)

    # (5) 최종 Excel 저장
    df_merged.to_excel(output_path, index=False)
    

    print(f"[행 단위 CET 초등용] 처리 완료 -> {output_path}")
