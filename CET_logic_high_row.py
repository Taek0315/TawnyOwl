import pandas as pd
import numpy as np
import warnings


def interest_encode(val, code):
    """
    흥미 문항에 대해, (문항값: 1/2/3/기타) -> (인코딩: IntX-Y) 처리를 행 단위로 하기 위한 함수.
    code = 1이면 val == 1일 때 1, 아니면 0
    code = 2이면 val == 2일 때 1, 아니면 0
    code = 3이면 val == 3일 때 1, 아니면 0
    code = 4이면 val이 1/2/3이 아닐 때 1, 그렇지 않으면 0
    """
    if code == 1:
        return 1 if val == 1 else 0
    elif code == 2:
        return 1 if val == 2 else 0
    elif code == 3:
        return 1 if val == 3 else 0
    elif code == 4:
        return 1 if val not in [1,2,3] else 0
    else:
        return 0


def tie_break_top(sorted_items, distance_map):
    """
    행 단위로 factor 순위를 구했을 때,
    1위 동점자가 있는 경우의 처리 로직을 그대로 옮긴 함수.
    sorted_items는 [ (코드, 점수), (코드, 점수), ... ] 형태 (내림차순).
    distance_map은 (코드1, 코드2) -> 거리 를 나타내는 딕셔너리.
    """
    def get_distance(a, b):
        return distance_map[(a,b)]

    # 1) 1위 점수를 확인하고, 동점인 코드들을 모음
    top_score = sorted_items[0][1]
    first_place_codes = [code for code, sc in sorted_items if sc == top_score]
    
    # 동점자가 1개면 그대로 반환
    if len(first_place_codes) == 1:
        return sorted_items
    
    # 6위 코드 (가장 점수 낮은 코드)
    code6 = sorted_items[-1][0]

    # -- 동점자 2개 --
    if len(first_place_codes) == 2:
        c1, c2 = first_place_codes
        dist1 = get_distance(c1, code6)
        dist2 = get_distance(c2, code6)
        
        if dist1 > dist2:
            new_order_top = [c1, c2]
        elif dist1 < dist2:
            new_order_top = [c2, c1]
        else:
            # dist 같으면 3위 코드와의 거리로 재판단
            code3 = sorted_items[2][0]
            d13 = get_distance(c1, code3)
            d23 = get_distance(c2, code3)
            if d13 < d23:
                new_order_top = [c1, c2]
            elif d13 > d23:
                new_order_top = [c2, c1]
            else:
                # 여전히 같으면 그대로
                new_order_top = [c1, c2]

        re_ordered = []
        used = set()
        for c in new_order_top:
            sc = [s for (cd, s) in sorted_items if cd == c][0]
            re_ordered.append((c, sc))
            used.add(c)
        for (cd, s) in sorted_items:
            if cd not in used:
                re_ordered.append((cd, s))
        return re_ordered
    
    # -- 동점자 3개 --
    elif len(first_place_codes) == 3:
        c1, c2, c3 = first_place_codes
        dist_c1 = get_distance(c1, code6)
        dist_c2 = get_distance(c2, code6)
        dist_c3 = get_distance(c3, code6)
        
        max_dist = max(dist_c1, dist_c2, dist_c3)
        candidates = []
        for c, d in [(c1, dist_c1), (c2, dist_c2), (c3, dist_c3)]:
            if d == max_dist:
                candidates.append(c)
        
        if len(candidates) == 1:
            first = candidates[0]
            others = [x for x in [c1, c2, c3] if x != first]
            # first와의 거리로 2,3위 결정
            d_oth = [(c, get_distance(c, first)) for c in others]
            d_oth.sort(key=lambda x: x[1])
            second = d_oth[0][0]
            third  = d_oth[1][0]
        else:
            # 후보가 둘 이상
            leftover = [x for x in [c1, c2, c3] if x not in candidates]
            if len(leftover) == 1:
                mid_code = leftover[0]
                dist_list = [(cd, get_distance(cd, mid_code)) for cd in candidates]
                dist_list.sort(key=lambda x: x[1])
                first = dist_list[0][0]
                second = dist_list[1][0]
                third = mid_code
            else:
                # 3개 전부 같은 거리 등 특수 케이스
                candidates.sort()
                first, second, third = candidates[0], candidates[1], candidates[2]
        
        re_ordered = []
        used = {first, second, third}
        sc_first  = [s for (cd, s) in sorted_items if cd == first][0]
        sc_second = [s for (cd, s) in sorted_items if cd == second][0]
        sc_third  = [s for (cd, s) in sorted_items if cd == third][0]
        re_ordered.append((first, sc_first))
        re_ordered.append((second, sc_second))
        re_ordered.append((third, sc_third))
        for (cd, s) in sorted_items:
            if cd not in used:
                re_ordered.append((cd, s))
        return re_ordered

    # -- 동점자 4명 이상 --
    else:
        return sorted_items


def get_sorted_labels_scores(factor_dict, distance_map):
    """
    factor_dict = {'S': x, 'E': y, ...} 형태
    반환: (labels, scores) - tie_break_top 적용 (1위 동점 처리)
    """
    # 점수 내림차순
    sorted_items = sorted(factor_dict.items(), key=lambda x: (-x[1], x[0]))
    # 1위 동점 처리
    sorted_items = tie_break_top(sorted_items, distance_map)
    # 분리
    sorted_labels = [x[0] for x in sorted_items]
    sorted_scores = [x[1] for x in sorted_items]
    return sorted_labels, sorted_scores


def simple_rank(scores_list, labels):
    """
    단순 ‘점수 내림차순’ 정렬 후 (labels, scores) 반환 (동점자 특별 처리 없이).
    scores_list = [ val1, val2, ... ] 에 해당하는 factor별 점수
    labels = ['S','E','C','R','I','A'] 등
    """
    arr = np.array(scores_list)
    sorted_indices = np.argsort(-arr)  # 내림차순
    sorted_lbls = [labels[i] for i in sorted_indices]
    sorted_vals = [arr[i] for i in sorted_indices]
    return (sorted_lbls, sorted_vals)


def process_single_person(row_data,
                          interest_score, person_score, apti_score, job_score,
                          env_p_t_score, env_d_i_score,
                          major_df, joblist_df, description_df):
    """
    하나의 응답(row_data)을 받아, CET 전체 로직을 수행한 결과를 dict로 반환한다.
    row_data: pd.Series 또는 dict처럼 참조 가능해야 함 (ex: row_data['Int_1'])
    """
    # --------------------
    # (A) 흥미(Interest) 항목 처리
    # --------------------
    # 24개 문항 각각 1/2/3/4 인코딩
    # 예: interest_S에 쓰이는 패턴들 (Int1-1, Int4-4, Int5-3, ...)
    interest_S_indices = [(1,1),(4,4),(5,3),(6,2),(7,1),(10,4),(11,3),(12,2),(13,1),
                          (16,4),(17,3),(18,2),(19,1),(22,4),(23,3),(24,2)]
    interest_E_indices = [(1,2),(2,1),(5,4),(6,3),(7,2),(8,1),(11,4),(12,3),(13,2),
                          (14,1),(17,4),(18,3),(19,2),(20,1),(23,4),(24,3)]
    interest_C_indices = [(1,3),(2,2),(3,1),(6,4),(7,3),(8,2),(9,1),(12,4),(13,3),
                          (14,2),(15,1),(18,4),(19,3),(20,2),(21,1),(24,4)]
    interest_R_indices = [(1,4),(2,3),(3,2),(4,1),(7,4),(8,3),(9,2),(10,1),(13,4),
                          (14,3),(15,2),(16,1),(19,4),(20,3),(21,2),(22,1)]
    interest_I_indices = [(2,4),(3,3),(4,2),(5,1),(8,4),(9,3),(10,2),(11,1),(14,4),
                          (15,3),(16,2),(17,1),(20,4),(21,3),(22,2),(23,1)]
    interest_A_indices = [(3,4),(4,3),(5,2),(6,1),(9,4),(10,3),(11,2),(12,1),(15,4),
                          (16,3),(17,2),(18,1),(21,4),(22,3),(23,2),(24,1)]

    def sum_interest(indices):
        return sum( interest_encode(row_data[f'Int_{i}'], code) for (i, code) in indices )

    interest_S = sum_interest(interest_S_indices)
    interest_E = sum_interest(interest_E_indices)
    interest_C = sum_interest(interest_C_indices)
    interest_R = sum_interest(interest_R_indices)
    interest_I = sum_interest(interest_I_indices)
    interest_A = sum_interest(interest_A_indices)

    # --------------------
    # (B) 성격(Person) 항목 처리 (30문항)
    # df['person_S'] = df[['P1','P7','P13','P19','P25']].sum(axis=1)
    # ...
    person_S = (row_data['P_1'] + row_data['P_7'] + row_data['P_13'] + row_data['P_19'] + row_data['P_25'])
    person_E = (row_data['P_2'] + row_data['P_8'] + row_data['P_14'] + row_data['P_20'] + row_data['P_26'])
    person_C = (row_data['P_3'] + row_data['P_9'] + row_data['P_15'] + row_data['P_21'] + row_data['P_27'])
    person_R = (row_data['P_4'] + row_data['P_10']+ row_data['P_16']+ row_data['P_22']+ row_data['P_28'])
    person_I = (row_data['P_5'] + row_data['P_11']+ row_data['P_17']+ row_data['P_23']+ row_data['P_29'])
    person_A = (row_data['P_6'] + row_data['P_12']+ row_data['P_18']+ row_data['P_24']+ row_data['P_30'])

    # --------------------
    # (C) 적성(Apt) 60문항
    # 예: df['apti_S'] = sum of Apt1, Apt7, ..., Apt55
    # ...
    def sum_columns(prefix, indices):
        return sum(row_data[f'{prefix}{i}'] for i in indices)

    apti_S = sum_columns('Apt_', [1,7,13,19,25,31,37,43,49,55])
    apti_E = sum_columns('Apt_', [2,8,14,20,26,32,38,44,50,56])
    apti_C = sum_columns('Apt_', [3,9,15,21,27,33,39,45,51,57])
    apti_R = sum_columns('Apt_', [4,10,16,22,28,34,40,46,52,58])
    apti_I = sum_columns('Apt_', [5,11,17,23,29,35,41,47,53,59])
    apti_A = sum_columns('Apt_', [6,12,18,24,30,36,42,48,54,60])

    # --------------------
    # (D) 직업환경(env) 30문항 (EJ1 ~ EJ30)
    # df['env_D'] = sum of EJ2-2, EJ4-1, ...
    # 여기서 EJn-1, EJn-2는 "EJ_n == 1" -> 1 or "EJ_n == 2" -> 1 인 식.
    # R 코드에서처럼 하나씩 지정해도 되나, 수작업이 매우 많음.
    # 여기서는 그대로 수작업하겠습니다:
    # env_D => EJ2-2, EJ4-1, ...
    # "EJ{n}-1" = (row_data['EJ_{n}'] == 1) ... "EJ{n}-2" = (row_data['EJ_{n}'] == 2)
    # df['env_D'] = EJ2-2 + EJ4-1 + EJ6-1 + EJ8-1 + EJ10-1 + EJ12-2 + EJ14-1 + EJ16-2 + EJ18-2 + EJ20-1 + EJ22-1 + EJ24-2 + EJ26-2 + EJ28-1 + EJ30-1
    def EJ1(n): return 1 if row_data[f'EJ_{n}'] == 1 else 0
    def EJ2(n): return 1 if row_data[f'EJ_{n}'] == 2 else 0

    env_D = (EJ2(2)+EJ1(4)+EJ1(6)+EJ1(8)+EJ1(10)+EJ2(12)+EJ1(14)+EJ2(16)+
             EJ2(18)+EJ1(20)+EJ1(22)+EJ2(24)+EJ2(26)+EJ1(28)+EJ1(30))

    env_I = (EJ1(2)+EJ2(4)+EJ2(6)+EJ2(8)+EJ2(10)+EJ1(12)+EJ2(14)+EJ1(16)+
             EJ1(18)+EJ2(20)+EJ2(22)+EJ1(24)+EJ1(26)+EJ2(28)+EJ2(30))

    env_P = (EJ1(1)+EJ1(3)+EJ2(5)+EJ1(7)+EJ1(9)+EJ2(11)+EJ2(13)+EJ1(15)+
             EJ2(17)+EJ1(19)+EJ2(21)+EJ1(23)+EJ2(25)+EJ1(27)+EJ2(29))

    env_T = (EJ2(1)+EJ2(3)+EJ1(5)+EJ2(7)+EJ2(9)+EJ1(11)+EJ1(13)+EJ2(15)+
             EJ1(17)+EJ2(19)+EJ1(21)+EJ2(23)+EJ1(25)+EJ2(27)+EJ1(29))

    # --------------------
    # (E) 선호직업 job_x
    # df['job_S'] = sum of pre1,7,13,...,55
    # ...
    job_S = sum_columns('pre_', [1,7,13,19,25,31,37,43,49,55])
    job_E = sum_columns('pre_', [2,8,14,20,26,32,38,44,50,56])
    job_C = sum_columns('pre_', [3,9,15,21,27,33,39,45,51,57])
    job_R = sum_columns('pre_', [4,10,16,22,28,34,40,46,52,58])
    job_I = sum_columns('pre_', [5,11,17,23,29,35,41,47,53,59])
    job_A = sum_columns('pre_', [6,12,18,24,30,36,42,48,54,60])

    # --------------------
    # (F) T점수 매핑
    # 먼저 score_df들을 dict로 바꿔서 lookup
    # 예: interest_score['score'], interest_score['T_score_m'] -> dict
    # 여기서는 process_single_person 안에서 직접 만들기엔 비효율적이므로
    # 보통은 미리 밖에서 만들고 인자로 넘기는 편임.
    # 여기서는 interest_score, person_score ... DataFrame을 직접 map(dict(zip(...))) 하겠습니다.

    # 성별
    gender = row_data['성별']
    # 혹은 남=1, 여=2 로 가정

    # interest
    map_int_m = dict(zip(interest_score['score'], interest_score['T_score_m']))
    map_int_f = dict(zip(interest_score['score'], interest_score['T_score_f']))

    def lookup_interest(val):
        if pd.isna(gender):
            return None
        if gender == 1:
            return map_int_m.get(val, None)
        else:
            return map_int_f.get(val, None)

    interest_S_T = lookup_interest(interest_S)
    interest_E_T = lookup_interest(interest_E)
    interest_C_T = lookup_interest(interest_C)
    interest_R_T = lookup_interest(interest_R)
    interest_I_T = lookup_interest(interest_I)
    interest_A_T = lookup_interest(interest_A)

    # person
    map_per_m = dict(zip(person_score['score'], person_score['T_score_m']))
    map_per_f = dict(zip(person_score['score'], person_score['T_score_f']))
    def lookup_person(val):
        if pd.isna(gender):
            return None
        return map_per_m[val] if (gender == 1 and val in map_per_m) else \
               map_per_f[val] if (gender != 1 and val in map_per_f) else None

    person_S_T = lookup_person(person_S)
    person_E_T = lookup_person(person_E)
    person_C_T = lookup_person(person_C)
    person_R_T = lookup_person(person_R)
    person_I_T = lookup_person(person_I)
    person_A_T = lookup_person(person_A)

    # apti
    map_apti_m = dict(zip(apti_score['score'], apti_score['T_score_m']))
    map_apti_f = dict(zip(apti_score['score'], apti_score['T_score_f']))
    def lookup_apti(val):
        if pd.isna(gender):
            return None
        return map_apti_m[val] if (gender == 1 and val in map_apti_m) else \
               map_apti_f[val] if (gender != 1 and val in map_apti_f) else None

    apti_S_T = lookup_apti(apti_S)
    apti_E_T = lookup_apti(apti_E)
    apti_C_T = lookup_apti(apti_C)
    apti_R_T = lookup_apti(apti_R)
    apti_I_T = lookup_apti(apti_I)
    apti_A_T = lookup_apti(apti_A)

    # job
    map_job_m = dict(zip(job_score['score'], job_score['T_score_m']))
    map_job_f = dict(zip(job_score['score'], job_score['T_score_f']))
    def lookup_job(val):
        if pd.isna(gender):
            return None
        return map_job_m[val] if (gender == 1 and val in map_job_m) else \
               map_job_f[val] if (gender != 1 and val in map_job_f) else None

    job_S_T = lookup_job(job_S)
    job_E_T = lookup_job(job_E)
    job_C_T = lookup_job(job_C)
    job_R_T = lookup_job(job_R)
    job_I_T = lookup_job(job_I)
    job_A_T = lookup_job(job_A)

    # env
    map_env_p_t_m = dict(zip(env_p_t_score['score'], env_p_t_score['T_score_m']))
    map_env_p_t_f = dict(zip(env_p_t_score['score'], env_p_t_score['T_score_f']))

    map_env_d_i_m = dict(zip(env_d_i_score['score'], env_d_i_score['T_score_m']))
    map_env_d_i_f = dict(zip(env_d_i_score['score'], env_d_i_score['T_score_f']))

    def lookup_env_p_t(val):
        if pd.isna(gender):
            return None
        return map_env_p_t_m[val] if (gender == 1 and val in map_env_p_t_m) else \
               map_env_p_t_f[val] if (gender != 1 and val in map_env_p_t_f) else None

    def lookup_env_d_i(val):
        if pd.isna(gender):
            return None
        # D,I 는 env_d_i_score
        return map_env_d_i_m[val] if (gender == 1 and val in map_env_d_i_m) else \
               map_env_d_i_f[val] if (gender != 1 and val in map_env_d_i_f) else None

    env_D_T = lookup_env_d_i(env_D)
    env_I_T = lookup_env_d_i(env_I)
    env_P_T = lookup_env_p_t(env_P)
    env_T_T = lookup_env_p_t(env_T)

    # --------------------
    # (G) 최종 factor 계산
    # factor_S = ((interest_S_T*2) + person_S_T + (apti_S_T*2) + job_S_T ) / 6
    # ...
    def safe_mul(a,b):
        return a*b if (a is not None and b is not None) else None
    def safe_add(*vals):
        """None이 섞여 있으면 계산 불가 -> None 반환"""
        if any(v is None for v in vals):
            return None
        return sum(vals)

    def safe_div(val, d):
        if val is None: return None
        return val/d

    factor_S = None
    if all(x is not None for x in [interest_S_T, person_S_T, apti_S_T, job_S_T]):
        factor_S = (interest_S_T*2 + person_S_T + apti_S_T*2 + job_S_T)/6
    factor_E = None
    if all(x is not None for x in [interest_E_T, person_E_T, apti_E_T, job_E_T]):
        factor_E = (interest_E_T*2 + person_E_T + apti_E_T*2 + job_E_T)/6
    factor_C = None
    if all(x is not None for x in [interest_C_T, person_C_T, apti_C_T, job_C_T]):
        factor_C = (interest_C_T*2 + person_C_T + apti_C_T*2 + job_C_T)/6
    factor_R = None
    if all(x is not None for x in [interest_R_T, person_R_T, apti_R_T, job_R_T]):
        factor_R = (interest_R_T*2 + person_R_T + apti_R_T*2 + job_R_T)/6
    factor_I = None
    if all(x is not None for x in [interest_I_T, person_I_T, apti_I_T, job_I_T]):
        factor_I = (interest_I_T*2 + person_I_T + apti_I_T*2 + job_I_T)/6
    factor_A = None
    if all(x is not None for x in [interest_A_T, person_A_T, apti_A_T, job_A_T]):
        factor_A = (interest_A_T*2 + person_A_T + apti_A_T*2 + job_A_T)/6

    # 6개 factor에 대해 S/E/C/R/I/A 라벨링
    factor_vals = {
        'S': factor_S if factor_S is not None else 0,
        'E': factor_E if factor_E is not None else 0,
        'C': factor_C if factor_C is not None else 0,
        'R': factor_R if factor_R is not None else 0,
        'I': factor_I if factor_I is not None else 0,
        'A': factor_A if factor_A is not None else 0,
    }

    # (H) 순위 결정 (1~6위)
    #    tie-break_top 로직 필요
    # distance_map
    distance_map = {
        ('S','S'):0, ('S','E'):1, ('S','C'):2, ('S','R'):3, ('S','I'):2, ('S','A'):1,
        ('E','S'):1, ('E','E'):0, ('E','C'):1, ('E','R'):2, ('E','I'):3, ('E','A'):2,
        ('C','S'):2, ('C','E'):1, ('C','C'):0, ('C','R'):1, ('C','I'):2, ('C','A'):3,
        ('R','S'):3, ('R','E'):2, ('R','C'):1, ('R','R'):0, ('R','I'):1, ('R','A'):2,
        ('I','S'):2, ('I','E'):3, ('I','C'):2, ('I','R'):1, ('I','I'):0, ('I','A'):1,
        ('A','S'):1, ('A','E'):2, ('A','C'):3, ('A','R'):2, ('A','I'):1, ('A','A'):0,
    }
    sorted_labels, sorted_scores = get_sorted_labels_scores(factor_vals, distance_map)

    # 1~6위
    rank_codes = {}
    rank_points = {}
    for r in range(6):
        rank_codes[r+1] = sorted_labels[r]
        rank_points[r+1] = round(sorted_scores[r], 2)  # 소수점 2자리 정도

    # (I) 흥미/성격/적성/직업 선호 각각 6가지 요인 점수에 대해 단순 순위
    #    (동점 특별 처리 없이) 
    # 예) interest_columns = [interest_S_T, interest_E_T, ...]
    # 여기서는 row 단위로 interest_S_T, ... 를 모아 simple_rank
    def none_to_zero(x):
        return x if x is not None else 0

    # 흥미
    interest_list = [none_to_zero(interest_S_T),
                     none_to_zero(interest_E_T),
                     none_to_zero(interest_C_T),
                     none_to_zero(interest_R_T),
                     none_to_zero(interest_I_T),
                     none_to_zero(interest_A_T)]
    lbl_interest = ['S','E','C','R','I','A']
    int_lbls, int_scores = simple_rank(interest_list, lbl_interest)

    # 성격
    person_list = [none_to_zero(person_S_T),
                   none_to_zero(person_E_T),
                   none_to_zero(person_C_T),
                   none_to_zero(person_R_T),
                   none_to_zero(person_I_T),
                   none_to_zero(person_A_T)]
    per_lbls, per_scores = simple_rank(person_list, lbl_interest)

    # 적성
    apti_list = [none_to_zero(apti_S_T),
                 none_to_zero(apti_E_T),
                 none_to_zero(apti_C_T),
                 none_to_zero(apti_R_T),
                 none_to_zero(apti_I_T),
                 none_to_zero(apti_A_T)]
    apti_lbls, apti_scores = simple_rank(apti_list, lbl_interest)

    # 직업선호
    job_list = [none_to_zero(job_S_T),
                none_to_zero(job_E_T),
                none_to_zero(job_C_T),
                none_to_zero(job_R_T),
                none_to_zero(job_I_T),
                none_to_zero(job_A_T)]
    job_lbls, job_scores = simple_rank(job_list, lbl_interest)

    # (J) 1~6위별 매칭 개수(해당 순위의 코드 vs 각 요인별 동일 순위 코드)
    # 예: '1순위'와 'Int1순위'가 같으면 +1, 'Per1순위'가 같으면 +1 ...
    matching_counts = {}
    for r in range(1,7):
        code_at_r = rank_codes[r]  # 예: '1순위' = rank_codes[1]
        # interest에서 r번째 = int_lbls[r-1]
        # person에서 r번째 = per_lbls[r-1]
        # apti에서 r번째 = apti_lbls[r-1]
        # job  에서 r번째 = job_lbls[r-1]
        mc = 0
        if code_at_r == int_lbls[r-1]:
            mc += 1
        if code_at_r == per_lbls[r-1]:
            mc += 1
        if code_at_r == apti_lbls[r-1]:
            mc += 1
        if code_at_r == job_lbls[r-1]:
            mc += 1
        matching_counts[r] = mc

    # (K) 반응적합도_비율
    # df['반응적합도_비율'] = round(((df['1순위_매칭개수']+...+df['6순위_매칭개수']) / 24)*100, 2)
    total_matches = sum(matching_counts.values())
    reaction_fit_ratio = round((total_matches/24)*100, 2)

    # (L) 반응적합 (원 코드에서 df_count.notnull().sum(axis=1) ... 204개 문항 중 유효응답 비율)
    # 여기서는 row 단위로 ‘Apt_1’부터 ‘pre_60’, ‘P_n’, ‘Int_n’, ‘EJ_n’ 등등을 체크해야 함.
    # 문항 총 개수 = 204(고등 버전), 결측 아닌 것 세어서 %화
    # 간단히 204개 모두 존재한다고 가정하거나, 실제로는 row_data에서 결측 체크
    answered = 0
    question_cols = []
    # Int_1..Int_24 (24)
    for i in range(1,25):
        question_cols.append(f'Int_{i}')
    # P_1..P_30 (30)
    for i in range(1,31):
        question_cols.append(f'P_{i}')
    # Apt_1..Apt_60 (60)
    for i in range(1,61):
        question_cols.append(f'Apt_{i}')
    # pre_1..pre_60 (60)
    for i in range(1,61):
        question_cols.append(f'pre_{i}')
    # EJ_1..EJ_30 (30)
    for i in range(1,31):
        question_cols.append(f'EJ_{i}')

    # 총 24+30+60+60+30=204
    for col in question_cols:
        val = row_data[col]
        if not pd.isna(val):
            answered += 1
    response_fit = round((answered / 204)*100, 2)

    # (M) 변별도 = 1순위 점수 - ((2순위_점수 + 4순위_점수)/2)
    # 1순위 점수 = rank_points[1], 2순위 점수 = rank_points[2], ...
    discriminant = None
    if (1 in rank_points) and (2 in rank_points) and (4 in rank_points):
        discriminant = round(rank_points[1] - ((rank_points[2]+rank_points[4])/2), 2)

    # (N) Code1 = 기본은 "1순위+2순위", but
    #  - if response_fit <=70 -> 분류불능a
    #  - elif discriminant <1  -> 분류불능b
    #  - elif factor_S..factor_A 전부 <30 -> 분류불능c
    #  - elif gender is na -> None
    #  - else -> 1순위+2순위
    def all_factor_below_30(*vals):
        return all((v is not None and v<30) for v in vals)

    code1 = None
    if response_fit <= 70:
        code1 = '분류불능a'
    elif discriminant is not None and discriminant <1:
        code1 = '분류불능b'
    elif all_factor_below_30(factor_S,factor_E,factor_C,factor_R,factor_I,factor_A):
        code1 = '분류불능c'
    elif pd.isna(gender):
        code1 = None
    else:
        # 1순위+2순위
        code1 = rank_codes[1] + rank_codes[2]

    # (O) Code2
    # = 분류불능 if code1 in [분류불능a,b,c]
    # else if (2순위_점수 - 3순위_점수) >=3 => 2순위+1순위
    # else => 1순위+3순위
    if code1 in ['분류불능a','분류불능b','분류불능c']:
        code2 = '분류불능'
    else:
        # 2순위 점수, 3순위 점수
        sc2 = rank_points[2] if 2 in rank_points else 0
        sc3 = rank_points[3] if 3 in rank_points else 0
        if (sc2 - sc3) >= 3:
            code2 = rank_codes[2] + rank_codes[1]
        else:
            code2 = rank_codes[1] + rank_codes[3]

    # (P) 1순위/2순위 코드 넘버
    priority_mapping = {'S':1,'E':2,'C':3,'R':4,'I':5,'A':6}
    first_code_num = priority_mapping.get(rank_codes[1], None)
    second_code_num = priority_mapping.get(rank_codes[2], None)

    # (Q) 직업환경 Code
    # conditions = [
    #   df['env_P_T'] > df['env_T_T'], # => 'P'
    #   df['env_P_T'] < df['env_T_T'], # => 'T'
    #   (df['env_P_T'] == df['env_T_T']) & df['1순위 코드 넘버'].isin([1,2,3]) => 'P'
    # default => 'T'
    # conditions2 = ...
    # 최종 = 직환code2 + 직환code1
    # 만약 (env_D_T <30)&(env_I_T<30)&(env_P_T<30)&(env_T_T<30) => '분류불능d'
    jobenv_code1 = None
    if env_P_T is None or env_T_T is None:
        pass
    else:
        if env_P_T > env_T_T:
            jobenv_code1 = 'P'
        elif env_P_T < env_T_T:
            jobenv_code1 = 'T'
        else:
            # env_P_T == env_T_T
            if first_code_num in [1,2,3]:
                jobenv_code1 = 'P'
            else:
                jobenv_code1 = 'T'

    jobenv_code2 = None
    if env_D_T is None or env_I_T is None:
        pass
    else:
        if env_D_T > env_I_T:
            jobenv_code2 = 'D'
        elif env_D_T < env_I_T:
            jobenv_code2 = 'I'
        else:
            # env_D_T == env_I_T
            if first_code_num in [1,2,3]:
                jobenv_code2 = 'D'
            else:
                jobenv_code2 = 'I'

    jobenv_final = None
    if env_D_T is not None and env_I_T is not None and env_P_T is not None and env_T_T is not None:
        if (env_D_T<30) and (env_I_T<30) and (env_P_T<30) and (env_T_T<30):
            jobenv_final = '분류불능d'
        else:
            if jobenv_code1 and jobenv_code2:
                jobenv_final = jobenv_code2 + jobenv_code1
    
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

    # (R) 학과, 직업, 해석 리스트 매핑
    # major_df : score_path의 "major" 시트
    # joblist_df: "joblist"
    # code1 => major, joblist
    # code2 => major2, joblist2
    major_map = dict(zip(major_df['code'], major_df['major']))
    joblist_map = dict(zip(joblist_df['code'], joblist_df['job']))
    description_map = dict(zip(description_df['code'],description_df['description']))

    def map_major(code_val):
        if code_val in ['분류불능a','분류불능b','분류불능c','분류불능']:
            return '분류불능'
        return major_map.get(code_val, None)

    def map_joblist(code_val):
        if code_val in ['분류불능a','분류불능b','분류불능c','분류불능']:
            return '분류불능'
        return joblist_map.get(code_val, None)
    def map_description(code_val):
        return description_map.get(code_val, None)

    major_1 = map_major(code1)
    joblist_1 = map_joblist(code1)
    major_2 = map_major(code2)
    joblist_2 = map_joblist(code2)
    description_1 = map_description(final_code)

    # --------------------
    # 반환할 dict 구성
    # 실제로는 원래 df에 존재하던 모든 중간 컬럼들도 넣어주려면 수십 개가 될 수 있음
    # 여기서는 최종 결과 위주로만 넣겠습니다.
    # 필요하다면 중간에 계산했던 interest_S 등도 반환 가능.

    result = {
        # 원본 index나 식별자도 같이 넣어주세요(필요하면).
        '성별' : gender,

        'interest_S': interest_S,
        'interest_E': interest_E,
        'interest_C': interest_C,
        'interest_R': interest_R,
        'interest_I': interest_I,
        'interest_A': interest_A,

        'interest_S_T': interest_S_T,
        'interest_E_T': interest_E_T,
        'interest_C_T': interest_C_T,
        'interest_R_T': interest_R_T,
        'interest_I_T': interest_I_T,
        'interest_A_T': interest_A_T,

        'person_S': person_S,
        'person_E': person_E,
        'person_C': person_C,
        'person_R': person_R,
        'person_I': person_I,
        'person_A': person_A,

        'person_S_T': person_S_T,
        'person_E_T': person_E_T,
        'person_C_T': person_C_T,
        'person_R_T': person_R_T,
        'person_I_T': person_I_T,
        'person_A_T': person_A_T,

        'apti_S': apti_S,
        'apti_E': apti_E,
        'apti_C': apti_C,
        'apti_R': apti_R,
        'apti_I': apti_I,
        'apti_A': apti_A,

        'apti_S_T': apti_S_T,
        'apti_E_T': apti_E_T,
        'apti_C_T': apti_C_T,
        'apti_R_T': apti_R_T,
        'apti_I_T': apti_I_T,
        'apti_A_T': apti_A_T,

        'job_S': job_S,
        'job_E': job_E,
        'job_C': job_C,
        'job_R': job_R,
        'job_I': job_I,
        'job_A': job_A,

        'job_S_T': job_S_T,
        'job_E_T': job_E_T,
        'job_C_T': job_C_T,
        'job_R_T': job_R_T,
        'job_I_T': job_I_T,
        'job_A_T': job_A_T,

        'env_D': env_D,
        'env_I': env_I,
        'env_P': env_P,
        'env_T': env_T,
        'env_D_T': env_D_T,
        'env_I_T': env_I_T,
        'env_P_T': env_P_T,
        'env_T_T': env_T_T,

        'factor_S': round(factor_S,2) if factor_S is not None else None,
        'factor_E': round(factor_E,2) if factor_E is not None else None,
        'factor_C': round(factor_C,2) if factor_C is not None else None,
        'factor_R': round(factor_R,2) if factor_R is not None else None,
        'factor_I': round(factor_I,2) if factor_I is not None else None,
        'factor_A': round(factor_A,2) if factor_A is not None else None,
    }

    # 순위 (1~6위 코드, 점수)
    for r in range(1,7):
        result[f'{r}순위'] = rank_codes[r]
        result[f'{r}순위_점수'] = rank_points[r]

    # 흥미/성격/적성/직업선호 각 1~6위
    # int_lbls, int_scores
    for r in range(1,7):
        result[f'Int{r}순위'] = int_lbls[r-1]
        result[f'Int{r}순위_점수'] = round(int_scores[r-1],2)
        result[f'Per{r}순위'] = per_lbls[r-1]
        result[f'Per{r}순위_점수'] = round(per_scores[r-1],2)
        result[f'Apti{r}순위'] = apti_lbls[r-1]
        result[f'Apti{r}순위_점수'] = round(apti_scores[r-1],2)
        result[f'jobpre{r}순위'] = job_lbls[r-1]
        result[f'jobpre{r}순위_점수'] = round(job_scores[r-1],2)
        result[f'{r}순위_매칭개수'] = matching_counts[r]

    # 반응적합도_비율, 반응적합(=응답률), 변별도
    result['반응적합도_비율'] = reaction_fit_ratio
    result['반응적합'] = response_fit
    result['변별도'] = discriminant

    # Code1, Code2
    result['Code1'] = code1
    result['Code2'] = code2

    result['1순위 코드 넘버'] = first_code_num
    result['2순위 코드 넘버'] = second_code_num

    result['직환code1'] = jobenv_code1
    result['직환code2'] = jobenv_code2
    result['직업환경 Code'] = jobenv_final

    # major/joblist
    result['major']   = major_1
    result['joblist'] = joblist_1
    result['major2']  = major_2
    result['joblist2']= joblist_2
    result['final_code'] = final_code
    result['description'] = description_1

    

    return result


def cet_process(input_path, score_path, output_path):
    """
    CET 고등용 스크립트를 '행 단위'로 수행하는 예시 함수.

    1) input_path 에서 응답데이터(DataFrame) 읽기
    2) score_path에서 각 점수표(sheet) 읽기
    3) 각 행(응답자)에 대해 process_single_person() 호출
    4) 결과 누적 후, 최종 DF로 만들어 output_path에 저장
    """
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # (1) 입력 데이터
    df_input = pd.read_excel(input_path)

    # (2) 점수표 읽기
    interest_score = pd.read_excel(score_path, sheet_name="interest")
    person_score   = pd.read_excel(score_path, sheet_name="person")
    apti_score     = pd.read_excel(score_path, sheet_name="apti")
    job_score      = pd.read_excel(score_path, sheet_name="job")
    env_p_t_score  = pd.read_excel(score_path, sheet_name="env(p_t)")
    env_d_i_score  = pd.read_excel(score_path, sheet_name="env(d_i)")
    major_df       = pd.read_excel(score_path, sheet_name="major")
    joblist_df     = pd.read_excel(score_path, sheet_name="joblist")
    description_df  = pd.read_excel(score_path, sheet_name="description")

    # (3) 각 행을 순회하며 처리
    results = []
    for idx, row in df_input.iterrows():
        row_result = process_single_person(
            row,  # pd.Series
            interest_score, person_score, apti_score, job_score,
            env_p_t_score, env_d_i_score,
            major_df, joblist_df, description_df
        )
        
        results.append(row_result)

    # (4) 결과를 DataFrame으로 변환, 엑셀로 저장
    df_output = pd.DataFrame(results)

    df_input_reset = df_input.reset_index(drop=True)
    df_output_reset = df_output.reset_index(drop=True)

    # (5) 좌: df_input, 우: df_output, 즉 컬럼을 가로로 이어붙이기
    df_merged = pd.concat([df_input_reset, df_output_reset], axis=1)

    # (6) 최종 Excel 저장
    df_merged.to_excel(output_path, index=False)
    

    print(f"[행 단위 로직] 처리 완료 -> 결과 {output_path} 저장 완료")
