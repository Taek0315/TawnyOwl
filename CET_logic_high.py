import pandas as pd
import numpy as np
import warnings

def cet_process(input_path, score_path, output_path):
    """
    CET 고등용 Python 스크립트 로직을 함수로 감싼 함수입니다.
    인자:
        input_path  :  입력 데이터가 담긴 엑셀 파일 경로
        score_path  :  규준 데이터(점수표)가 있는 엑셀 파일 경로
        output_path :  최종 결과를 저장할 엑셀 파일 경로
    반환:
        없음(결과가 output_path로 저장됨)
    """

    # 경고 끄기
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # (1) 입력 데이터 불러오기
    df = pd.read_excel(input_path)

    # (4) 규준 데이터(점수표) 불러오기
    #     모든 sheet를 여기서 읽어옴
    interest_score    = pd.read_excel(score_path, sheet_name="interest")
    person_score      = pd.read_excel(score_path, sheet_name="person")
    apti_score        = pd.read_excel(score_path, sheet_name="apti")
    job_score         = pd.read_excel(score_path, sheet_name="job")
    env_p_t_score     = pd.read_excel(score_path, sheet_name="env(p_t)")
    env_d_i__score    = pd.read_excel(score_path, sheet_name="env(d_i)")
    major            = pd.read_excel(score_path, sheet_name="major")
    joblist          = pd.read_excel(score_path, sheet_name="joblist")

    # (2) 흥미, 성격, 적성, 직업환경, 선호직업 전처리 (원래 코드 그대로)
    # ------------------------------------------------------
    # 흥미 관련 문항 총 24개
    for i in range(1,25):
        df[f'Int{i}-1'] = np.where(df[f'Int_{i}'] == 1, 1, 0)
        df[f'Int{i}-2'] = np.where(df[f'Int_{i}'] == 2, 1, 0)
        df[f'Int{i}-3'] = np.where(df[f'Int_{i}'] == 3, 1, 0)
        df[f'Int{i}-4'] = np.where(~df[f'Int_{i}'].isin([1, 2, 3]), 1, 0)

    # 성격 데이터 및 직업환경 선호도
    for n in range(1,31):
        df[f'P{n}'] = df[f'P_{n}'].copy()
        df[f'EJ{n}-1'] = np.where(df[f'EJ_{n}'] == 1,1,0)
        df[f'EJ{n}-2'] = np.where(df[f'EJ_{n}'] == 2,1,0)

    # 적성 및 선호직업
    for w in range(1,61):
        df[f'Apt{w}'] = df[f'Apt_{w}'].copy()
        df[f'pre{w}'] = df[f'pre_{w}'].copy()
    # ------------------------------------------------------

    # (3) 6개 흥미요인, 성격요인, 적성요인, 직업환경요인, 선호직업요인 계산
    # ------------------------------------------------------
    # 흥미
    df['interest_S'] = df[['Int1-1','Int4-4','Int5-3','Int6-2','Int7-1','Int10-4','Int11-3','Int12-2','Int13-1','Int16-4','Int17-3','Int18-2','Int19-1','Int22-4','Int23-3','Int24-2']].sum(axis=1)
    df['interest_E'] = df[['Int1-2','Int2-1','Int5-4','Int6-3','Int7-2','Int8-1','Int11-4','Int12-3','Int13-2','Int14-1','Int17-4','Int18-3','Int19-2','Int20-1','Int23-4','Int24-3']].sum(axis=1)
    df['interest_C'] = df[['Int1-3','Int2-2','Int3-1','Int6-4','Int7-3','Int8-2','Int9-1','Int12-4','Int13-3','Int14-2','Int15-1','Int18-4','Int19-3','Int20-2','Int21-1','Int24-4']].sum(axis=1)
    df['interest_R'] = df[['Int1-4','Int2-3','Int3-2','Int4-1','Int7-4','Int8-3','Int9-2','Int10-1','Int13-4','Int14-3','Int15-2','Int16-1','Int19-4','Int20-3','Int21-2','Int22-1']].sum(axis=1)
    df['interest_I'] = df[['Int2-4','Int3-3','Int4-2','Int5-1','Int8-4','Int9-3','Int10-2','Int11-1','Int14-4','Int15-3','Int16-2','Int17-1','Int20-4','Int21-3','Int22-2','Int23-1']].sum(axis=1)
    df['interest_A'] = df[['Int3-4','Int4-3','Int5-2','Int6-1','Int9-4','Int10-3','Int11-2','Int12-1','Int15-4','Int16-3','Int17-2','Int18-1','Int21-4','Int22-3','Int23-2','Int24-1']].sum(axis=1)


    # 성격
    df['person_S'] = df[['P1','P7','P13','P19','P25']].sum(axis=1)
    df['person_E'] = df[['P2','P8','P14','P20','P26']].sum(axis=1)
    df['person_C'] = df[['P3','P9','P15','P21','P27']].sum(axis=1)
    df['person_R'] = df[['P4','P10','P16','P22','P28']].sum(axis=1)
    df['person_I'] = df[['P5','P11','P17','P23','P29']].sum(axis=1)
    df['person_A'] = df[['P6','P12','P18','P24','P30']].sum(axis=1)

    # 적성
    df['apti_S'] = df[['Apt1','Apt7','Apt13','Apt19','Apt25','Apt31','Apt37','Apt43','Apt49','Apt55']].sum(axis=1)
    df['apti_E'] = df[['Apt2','Apt8','Apt14','Apt20','Apt26','Apt32','Apt38','Apt44','Apt50','Apt56']].sum(axis=1)
    df['apti_C'] = df[['Apt3','Apt9','Apt15','Apt21','Apt27','Apt33','Apt39','Apt45','Apt51','Apt57']].sum(axis=1)
    df['apti_R'] = df[['Apt4','Apt10','Apt16','Apt22','Apt28','Apt34','Apt40','Apt46','Apt52','Apt58']].sum(axis=1)
    df['apti_I'] = df[['Apt5','Apt11','Apt17','Apt23','Apt29','Apt35','Apt41','Apt47','Apt53','Apt59']].sum(axis=1)
    df['apti_A'] = df[['Apt6','Apt12','Apt18','Apt24','Apt30','Apt36','Apt42','Apt48','Apt54','Apt60']].sum(axis=1)


    # 직업환경
    df['env_D'] = df[['EJ2-2','EJ4-1','EJ6-1','EJ8-1','EJ10-1','EJ12-2','EJ14-1','EJ16-2','EJ18-2','EJ20-1','EJ22-1','EJ24-2','EJ26-2','EJ28-1','EJ30-1']].sum(axis=1)
    df['env_I'] = df[['EJ2-1','EJ4-2','EJ6-2','EJ8-2','EJ10-2','EJ12-1','EJ14-2','EJ16-1','EJ18-1','EJ20-2','EJ22-2','EJ24-1','EJ26-1','EJ28-2','EJ30-2']].sum(axis=1)
    df['env_P'] = df[['EJ1-1','EJ3-1','EJ5-2','EJ7-1','EJ9-1','EJ11-2','EJ13-2','EJ15-1','EJ17-2','EJ19-1','EJ21-2','EJ23-1','EJ25-2','EJ27-1','EJ29-2']].sum(axis=1)
    df['env_T'] = df[['EJ1-2','EJ3-2','EJ5-1','EJ7-2','EJ9-2','EJ11-1','EJ13-1','EJ15-2','EJ17-1','EJ19-2','EJ21-1','EJ23-2','EJ25-1','EJ27-2','EJ29-1']].sum(axis=1)

    # 선호직업
    df['job_S'] = df[['pre1','pre7','pre13','pre19','pre25','pre31','pre37','pre43','pre49','pre55']].sum(axis=1)
    df['job_E'] = df[['pre2','pre8','pre14','pre20','pre26','pre32','pre38','pre44','pre50','pre56']].sum(axis=1)
    df['job_C'] = df[['pre3','pre9','pre15','pre21','pre27','pre33','pre39','pre45','pre51','pre57']].sum(axis=1)
    df['job_R'] = df[['pre4','pre10','pre16','pre22','pre28','pre34','pre40','pre46','pre52','pre58']].sum(axis=1)
    df['job_I'] = df[['pre5','pre11','pre17','pre23','pre29','pre35','pre41','pre47','pre53','pre59']].sum(axis=1)
    df['job_A'] = df[['pre6','pre12','pre18','pre24','pre30','pre36','pre42','pre48','pre54','pre60']].sum(axis=1)
    # ------------------------------------------------------

    
    # (5) T점수 매핑 (원본 로직 그대로)
    # ------------------------------------------------------
    df['interest_S_T'] = np.where(
        df['성별'] == 1,
        df['interest_S'].map(dict(zip(interest_score['score'], interest_score['T_score_m']))),
        df['interest_S'].map(dict(zip(interest_score['score'], interest_score['T_score_f'])))
    )
    df['interest_E_T'] = np.where(
        df['성별'] == 1,
        df['interest_E'].map(dict(zip(interest_score['score'], interest_score['T_score_m']))),
        df['interest_E'].map(dict(zip(interest_score['score'], interest_score['T_score_f'])))
    )
    df['interest_C_T'] = np.where(
        df['성별'] == 1,
        df['interest_C'].map(dict(zip(interest_score['score'], interest_score['T_score_m']))),
        df['interest_C'].map(dict(zip(interest_score['score'], interest_score['T_score_f'])))
    )
    df['interest_R_T'] = np.where(
        df['성별'] == 1,
        df['interest_R'].map(dict(zip(interest_score['score'], interest_score['T_score_m']))),
        df['interest_R'].map(dict(zip(interest_score['score'], interest_score['T_score_f'])))
    )
    df['interest_I_T'] = np.where(
        df['성별'] == 1,
        df['interest_I'].map(dict(zip(interest_score['score'], interest_score['T_score_m']))),
        df['interest_I'].map(dict(zip(interest_score['score'], interest_score['T_score_f'])))
    )
    df['interest_A_T'] = np.where(
        df['성별'] == 1,
        df['interest_A'].map(dict(zip(interest_score['score'], interest_score['T_score_m']))),
        df['interest_A'].map(dict(zip(interest_score['score'], interest_score['T_score_f'])))
    )

    # 성격 T점수
    df['person_S_T'] = np.where(
        df['성별'] == 1,
        df['person_S'].map(dict(zip(person_score['score'], person_score['T_score_m']))),
        df['person_S'].map(dict(zip(person_score['score'], person_score['T_score_f'])))
    )
    df['person_E_T'] = np.where(
        df['성별'] == 1,
        df['person_E'].map(dict(zip(person_score['score'], person_score['T_score_m']))),
        df['person_E'].map(dict(zip(person_score['score'], person_score['T_score_f'])))
    )
    df['person_C_T'] = np.where(
        df['성별'] == 1,
        df['person_C'].map(dict(zip(person_score['score'], person_score['T_score_m']))),
        df['person_C'].map(dict(zip(person_score['score'], person_score['T_score_f'])))
    )
    df['person_R_T'] = np.where(
        df['성별'] == 1,
        df['person_R'].map(dict(zip(person_score['score'], person_score['T_score_m']))),
        df['person_R'].map(dict(zip(person_score['score'], person_score['T_score_f'])))
    )
    df['person_I_T'] = np.where(
        df['성별'] == 1,
        df['person_I'].map(dict(zip(person_score['score'], person_score['T_score_m']))),
        df['person_I'].map(dict(zip(person_score['score'], person_score['T_score_f'])))
    )
    df['person_A_T'] = np.where(
        df['성별'] == 1,
        df['person_A'].map(dict(zip(person_score['score'], person_score['T_score_m']))),
        df['person_A'].map(dict(zip(person_score['score'], person_score['T_score_f'])))
    )

    # 적성 T점수
    df['apti_S_T'] = np.where(
        df['성별'] == 1,
        df['apti_S'].map(dict(zip(apti_score['score'], apti_score['T_score_m']))),
        df['apti_S'].map(dict(zip(apti_score['score'], apti_score['T_score_f'])))
    )
    df['apti_E_T'] = np.where(
        df['성별'] == 1,
        df['apti_E'].map(dict(zip(apti_score['score'], apti_score['T_score_m']))),
        df['apti_E'].map(dict(zip(apti_score['score'], apti_score['T_score_f'])))
    )
    df['apti_C_T'] = np.where(
        df['성별'] == 1,
        df['apti_C'].map(dict(zip(apti_score['score'], apti_score['T_score_m']))),
        df['apti_C'].map(dict(zip(apti_score['score'], apti_score['T_score_f'])))
    )
    df['apti_R_T'] = np.where(
        df['성별'] == 1,
        df['apti_R'].map(dict(zip(apti_score['score'], apti_score['T_score_m']))),
        df['apti_R'].map(dict(zip(apti_score['score'], apti_score['T_score_f'])))
    )
    df['apti_I_T'] = np.where(
        df['성별'] == 1,
        df['apti_I'].map(dict(zip(apti_score['score'], apti_score['T_score_m']))),
        df['apti_I'].map(dict(zip(apti_score['score'], apti_score['T_score_f'])))
    )
    df['apti_A_T'] = np.where(
        df['성별'] == 1,
        df['apti_A'].map(dict(zip(apti_score['score'], apti_score['T_score_m']))),
        df['apti_A'].map(dict(zip(apti_score['score'], apti_score['T_score_f'])))
    )

    # 선호직업 T점수
    df['job_S_T'] = np.where(
        df['성별'] == 1,
        df['job_S'].map(dict(zip(job_score['score'], job_score['T_score_m']))),
        df['job_S'].map(dict(zip(job_score['score'], job_score['T_score_f'])))
    )
    df['job_E_T'] = np.where(
        df['성별'] == 1,
        df['job_E'].map(dict(zip(job_score['score'], job_score['T_score_m']))),
        df['job_E'].map(dict(zip(job_score['score'], job_score['T_score_f'])))
    )
    df['job_C_T'] = np.where(
        df['성별'] == 1,
        df['job_C'].map(dict(zip(job_score['score'], job_score['T_score_m']))),
        df['job_C'].map(dict(zip(job_score['score'], job_score['T_score_f'])))
    )
    df['job_R_T'] = np.where(
        df['성별'] == 1,
        df['job_R'].map(dict(zip(job_score['score'], job_score['T_score_m']))),
        df['job_R'].map(dict(zip(job_score['score'], job_score['T_score_f'])))
    )
    df['job_I_T'] = np.where(
        df['성별'] == 1,
        df['job_I'].map(dict(zip(job_score['score'], job_score['T_score_m']))),
        df['job_I'].map(dict(zip(job_score['score'], job_score['T_score_f'])))
    )
    df['job_A_T'] = np.where(
        df['성별'] == 1,
        df['job_A'].map(dict(zip(job_score['score'], job_score['T_score_m']))),
        df['job_A'].map(dict(zip(job_score['score'], job_score['T_score_f'])))
    )

    # 직업환경 T점수
    df['env_D_T'] = np.where(
        df['성별'] == 1,
        df['env_D'].map(dict(zip(env_d_i__score['score'], env_d_i__score['T_score_m']))),
        df['env_D'].map(dict(zip(env_d_i__score['score'], env_d_i__score['T_score_f'])))
    )
    df['env_I_T'] = np.where(
        df['성별'] == 1,
        df['env_I'].map(dict(zip(env_d_i__score['score'], env_d_i__score['T_score_m']))),
        df['env_I'].map(dict(zip(env_d_i__score['score'], env_d_i__score['T_score_f'])))
    )
    df['env_P_T'] = np.where(
        df['성별'] == 1,
        df['env_P'].map(dict(zip(env_p_t_score['score'], env_p_t_score['T_score_m']))),
        df['env_P'].map(dict(zip(env_p_t_score['score'], env_p_t_score['T_score_f'])))
    )
    df['env_T_T'] = np.where(
        df['성별'] == 1,
        df['env_T'].map(dict(zip(env_p_t_score['score'], env_p_t_score['T_score_m']))),
        df['env_T'].map(dict(zip(env_p_t_score['score'], env_p_t_score['T_score_f'])))
    )
    # ------------------------------------------------------

    # (6) 최종 factor 계산 및 순위
    df['factor_S'] = ((df['interest_S_T']*2) + df['person_S_T'] + (df['apti_S_T']*2) + df['job_S_T'])/6
    df['factor_E'] = ((df['interest_E_T']*2) + df['person_E_T'] + (df['apti_E_T']*2) + df['job_E_T'])/6
    df['factor_C'] = ((df['interest_C_T']*2) + df['person_C_T'] + (df['apti_C_T']*2) + df['job_C_T'])/6
    df['factor_R'] = ((df['interest_R_T']*2) + df['person_R_T'] + (df['apti_R_T']*2) + df['job_R_T'])/6
    df['factor_I'] = ((df['interest_I_T']*2) + df['person_I_T'] + (df['apti_I_T']*2) + df['job_I_T'])/6
    df['factor_A'] = ((df['interest_A_T']*2) + df['person_A_T'] + (df['apti_A_T']*2) + df['job_A_T'])/6

    factor_columns = ['factor_S','factor_E','factor_C','factor_R','factor_I','factor_A']
    factor_labels  = ['S','E','C','R','I','A']

    # Holland 육각형에서 각 코드 간의 '거리'를 정의
    # S, E, C, R, I, A 순으로 육각형이 배치되어 있다고 가정
    distance_map = {
        ('S','S'):0, ('S','E'):1, ('S','C'):2, ('S','R'):3, ('S','I'):2, ('S','A'):1,
        ('E','S'):1, ('E','E'):0, ('E','C'):1, ('E','R'):2, ('E','I'):3, ('E','A'):2,
        ('C','S'):2, ('C','E'):1, ('C','C'):0, ('C','R'):1, ('C','I'):2, ('C','A'):3,
        ('R','S'):3, ('R','E'):2, ('R','C'):1, ('R','R'):0, ('R','I'):1, ('R','A'):2,
        ('I','S'):2, ('I','E'):3, ('I','C'):2, ('I','R'):1, ('I','I'):0, ('I','A'):1,
        ('A','S'):1, ('A','E'):2, ('A','C'):3, ('A','R'):2, ('A','I'):1, ('A','A'):0,
    }

    def get_distance(a, b):
        """두 Holland 코드 a, b 사이의 미리 정의된 거리 반환"""
        return distance_map[(a, b)]

    def tie_break_top(sorted_items):
        """
        sorted_items는 [(코드, 점수), (코드, 점수), ...] 형태로
        점수가 큰 순서대로 정렬된 리스트입니다.
        이 중 1위(첫 번째 그룹)에서 동점자 처리만 특별 로직을 적용합니다.
        """
        # 1) 먼저 1위의 점수를 확인하고, 동점인 코드들을 모읍니다.
        top_score = sorted_items[0][1]
        first_place_codes = [code for code, sc in sorted_items if sc == top_score]
        
        # 동점자가 1개면 특별 처리를 하지 않아도 됨
        if len(first_place_codes) == 1:
            return sorted_items
        
        # 6위(가장 점수가 낮은) 코드 찾기
        # sorted_items가 이미 내림차순이므로 제일 끝이 6위
        # (만약 여러 개가 똑같이 최저점이라면, 그 중 마지막을 사용)
        code6 = sorted_items[-1][0]

        # ---- 동점자가 2개인 경우 ----
        if len(first_place_codes) == 2:
            c1, c2 = first_place_codes
            dist1 = get_distance(c1, code6)
            dist2 = get_distance(c2, code6)
            
            if dist1 > dist2:
                # c1이 1위, c2가 2위
                new_order_top = [c1, c2]
            elif dist1 < dist2:
                # c2가 1위, c1이 2위
                new_order_top = [c2, c1]
            else:
                # dist1 == dist2 (6위와의 거리가 동일)인 경우
                # -> 3위(정렬 기준상 index=2) 코드를 찾아서 더 가까운 쪽을 1위로
                code3 = sorted_items[2][0]  # 0번:1위,1번:2위,2번:3위
                d13 = get_distance(c1, code3)
                d23 = get_distance(c2, code3)
                if d13 < d23:
                    new_order_top = [c1, c2]
                elif d13 > d23:
                    new_order_top = [c2, c1]
                else:
                    # 만약 여기서도 같으면, 그대로 두거나 임의로 결정
                    new_order_top = [c1, c2]
            
            # 새로운 1,2위 순서를 반영하여 전체 순서를 재배치
            # first_place_codes 에 해당되는 부분만 교체
            # sorted_items는 (code, score) 튜플 리스트이므로,
            # new_order_top 기준으로 앞에 재배치
            re_ordered = []
            used = set()
            # 1위,2위 재배치
            for c in new_order_top:
                sc = [s for (cd, s) in sorted_items if cd == c][0]
                re_ordered.append((c, sc))
                used.add(c)
            # 나머지 (3~6위) 재배치
            for (cd, s) in sorted_items:
                if cd not in used:
                    re_ordered.append((cd, s))
            return re_ordered
        
        # ---- 동점자가 3개인 경우 ----
        elif len(first_place_codes) == 3:
            c1, c2, c3 = first_place_codes
            dist_c1 = get_distance(c1, code6)
            dist_c2 = get_distance(c2, code6)
            dist_c3 = get_distance(c3, code6)
            
            # 세 개 중 가장 먼 거리를 찾는다
            max_dist = max(dist_c1, dist_c2, dist_c3)
            # 이 값에 해당하는 코드(들)을 찾는다
            candidates = []
            for c, d in [(c1, dist_c1), (c2, dist_c2), (c3, dist_c3)]:
                if d == max_dist:
                    candidates.append(c)
            
            if len(candidates) == 1:
                # 1명만 최대 거리 -> 그 코드가 1위
                first = candidates[0]
                # 나머지 2개는 first와의 거리가 가까운 순으로 2,3위
                others = [x for x in [c1, c2, c3] if x != first]
                d_oth = [(c, get_distance(c, first)) for c in others]
                d_oth.sort(key=lambda x: x[1])  # 거리 오름차순
                second = d_oth[0][0]
                third = d_oth[1][0]
                
            else:
                # 2개 이상이 최대 거리 -> 그 둘(혹은 세 개) 중에서 추가 판단
                # 예: S, I가 동점 최대 거리, 나머지 E가 상대적으로 작음
                #    -> 나머지 E와의 거리가 더 가까운 쪽이 1위
                # (세 개가 모두 같은 거리인 극단적 상황은 가정치 않거나,
                #  임의 처리를 해주면 됨)
                leftover = [x for x in [c1, c2, c3] if x not in candidates]
                # leftover가 비어있지 않다면, leftover[0]을 기준으로 더 가까운 쪽이 1위
                # leftover가 비어있으면(=3개 전부 같은 거리) 적당히 처리
                if len(leftover) == 1:
                    mid_code = leftover[0]
                    # candidates 리스트 안의 코드들과 mid_code 거리 비교
                    dist_list = [(cd, get_distance(cd, mid_code)) for cd in candidates]
                    # mid_code 에게 더 가까운 순으로 정렬
                    dist_list.sort(key=lambda x: x[1])
                    first = dist_list[0][0]   # 더 가까운 쪽
                    second = dist_list[1][0] # 그 다음
                    # leftover는 최대거리 아니므로 3위
                    third = mid_code
                else:
                    # 혹은 세 개가 전부 같은 거리 등등 특수 케이스
                    # 일단 임의로 정렬 (알파벳 순 등)
                    candidates.sort()
                    first = candidates[0]
                    second = candidates[1]
                    third = candidates[2]
                
            # 이제 first, second, third 순서를 전체 순서에 반영
            re_ordered = []
            used = {first, second, third}
            # 1위,2위,3위
            sc_first = [s for (cd, s) in sorted_items if cd == first][0]
            sc_second = [s for (cd, s) in sorted_items if cd == second][0]
            sc_third = [s for (cd, s) in sorted_items if cd == third][0]
            re_ordered.append((first, sc_first))
            re_ordered.append((second, sc_second))
            re_ordered.append((third, sc_third))
            # 나머지 4~6위
            for (cd, s) in sorted_items:
                if cd not in used:
                    re_ordered.append((cd, s))
            return re_ordered
        
        else:
            # 동점자가 4명 이상인 경우 등은 별도 처리(여기서는 기본 정렬 유지)
            return sorted_items


    def get_rank_with_holland_tie(row, factor_labels, factor_columns):
        # label과 실제 컬럼명을 매핑하여 점수를 가져옴
        scores = {}
        for lbl, col in zip(factor_labels, factor_columns):
            scores[lbl] = row[col]

        # 점수 내림차순 정렬
        sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        # 1위 동점자 처리
        sorted_items = tie_break_top(sorted_items)

        # (labels, scores) 분리
        sorted_labels = [x[0] for x in sorted_items]
        sorted_scores = [x[1] for x in sorted_items]
        return sorted_labels, sorted_scores

    # ------------------------------------------
    # 실제 DF에 적용
    factor_columns = ['factor_S', 'factor_E', 'factor_C', 'factor_R', 'factor_I', 'factor_A']
    factor_labels  = ['S', 'E', 'C', 'R', 'I', 'A']

    df[['순위_labels', '순위_scores']] = df[factor_columns].apply(
        lambda row: get_rank_with_holland_tie(row, factor_labels, factor_columns),
        axis=1,
        result_type='expand'
    )

    for rank in range(1, 7):
        df[f'{rank}순위'] = df['순위_labels'].apply(lambda x: x[rank - 1])
        df[f'{rank}순위_점수'] = df['순위_scores'].apply(lambda x: x[rank - 1])

    df.drop(['순위_labels', '순위_scores'], axis=1, inplace=True)
    df.head()



    #요인별 순위 구하기(흥미, 적성, 성격, 직업선호)
    ##흥미
    interest_columns = ['interest_S_T', 'interest_E_T', 'interest_C_T', 'interest_R_T', 'interest_I_T', 'interest_A_T']
    factor_labels = ['S', 'E', 'C', 'R', 'I', 'A']

    def get_rank(row, factor_labels):
        """동점을 고려하여 순위를 결정하는 함수."""
        scores = row.values
        sorted_indices = np.argsort(-scores)  # 큰 순서대로 정렬
        sorted_labels = [factor_labels[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_labels, sorted_scores

    # 순위 계산
    df[['int순위_labels', 'int순위_scores']] = df[interest_columns].apply(
        lambda row: get_rank(row, factor_labels), axis=1, result_type='expand'
    )

    # 각 순위 및 점수 컬럼 추가
    for rank in range(1, 7):  # 1순위부터 6순위까지
        df[f'Int{rank}순위'] = df['int순위_labels'].apply(lambda x: x[rank - 1])
        df[f'Int{rank}순위_점수'] = df['int순위_scores'].apply(lambda x: x[rank - 1])

    # 결과에서 임시 컬럼 제거
    df.drop(['int순위_labels', 'int순위_scores'], axis=1, inplace=True)
    ###성격
    person_columns = ['person_S_T', 'person_E_T', 'person_C_T', 'person_R_T', 'person_I_T', 'person_A_T']
    factor_labels = ['S', 'E', 'C', 'R', 'I', 'A']

    def get_rank(row, factor_labels):
        """동점을 고려하여 순위를 결정하는 함수."""
        scores = row.values
        sorted_indices = np.argsort(-scores)  # 큰 순서대로 정렬
        sorted_labels = [factor_labels[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_labels, sorted_scores

    # 순위 계산
    df[['per순위_labels', 'per순위_scores']] = df[person_columns].apply(
        lambda row: get_rank(row, factor_labels), axis=1, result_type='expand'
    )

    # 각 순위 및 점수 컬럼 추가
    for rank in range(1, 7):  # 1순위부터 6순위까지
        df[f'Per{rank}순위'] = df['per순위_labels'].apply(lambda x: x[rank - 1])
        df[f'Per{rank}순위_점수'] = df['per순위_scores'].apply(lambda x: x[rank - 1])

    # 결과에서 임시 컬럼 제거
    df.drop(['per순위_labels', 'per순위_scores'], axis=1, inplace=True)

    ###적성
    apti_columns = ['apti_S_T', 'apti_E_T', 'apti_C_T', 'apti_R_T', 'apti_I_T', 'apti_A_T']
    factor_labels = ['S', 'E', 'C', 'R', 'I', 'A']

    def get_rank(row, factor_labels):
        """동점을 고려하여 순위를 결정하는 함수."""
        scores = row.values
        sorted_indices = np.argsort(-scores)  # 큰 순서대로 정렬
        sorted_labels = [factor_labels[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_labels, sorted_scores

    # 순위 계산
    df[['apit순위_labels', 'apit순위_scores']] = df[apti_columns].apply(
        lambda row: get_rank(row,factor_labels), axis=1, result_type='expand')

    # 각 순위 및 점수 컬럼 추가
    for rank in range(1, 7):  # 1순위부터 6순위까지
        df[f'Apti{rank}순위'] = df['apit순위_labels'].apply(lambda x: x[rank - 1])
        df[f'Apti{rank}순위_점수'] = df['apit순위_scores'].apply(lambda x: x[rank - 1])

    # 결과에서 임시 컬럼 제거
    df.drop(['apit순위_labels', 'apit순위_scores'], axis=1, inplace=True)


    ###직업 선호
    job_columns = ['job_S_T', 'job_E_T', 'job_C_T', 'job_R_T', 'job_I_T', 'job_A_T']
    factor_labels = ['S', 'E', 'C', 'R', 'I', 'A']

    def get_rank(row, factor_labels):
        """동점을 고려하여 순위를 결정하는 함수."""
        scores = row.values
        sorted_indices = np.argsort(-scores)  # 큰 순서대로 정렬
        sorted_labels = [factor_labels[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_labels, sorted_scores

    # 순위 계산
    df[['job순위_labels', 'job순위_scores']] = df[job_columns].apply(
        lambda row: get_rank(row, factor_labels), axis=1, result_type='expand'
    )

    # 각 순위 및 점수 컬럼 추가
    for rank in range(1, 7):  # 1순위부터 6순위까지
        df[f'jobpre{rank}순위'] = df['job순위_labels'].apply(lambda x: x[rank - 1])
        df[f'jobpre{rank}순위_점수'] = df['job순위_scores'].apply(lambda x: x[rank - 1])

    # 결과에서 임시 컬럼 제거
    df.drop(['job순위_labels', 'job순위_scores'], axis=1, inplace=True)


    ####매칭 개수 계산
    for rank in range(1, 7):
        # 예) '1순위' 컬럼명, '2순위' 컬럼명 ...
        col_code = f'{rank}순위'
        
        # 예) 'Int1순위', 'Int2순위', ... 'Per1순위', 'Apti1순위', 'jobpre1순위' ...
        col_int  = f'Int{rank}순위'
        col_per  = f'Per{rank}순위'
        col_apti = f'Apti{rank}순위'
        col_job  = f'jobpre{rank}순위'
    
        # 각 순위별 코드가 동일한지 체크하여 True(1) / False(0) 로 더해줌
        df[f'{rank}순위_매칭개수'] = (
            (df[col_code] == df[col_int]).astype(int)
            + (df[col_code] == df[col_per]).astype(int)
            + (df[col_code] == df[col_apti]).astype(int)
            + (df[col_code] == df[col_job]).astype(int))
        

    df['반응적합도_비율'] = round(((df['1순위_매칭개수']+df['2순위_매칭개수']+df['3순위_매칭개수']+df['4순위_매칭개수']+df['5순위_매칭개수']+df['6순위_매칭개수'])/24)*100, 2)

    df_count = df.iloc[:,9:213]
    df_count.notnull().sum(axis=1)

    df['반응적합'] = round((df_count.notnull().sum(axis=1)/204)*100, 2)
    
    df['변별도']   = round(df['1순위_점수'] - ((df['2순위_점수']+df['4순위_점수'])/2), 2)

    # 최적 코드 분류
    df['Code1'] = np.where(
        (df['반응적합'] <= 70),'분류불능a',
        np.where(
            (df['변별도'] < 1),'분류불능b',
            np.where(
                (df['factor_S']<30)&(df['factor_E']<30)&(df['factor_C']<30)&
                (df['factor_R']<30)&(df['factor_I']<30)&(df['factor_A']<30), '분류불능c',
                np.where(
                    df['성별'].isna(), None,
                    df['1순위'] + df['2순위']  # 기본은 1순위+2순위 결합
                )
            )
        )
    )

    # 적합 코드 분류
    df['Code2'] = np.where(
        df['Code1'].isin(['분류불능a','분류불능b','분류불능c']), '분류불능',
        np.where(
            df['2순위_점수']-df['3순위_점수']>=3,
            df['2순위']+df['1순위'],
            df['1순위']+df['3순위']
        )
    )

    # 1순위/2순위 코드 넘버링
    priority_mapping = {'S':1,'E':2,'C':3,'R':4,'I':5,'A':6}
    df['1순위 코드 넘버'] = df['1순위'].map(priority_mapping)
    df['2순위 코드 넘버'] = df['2순위'].map(priority_mapping)

    # 직업환경 코드
    conditions = [
        df['env_P_T'] > df['env_T_T'],                               # P가 더 큰 경우
        df['env_P_T'] < df['env_T_T'],                               # T가 더 큰 경우
        (df['env_P_T'] == df['env_T_T']) & df['1순위 코드 넘버'].isin([1,2,3])
    ]
    choices = ['P','T','P']
    default_value = 'T'
    df['직환code1'] = np.select(conditions, choices, default=default_value)

    conditions2 = [
        df['env_D_T'] > df['env_I_T'],
        df['env_D_T'] < df['env_I_T'],
        (df['env_D_T'] == df['env_I_T']) & df['1순위 코드 넘버'].isin([1,2,3])
    ]
    choices2 = ['D','I','D']
    default_value2 = 'I'
    df['직환code2'] = np.select(conditions2, choices2, default=default_value2)

    df['직업환경 Code'] = np.where(
        (df['env_D_T']<30)&(df['env_I_T']<30)&(df['env_P_T']<30)&(df['env_T_T']<30),
        '분류불능d',
        (df['직환code2']+df['직환code1'])
    )

    # 학과, 직업 리스트 매핑
    df['major'] = np.where(
        df['Code1'].isin(['분류불능a','분류불능b','분류불능c']),
        '분류불능',
        df['Code1'].map(dict(zip(major['code'],major['major'])))
    )
    df['joblist'] = np.where(
        df['Code1'].isin(['분류불능a','분류불능b','분류불능c']),
        '분류불능',
        df['Code1'].map(dict(zip(joblist['code'],joblist['job'])))
    )
    df['major2'] = np.where(
        df['Code2'] == '분류불능',
        '분류불능',
        df['Code2'].map(dict(zip(major['code'],major['major'])))
    )
    df['joblist2'] = np.where(
        df['Code2'] == '분류불능',
        '분류불능',
        df['Code2'].map(dict(zip(joblist['code'],joblist['job'])))
    )
    # ------------------------------------------------------

    # (7) 결과 저장
    df.to_excel(output_path, index=False)

    print(df_count)

    # 함수 끝
