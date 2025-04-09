import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
df = pd.read_excel("C:\\Users\\user\\Desktop\\연구사업부\\code\\CET logic\\example.xlsx")

##흥미 부분 전처리

import numpy as np
#흥미 관련 문항 총 24개를 for문을 활용하여 각 응답에 알맞게 전처리 코딩 작업
for i in range(1,25):
    df[f'I{i}-1'] = np.where(df[f'I_{i}'] == 1, 1, 0)
    df[f'I{i}-2'] = np.where(df[f'I_{i}'] == 2, 1, 0)
    df[f'I{i}-3'] = np.where(df[f'I_{i}'] == 3, 1, 0)
    df[f'I{i}-4'] = np.where(~df[f'I_{i}'].isin([1, 2, 3]), 1, 0)
#성격 데이터 불러오기 및 직업환경 선호도 전처리
for n in range(1,31):
    df[f'P{n}'] = df[f'P_{n}'].copy()
    df[f'EJ{n}-1'] = np.where(df[f'EJ_{n}'] == 1,1,0)
    df[f'EJ{n}-2'] = np.where(df[f'EJ_{n}'] == 2,1,0)
#적성 및 선호직업 불러오기
for w in range(1,61):
    df[f'T{w}'] = df[f'T_{w}'].copy()
    df[f'pre{w}'] = df[f'pre_{w}'].copy()



#흥미요인 계산
df['interest_S'] = df[['I1-1', 'I4-4', 'I5-3', 'I6-2', 'I7-1', 'I10-4', 'I11-3', 'I12-2', 'I13-1', 'I16-4', 'I17-3', 'I18-2', 'I19-1', 'I22-4', 'I23-3', 'I24-2']].sum(axis=1)
df['interest_E'] = df[['I1-2', 'I2-1', 'I5-4', 'I6-3', 'I7-2','I8-1', 'I11-4', 'I12-3', 'I13-2', 'I14-1', 'I17-4', 'I18-3', 'I19-2', 'I20-1', 'I23-4', 'I24-3' ]].sum(axis=1)
df['interest_C'] = df[['I1-3', 'I2-2', 'I3-1', 'I6-4', 'I7-3', 'I8-2', 'I9-1', 'I12-4', 'I13-3', 'I14-2', 'I15-1', 'I18-4', 'I19-3', 'I20-2', 'I21-1', 'I24-4']].sum(axis=1)
df['interest_R'] = df[['I1-4', 'I2-3', 'I3-2', 'I4-1', 'I7-4', 'I8-3', 'I9-2', 'I10-1', 'I13-4', 'I14-3', 'I15-2', 'I16-1', 'I19-4', 'I20-3', 'I21-2', 'I22-1']].sum(axis=1)
df['interest_I'] = df[['I2-4', 'I3-3', 'I4-2', 'I5-1', 'I8-4', 'I9-3', 'I10-2', 'I11-1', 'I14-4', 'I15-3', 'I16-2', 'I17-1', 'I20-4', 'I21-3', 'I22-2', 'I23-1']].sum(axis=1)
df['interest_A'] = df[['I3-4', 'I4-3', 'I5-2', 'I6-1', 'I9-4', 'I10-3', 'I11-2', 'I12-1', 'I15-4', 'I16-3', 'I17-2', 'I18-1', 'I21-4', 'I22-3', 'I23-2', 'I24-1']].sum(axis=1)

#성격요인 계산
df['person_S'] = df[['P1','P7','P13','P19','P25']].sum(axis=1)
df['person_E'] = df[['P2','P8','P14','P20','P26']].sum(axis=1)
df['person_C'] = df[['P3','P9','P15','P21','P27']].sum(axis=1)
df['person_R'] = df[['P4','P10','P16','P22','P28']].sum(axis=1)
df['person_I'] = df[['P5','P11','P17','P23','P29']].sum(axis=1)
df['person_A'] = df[['P6','P12','P18','P24','P30']].sum(axis=1)

#적성요인 계산
df['apti_S'] = df[['T1', 'T7', 'T13', 'T19', 'T25', 'T31', 'T37', 'T43', 'T49', 'T55']].sum(axis=1)
df['apti_E'] = df[['T2', 'T8', 'T14', 'T20', 'T26', 'T32', 'T38', 'T44', 'T50', 'T56']].sum(axis=1)
df['apti_C'] = df[['T3', 'T9', 'T15', 'T21', 'T27', 'T33', 'T39', 'T45', 'T51', 'T57']].sum(axis=1)
df['apti_R'] = df[['T4', 'T10', 'T16', 'T22', 'T28', 'T34', 'T40', 'T46', 'T52', 'T58']].sum(axis=1)
df['apti_I'] = df[['T5', 'T11', 'T17', 'T23', 'T29', 'T35', 'T41', 'T47', 'T53', 'T59']].sum(axis=1)
df['apti_A'] = df[['T6', 'T12', 'T18', 'T24', 'T30', 'T36', 'T42', 'T48', 'T54', 'T60']].sum(axis=1)

#직업환경요인 계산
df['env_D'] = df[['EJ2-2', 'EJ4-1', 'EJ6-1', 'EJ8-1', 'EJ10-1', 'EJ12-2', 'EJ14-1', 'EJ16-2', 'EJ18-2', 'EJ20-1', 'EJ22-1', 'EJ24-2', 'EJ26-2', 'EJ28-1', 'EJ30-1']].sum(axis=1)
df['env_I'] = df[['EJ2-1', 'EJ4-2', 'EJ6-2', 'EJ8-2', 'EJ10-2', 'EJ12-1', 'EJ14-2', 'EJ16-1', 'EJ18-1', 'EJ20-2', 'EJ22-2', 'EJ24-1', 'EJ26-1', 'EJ28-2', 'EJ30-2']].sum(axis=1)
df['env_P'] = df[['EJ1-1', 'EJ3-1', 'EJ5-2', 'EJ7-1', 'EJ9-1', 'EJ11-2', 'EJ13-2', 'EJ15-1', 'EJ17-2', 'EJ19-1', 'EJ21-2', 'EJ23-1', 'EJ25-2', 'EJ27-1', 'EJ29-2']].sum(axis=1)
df['env_T'] = df[['EJ1-2', 'EJ3-2', 'EJ5-1', 'EJ7-2', 'EJ9-2', 'EJ11-1', 'EJ13-1', 'EJ15-2', 'EJ17-1', 'EJ19-2', 'EJ21-1', 'EJ23-2', 'EJ25-1', 'EJ27-2', 'EJ29-1']].sum(axis=1)

#선호직업요인 계산
df['job_S'] = df[['pre1', 'pre7', 'pre13', 'pre19', 'pre25', 'pre31', 'pre37', 'pre43', 'pre49', 'pre55']].sum(axis=1)
df['job_E'] = df[['pre2', 'pre8', 'pre14', 'pre20', 'pre26', 'pre32', 'pre38', 'pre44', 'pre50', 'pre56']].sum(axis=1)
df['job_C'] = df[['pre3', 'pre9', 'pre15', 'pre21', 'pre27', 'pre33', 'pre39', 'pre45', 'pre51', 'pre57']].sum(axis=1)
df['job_R'] = df[['pre4', 'pre10', 'pre16', 'pre22', 'pre28', 'pre34', 'pre40', 'pre46', 'pre52', 'pre58']].sum(axis=1)
df['job_I'] = df[['pre5', 'pre11', 'pre17', 'pre23', 'pre29', 'pre35', 'pre41', 'pre47', 'pre53', 'pre59']].sum(axis=1)
df['job_A'] = df[['pre6', 'pre12', 'pre18', 'pre24', 'pre30', 'pre36', 'pre42', 'pre48', 'pre54', 'pre60']].sum(axis=1)



#규준 데이터 불러오기
interest_score = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="interest")
person_score = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="person")
apti_score = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="apti")
job_score = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="job")
env_p_t_score = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="env(p_t)")
env_d_i__score = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="env(d_i)")



## T점수 작성하기

#흥미 T점수
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

#성격 T점수
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

#적성 T점수
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

#선호직업 T점수
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

#직업환경 T점수
df['env_D_T'] = np.where(
    df['성별'] == 1,
    df['env_D'].map(dict(zip(env_d_i__score['score'],env_d_i__score['T_score_m']))),
    df['env_D'].map(dict(zip(env_d_i__score['score'],env_d_i__score['T_score_f'])))
)

df['env_I_T'] = np.where(
    df['성별'] == 1,
    df['env_I'].map(dict(zip(env_d_i__score['score'],env_d_i__score['T_score_m']))),
    df['env_I'].map(dict(zip(env_d_i__score['score'],env_d_i__score['T_score_f'])))
)

df['env_P_T'] = np.where(
    df['성별'] == 1,
    df['env_P'].map(dict(zip(env_p_t_score['score'],env_p_t_score['T_score_m']))),
    df['env_P'].map(dict(zip(env_p_t_score['score'],env_p_t_score['T_score_f'])))
)

df['env_T_T'] = np.where(
    df['성별'] == 1,
    df['env_T'].map(dict(zip(env_p_t_score['score'],env_p_t_score['T_score_m']))),
    df['env_T'].map(dict(zip(env_p_t_score['score'],env_p_t_score['T_score_f'])))
)



df['factor_S'] = ((df['interest_S_T']*2) + df['person_S_T'] + (df['apti_S_T']*2) + df['job_S_T'])/6
df['factor_E'] = ((df['interest_E_T']*2) + df['person_E_T'] + (df['apti_E_T']*2) + df['job_E_T'])/6
df['factor_C'] = ((df['interest_C_T']*2) + df['person_C_T'] + (df['apti_C_T']*2) + df['job_C_T'])/6
df['factor_R'] = ((df['interest_R_T']*2) + df['person_R_T'] + (df['apti_R_T']*2) + df['job_R_T'])/6
df['factor_I'] = ((df['interest_I_T']*2) + df['person_I_T'] + (df['apti_I_T']*2) + df['job_I_T'])/6
df['factor_A'] = ((df['interest_A_T']*2) + df['person_A_T'] + (df['apti_A_T']*2) + df['job_A_T'])/6



factor_columns = ['factor_S', 'factor_E', 'factor_C', 'factor_R', 'factor_I', 'factor_A']
factor_labels = ['S', 'E', 'C', 'R', 'I', 'A']

def get_rank(row, factor_labels):
    """동점을 고려하여 순위를 결정하는 함수."""
    scores = row.values
    sorted_indices = np.argsort(-scores)  # 큰 순서대로 정렬
    sorted_labels = [factor_labels[i] for i in sorted_indices]
    sorted_scores = scores[sorted_indices]
    return sorted_labels, sorted_scores

# 순위 계산
df[['순위_labels', '순위_scores']] = df[factor_columns].apply(
    lambda row: get_rank(row, factor_labels), axis=1, result_type='expand'
)

# 각 순위 및 점수 컬럼 추가
for rank in range(1, 7):  # 1순위부터 6순위까지
    df[f'{rank}순위'] = df['순위_labels'].apply(lambda x: x[rank - 1])
    df[f'{rank}순위_점수'] = df['순위_scores'].apply(lambda x: x[rank - 1])

# 결과에서 임시 컬럼 제거
df.drop(['순위_labels', '순위_scores'], axis=1, inplace=True)


df_count = df.iloc[:,9:215]
df_count.notnull().sum(axis=1)

df['반응적합'] = (df_count.notnull().sum(axis=1)/204)*100

df['변별도'] = (df['1순위_점수'] - ((df['2순위_점수']+df['4순위_점수']))/2)


#최적 코드 변별

df['Code1'] = np.where(
    (df['반응적합'] <= 70), '분류불능a', 
    np.where((df['변별도'] < 1), '분류불능b',
    np.where((df['factor_S']<30)&(df['factor_E']<30)&(df['factor_C']<30)&
             (df['factor_R']<30)&(df['factor_I']<30)&(df['factor_A']<30), '분류불능c',
    np.where(df['성별'].isna(),None,df['1순위'] + df['2순위']  # 그 외에는 '1순위'와 '2순위' 결합
    )))
)

#적합 코드 변별

df['Code2'] = np.where(
    ((df['Code1'] == '분류불능a')|(df['Code1'] == '분류불능b')|(df['Code1'] == '분류불능c')), '분류불능',
    np.where(df['2순위_점수']-df['3순위_점수']>=3, (df['2순위']+df['1순위']),
             (df['1순위']+df['3순위']))
)



#1순위 2순위 코드 넘버링
#S:1, E:2, C:3, R:4, I:5, A:6
priority_mapping = {'S': 1, 'E': 2, 'C': 3, 'R': 4, 'I': 5, 'A': 6}
df['1순위 코드 넘버'] = df['1순위'].map(priority_mapping)
df['2순위 코드 넘버'] = df['2순위'].map(priority_mapping)



##직업 환경 코드 변별

conditions = [
    df['env_P_T'] > df['env_T_T'],  # env_P_T가 더 큰 경우
    df['env_P_T'] < df['env_T_T'],  # env_T_T가 더 큰 경우
    (df['env_P_T'] == df['env_T_T']) & df['1순위 코드 넘버'].isin([1, 2, 3])  # env_P_T == env_T_T이면서 1순위 코드 넘버가 1, 2, 3인 경우
]

# 각 조건에 해당하는 결과 값
choices = ['P', 'T', 'P']

# 기본 값 정의 (위 조건에 해당하지 않는 경우)
default_value = 'T'

# numpy.select로 '직환code1' 컬럼 생성
df['직환code1'] = np.select(conditions, choices, default=default_value)

conditions2 = [
    df['env_D_T'] > df['env_I_T'],  # env_P_T가 더 큰 경우
    df['env_D_T'] < df['env_I_T'],  # env_T_T가 더 큰 경우
    (df['env_D_T'] == df['env_I_T']) & df['1순위 코드 넘버'].isin([1, 2, 3])  # env_P_T == env_T_T이면서 1순위 코드 넘버가 1, 2, 3인 경우
]

# 각 조건에 해당하는 결과 값
choices2 = ['D', 'I', 'D']

# 기본 값 정의 (위 조건에 해당하지 않는 경우)
default_value2 = 'I'

# numpy.select로 '직환code1' 컬럼 생성
df['직환code2'] = np.select(conditions2, choices2, default=default_value2)



df['직업환경 Code'] = np.where((df['env_D_T'] < 30)&(df['env_I_T']<30)&(df['env_P_T']<30)&(df['env_T_T']<30) , '분류불능d',
                                  (df['직환code2']+df['직환code1']))



major = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="major")
joblist = pd.read_excel("C:\\Users\\user\\Desktop\\logic\\score.xlsx",sheet_name="joblist")



df['major'] = np.where((df['Code1'] == '분류불능a')|(df['Code1'] == '분류불능b')|(df['Code1'] == '분류불능c'), 
                              '분류불능', df['Code1'].map(dict(zip(major['code'], major['major']))))
df['joblist'] = np.where((df['Code1'] == '분류불능a')|(df['Code1'] == '분류불능b')|(df['Code1'] == '분류불능c'),
                                '분류불능', df['Code1'].map(dict(zip(joblist['code'], joblist['job']))))

df['major2'] = np.where((df['Code2'] == '분류불능'), 
                              '분류불능', df['Code2'].map(dict(zip(major['code'], major['major']))))
df['joblist2'] = np.where((df['Code2'] == '분류불능'),
                                '분류불능', df['Code2'].map(dict(zip(joblist['code'], joblist['job']))))

df.to_excel("C:\\Users\\user\\Desktop\\logic\\example_test_0321.xlsx")