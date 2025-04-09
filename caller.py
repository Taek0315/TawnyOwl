from importlib import reload
import CET_logic_ele_row  # 이미 로드되어 있는 모듈

reload(CET_logic_ele_row)  # 수정된 .py 파일 내용을 다시 반영
# 모듈이 재로딩된 상태에서 함수를 호출
from datetime import datetime

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path_with_timestamp = f"C:/Users/user/Desktop/연구사업부/code/CET logic/example_test_{now_str}.xlsx"

CET_logic_ele_row.cet_process(
    input_path="C:/Users/user/Desktop/연구사업부/code/CET logic/example_초등.xlsx",
    score_path="C:/Users/user/Desktop/연구사업부/code/CET logic/score_ele.xlsx",
    output_path=output_path_with_timestamp
)

print("CET 로직 수행 완료!")
