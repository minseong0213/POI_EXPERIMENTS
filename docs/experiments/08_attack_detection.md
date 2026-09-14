# 08. 공격 탐지 이진분류

normal=0, adversarial=1로 판별한다. 한 POI에서 나온 두 행을 독립 분할하면 공격 전후
쌍이 train/test에 갈라져 누수가 생기므로 기존 POI split을 그대로 사용한다. 모델 입력에
POI_ID, region 정답, split, 공격 여부를 직접 복원하는 메타데이터를 넣지 않는다.

8개 모델에 대해 precision, recall, F1, accuracy, ROC-AUC, AP, confusion matrix,
PR/ROC curve를 계산한다. 전체와 17개 실제 지역 subgroup을 모두 보고한다. 정상 POI를
공격으로 거부하는 false positive rate와 공격을 정상으로 통과시키는 false negative rate를
명시한다.

판별 odds ratio는 표준화 logistic regression으로 OR/95% CI/p/q를 구한다. 공격 전후
값이 기계적으로 다른 문자열 이름·주소는 탐지 성능을 과장할 수 있으므로 지역 분류와
동일한 허용 피처만 쓰는 주 분석과 확장 피처 분석을 분리한다.
