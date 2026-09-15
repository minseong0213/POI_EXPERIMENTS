# 선택 모델의 적대적 공격 성능

clean validation 1위 `tabpfn_v3`를 clean train으로 학습하고 32개 검증 공격 전체를 평가했다. test split은 사용하지 않았다.

Macro OvR F1이 가장 낮은 조건은 `pgd_m1000`이며 0.988409이다. 전체 F1·precision·recall·accuracy·ROC-AUC·AP와 clean 대비 감소량은 tables에, confusion matrix·PR·ROC 및 전 조건 heatmap은 300dpi PNG로 저장했다.
