# 01. 적대적 공격 생성 및 유효성 검증

지역·주소·ID를 입력에서 제외한 5개 피처를 대상으로 train과 validation에 총
32개 공격 조건, 870,400개 결과 행을 만들었다. test는 열지 않았다.
수치 공격은 좌표 이동을 미터 단위 L2 거리로 제한하고, 범주 공격은 공격 대상 clean
split에서 label과 무관하게 고정한 97개 대·중·소분류 조합만 허용했다.
test 조합은 사용하지 않았고 실패 표본도 원본과 함께 저장했다.

FGSM, PGD, CW-L2는 표준 정의를 좌표에 적용했다. CAPGD, PCAA, MOEVA와 CAA는 현재
PyTorch/TabPFN 환경에서 동작하도록 POI 제약에 맞춰 이식한 구현이며 원 논문 구현과
byte-identical하지 않다. CAA는 CAPGD 후 실패 표본에 MOEVA를 적용한다.

모든 행에는 공격 방법·예산·source model·split·이동거리·변경 mask·query 수·제약 충족·
source 성공 여부를 기록했다. 유효성 검사에서 제약 위반이 한 건이라도 있으면 이 단계는
성공 마커를 만들지 않는다.
