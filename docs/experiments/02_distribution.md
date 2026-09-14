# 02. 분포 시각화

숫자 피처는 지역별 normal/adversarial paired box plot과 histogram을 만든다.
점이 너무 많을 때 box plot은 전체 통계를 사용하되 scatter 표시는 고정 seed의 표본을
사용한다. 좌표 scatter는 같은 POI의 공격 전후를 선으로 연결한 표본 그림과 전체 밀도
그림을 모두 저장한다.

## 산출물

- 지역 × 조건 box plot: X/Y 좌표 및 이후 추가되는 연속형 피처
- 피처별 histogram/KDE: 공통 bin과 축 범위 사용
- X–Y scatter/hexbin: 지역 색상, 공격 조건 marker
- 공격 전후 변화량(delta)의 histogram과 지역별 box plot
- 그래프별 원자료 또는 bin/summary CSV

지역 17개를 한 그래프에 과도하게 겹치지 않고 facet과 전체 요약을 함께 쓴다. normal과
adversarial을 서로 다른 축 범위로 그리지 않는다. 식별 가능한 POI 이름·주소는 그림과
보고서에 출력하지 않는다.
