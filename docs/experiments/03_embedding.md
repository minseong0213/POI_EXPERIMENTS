# 03. 임베딩과 군집

임베딩은 모델 성능 증거가 아니라 구조와 공격 이동을 탐색하는 시각화다. 전처리는
학습 데이터에서 fit한다. normal과 adversarial을 각각 따로 임베딩해 좌표를 직접
비교하지 않는다.

## t-SNE

같은 POI_ID의 normal/adversarial 행을 함께 고정 표본으로 뽑아 결합한 공간에서 한 번
fit한다. PCA 사전 축소, perplexity, learning rate, iteration, seed를 기록한다.
지역 색상·조건 marker 그림과 공격 전후 이동선 표본을 저장한다.

## UMAP

clean train에서 UMAP을 fit하고 normal/adversarial validation을 같은 embedding으로
transform한다. n_neighbors, min_dist, metric, seed를 기록한다. 전체/지역별 그림과
POI별 embedding 이동 거리 표를 저장한다.

## Dendrogram

34,000개 행 전체의 dendrogram을 만들지 않는다. 지역×조건별 표준화 피처 centroid를
구하고 bootstrap 안정성을 포함한 계층 군집을 만든다. 거리 metric과 linkage를 기록한다.

산출물은 `figures/tsne_*`, `figures/umap_*`, `figures/dendrogram_*`, 좌표 CSV,
파라미터 JSON, 정량 보조지표를 포함한다. seed를 바꿨을 때 주요 구조가 유지되는지는
10단계 stability 분석에서 확인한다.
