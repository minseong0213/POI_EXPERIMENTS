# 03. t-SNE, UMAP, dendrogram

지역당 100개 paired POI를 사용해 joint t-SNE와 clean-fit UMAP을 생성했다. Dendrogram은 개별 23,800행이 아니라 34개 지역×조건 centroid를 Ward 방식으로 군집화했다. 임베딩은 탐색 시각화이며 분류 성능 근거로 사용하지 않는다.

## 생성 파일

- `figures/dendrogram_region_condition.png`
- `figures/tsne_region_condition.png`
- `figures/umap_region_condition.png`
- `tables/dendrogram_centroids.csv`
- `tables/embedding_coordinates.csv`

## 실행 범위

- 분석 데이터: train split의 paired POI 23,800개
- Seed: 42
- 모든 그림을 PNG 300 dpi로 생성함
- 상세 수치는 CSV와 metrics.json에 저장함

## 공격 범위

시각화는 사전 지정 대표 조건 `caa_high`를 사용했다. 전체 32개 공격 조건의 수치 결과는 CSV에 기록했다.
