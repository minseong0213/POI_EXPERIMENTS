"""Pinned provenance for the primary URE implementation."""

PAPER_URL = "https://arxiv.org/abs/2306.04064"
PAPER_TITLE = (
    "Transferable Adversarial Robustness for Categorical Data via Universal Robust "
    "Embeddings"
)
OFFICIAL_REPOSITORY = "https://github.com/spring-epfl/Transferable-Cat-Robustness"
OFFICIAL_COMMIT = "18aefb51ba880e57ce90507876a56886e957e515"

# The source repository does not ship a LICENSE file at the pinned commit.  This project
# therefore implements the published method independently and records behavioral
# correspondences; it does not copy the authors' source into the POI package.
IMPLEMENTATION_NOTES = {
    "embedding": "one bias-free linear projection per one-hot feature",
    "normalization": "unit-normalize every embedding column after embedding updates",
    "optimization": (
        "alternate clean downstream-classifier-only and adversarial embedding-only "
        "updates, matching the official emb_only/emb_fine roles"
    ),
    "numeric": "quantile-bin numerical variables before embedding (10 bins by default)",
    "transfer": (
        "freeze train-only learned embeddings, then apply paper Algorithm 3: global "
        "Euclidean-distance percentile, complete-linkage clustering at that maximum "
        "within-cluster distance, and cluster-mean embeddings before Random Forest"
    ),
    "poi_adaptation": (
        "multiclass region supervision and a portable relaxed cost-constrained PGD inner "
        "loop replace the paper repository's dataset-specific binary TabNet scripts; "
        "clean and adversarial updates reuse a minibatch sequentially rather than "
        "alternating separate minibatches"
    ),
    "projection_deviation": (
        "The portable Cat-PGD inner loop projects each field to its simplex and then "
        "contracts the joint proposal along the ray from the original point until its "
        "expected additive cost meets the budget. The paper uses iterative Dykstra "
        "projection onto the intersection of product simplices and a weighted L1 cost "
        "ball. Ray contraction is feasible but is not the Euclidean/Dykstra projection."
    ),
}
