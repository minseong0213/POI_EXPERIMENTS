"""Generate constraint-valid POI attacks against a train-only neural surrogate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .data import CATEGORICAL, FEATURES, NUMERIC, load_clean_splits, sha256
from .geospatial import AdministrativeBoundaryValidator


SURROGATE_ID = 'poi_mlp_surrogate_v1'
METRES_PER_LAT_DEGREE = 110_574.0
METRES_PER_LON_DEGREE = 111_320.0
ATTACK_SCHEMA_VERSION = 'poi-attack-generation-v2'
DISTANCE_METRIC = 'local_equirectangular_wgs84_approximation'
QUERY_COUNT_SCOPE = (
    'attack_search_candidate_forward_evaluations_per_poi; includes CAA routing; '
    'excludes common clean-baseline and final-record prediction evaluations'
)
PENDING_STATUS = 'generated_pending_fresh_independent_review'


class Surrogate(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, classes),
        )

    def forward(self, values):
        return self.network(values)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_rng_seed(generation_seed, split, condition):
    token = f'{generation_seed}:{split}:{condition}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'big') % (2 ** 31)


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def _atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.tmp{path.suffix}')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _validate_budget(value, name, *, integer=False, none_means_zero=False):
    implicit_zero = value is None
    if implicit_zero:
        if not none_means_zero:
            raise ValueError(f'{name} must be explicit')
        number = 0.0
    else:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be numeric, not boolean')
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must be finite and nonnegative') from error
    if not np.isfinite(number) or number < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    if integer and not number.is_integer():
        raise ValueError(f'{name} must be a nonnegative integer')
    return (int(number) if integer else number), implicit_zero


def distance_contract(config=None):
    config = config or {}
    metric = config.get('metric', DISTANCE_METRIC)
    if metric != DISTANCE_METRIC:
        raise ValueError(f'Unsupported distance metric: {metric}')
    numeric_tolerance, _ = _validate_budget(
        config.get('numeric_budget_tolerance_m', 0.05),
        'numeric_budget_tolerance_m')
    immutable_tolerance, _ = _validate_budget(
        config.get('categorical_only_tolerance_m', 1e-6),
        'categorical_only_tolerance_m')
    if immutable_tolerance > numeric_tolerance:
        raise ValueError('categorical-only tolerance cannot exceed numeric-budget tolerance')
    return {
        'metric': metric,
        'numeric_budget_tolerance_m': numeric_tolerance,
        'categorical_only_tolerance_m': immutable_tolerance,
        'earth_model': (
            'longitude scaled by cos(mean latitude) using 111320 m/degree; '
            'latitude scaled by 110574 m/degree'
        ),
    }


def numeric_frame(frame):
    result = frame[['POI_ID', 'region', 'split'] + FEATURES].copy()
    for column in FEATURES:
        result[column] = pd.to_numeric(result[column], errors='raise')
    return result


def encoder_from_train(train):
    category_values = {
        feature: sorted(train[feature].astype(float).unique().tolist())
        for feature in CATEGORICAL
    }
    region_values = sorted(train.region.unique().tolist())
    numeric_mean = train[NUMERIC].mean().to_numpy(dtype=np.float32)
    numeric_std = train[NUMERIC].std(ddof=0).replace(0, 1).to_numpy(dtype=np.float32)
    return {
        'category_values': category_values,
        'regions': region_values,
        'numeric_mean': numeric_mean,
        'numeric_std': numeric_std,
    }


def category_onehot(values, encoder):
    pieces = []
    for index, feature in enumerate(CATEGORICAL):
        vocabulary = encoder['category_values'][feature]
        lookup = {float(value): position for position, value in enumerate(vocabulary)}
        try:
            positions = np.array([lookup[float(value)] for value in values[:, index]])
        except KeyError as error:
            raise ValueError(f'Unknown category for {feature}: {error}') from error
        pieces.append(np.eye(len(vocabulary), dtype=np.float32)[positions])
    return np.concatenate(pieces, axis=1)


def encoded_features(coords, categories, encoder):
    scaled = ((coords.astype(np.float32) - encoder['numeric_mean']) /
              encoder['numeric_std'])
    return np.concatenate([scaled, category_onehot(categories, encoder)], axis=1)


def tensor_features(coords, category_vectors, encoder):
    mean = torch.as_tensor(encoder['numeric_mean'], device=coords.device)
    std = torch.as_tensor(encoder['numeric_std'], device=coords.device)
    return torch.cat([(coords - mean) / std, category_vectors], dim=-1)


def _serializable_encoder(encoder):
    return {
        'category_values': encoder['category_values'], 'regions': encoder['regions'],
        'numeric_mean': np.asarray(encoder['numeric_mean']).tolist(),
        'numeric_std': np.asarray(encoder['numeric_std']).tolist(),
    }


def train_surrogate(train, cfg, output, device, request_sha256=None):
    encoder = encoder_from_train(train)
    coords = train[NUMERIC].to_numpy(dtype=np.float32)
    categories = train[CATEGORICAL].to_numpy(dtype=np.float32)
    x = encoded_features(coords, categories, encoder)
    region_lookup = {region: index for index, region in enumerate(encoder['regions'])}
    y = np.array([region_lookup[region] for region in train.region], dtype=np.int64)

    surrogate_seed = cfg['surrogate_seed']
    rng = np.random.default_rng(surrogate_seed)
    train_indices, holdout_indices = [], []
    for label in range(len(encoder['regions'])):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * cfg['surrogate']['holdout_fraction'])))
        holdout_indices.extend(indices[:cut])
        train_indices.extend(indices[cut:])
    train_indices = np.asarray(train_indices)
    holdout_indices = np.asarray(holdout_indices)
    dataset = TensorDataset(torch.from_numpy(x[train_indices]), torch.from_numpy(y[train_indices]))
    generator = torch.Generator().manual_seed(surrogate_seed)
    loader = DataLoader(dataset, batch_size=cfg['surrogate']['batch_size'], shuffle=True,
                        generator=generator, num_workers=0)
    model = Surrogate(x.shape[1], len(encoder['regions'])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['surrogate']['learning_rate'],
        weight_decay=cfg['surrogate']['weight_decay'])
    holdout_x = torch.from_numpy(x[holdout_indices]).to(device)
    holdout_y = torch.from_numpy(y[holdout_indices]).to(device)
    best_state, best_loss, stale = None, float('inf'), 0
    history = []
    for epoch in range(cfg['surrogate']['epochs']):
        model.train()
        train_loss, seen = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch_x)
            seen += len(batch_x)
        model.eval()
        with torch.no_grad():
            logits = model(holdout_x)
            holdout_loss = float(F.cross_entropy(logits, holdout_y))
            holdout_accuracy = float(logits.argmax(1).eq(holdout_y).float().mean())
        history.append({'epoch': epoch + 1, 'train_loss': train_loss / seen,
                        'holdout_loss': holdout_loss,
                        'holdout_accuracy': holdout_accuracy})
        if holdout_loss < best_loss - 1e-5:
            best_loss = holdout_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg['surrogate']['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'surrogate.pt'
    encoder_path = output / 'surrogate_encoder.json'
    history_path = output / 'surrogate_training.csv'
    if checkpoint.exists() or encoder_path.exists() or history_path.exists():
        raise FileExistsError('Refusing to overwrite an existing surrogate artifact')
    train_ids_sha256 = _canonical_sha256(train.POI_ID.astype(str).tolist())
    _atomic_torch_save({
        'format': 'poi-surrogate-v2',
        'state_dict': best_state,
        'input_dim': x.shape[1],
        'classes': len(encoder['regions']),
        'surrogate_seed': int(cfg['surrogate_seed']),
        'generation_request_sha256': request_sha256,
        'train_ids_sha256': train_ids_sha256,
    }, checkpoint)
    history_temporary = history_path.with_name(f'.{history_path.name}.tmp')
    pd.DataFrame(history).to_csv(history_temporary, index=False)
    history_temporary.replace(history_path)
    _atomic_json(encoder_path, _serializable_encoder(encoder))
    return model, encoder, history[-1], checkpoint


def load_surrogate(train, cfg, output, device, request_sha256):
    """Load the immutable surrogate bound to a generation request."""
    output = Path(output)
    checkpoint = output / 'surrogate.pt'
    encoder_path = output / 'surrogate_encoder.json'
    history_path = output / 'surrogate_training.csv'
    missing = [str(path) for path in (checkpoint, encoder_path, history_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Incomplete persisted surrogate: {missing}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('format') != 'poi-surrogate-v2':
        raise ValueError('Unsupported or legacy surrogate checkpoint')
    if payload.get('generation_request_sha256') != request_sha256:
        raise ValueError('Surrogate checkpoint belongs to a different generation request')
    if payload.get('surrogate_seed') != int(cfg['surrogate_seed']):
        raise ValueError('Surrogate checkpoint seed differs from configuration')
    expected_train_hash = _canonical_sha256(train.POI_ID.astype(str).tolist())
    if payload.get('train_ids_sha256') != expected_train_hash:
        raise ValueError('Surrogate checkpoint train IDs differ from current data')
    expected_encoder = encoder_from_train(train)
    stored_encoder = json.loads(encoder_path.read_text())
    if _canonical_sha256(stored_encoder) != _canonical_sha256(
            _serializable_encoder(expected_encoder)):
        raise ValueError('Persisted surrogate encoder differs from current train data')
    if (payload.get('input_dim') != encoded_features(
            train[NUMERIC].to_numpy(dtype=np.float32),
            train[CATEGORICAL].to_numpy(dtype=np.float32), expected_encoder).shape[1]
            or payload.get('classes') != len(expected_encoder['regions'])):
        raise ValueError('Surrogate checkpoint dimensions differ from current train data')
    model = Surrogate(payload['input_dim'], payload['classes']).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError('Persisted surrogate training history is empty')
    return model, expected_encoder, history.iloc[-1].to_dict(), checkpoint


def true_margin(logits, labels):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return masked.max(1).values - true


def offsets_to_coords(base, offsets):
    latitude = base[..., 1]
    lon_scale = METRES_PER_LON_DEGREE * torch.cos(torch.deg2rad(latitude)).clamp_min(.1)
    longitude = base[..., 0] + offsets[..., 0] / lon_scale
    new_latitude = latitude + offsets[..., 1] / METRES_PER_LAT_DEGREE
    return torch.stack([longitude, new_latitude], dim=-1)


def project_l2(offsets, budget):
    budget, _ = _validate_budget(budget, 'numeric_budget_m')
    norm = offsets.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return offsets * torch.clamp(budget / norm, max=1.0)


def numeric_attack_batch(method, base, categories, labels, model, encoder, budget, cfg):
    budget, _ = _validate_budget(budget, 'numeric_budget_m')
    if method not in {'fgsm', 'pgd', 'cw_l2', 'capgd'}:
        raise ValueError(f'Unknown numeric attack: {method}')
    onehot = torch.from_numpy(category_onehot(categories.cpu().numpy(), encoder)).to(base.device)
    steps = int(cfg['attacks'][method]['steps'])
    if steps <= 0:
        raise ValueError(f'{method} steps must be positive')
    if method == 'fgsm':
        delta = torch.zeros_like(base, requires_grad=True)
        loss = F.cross_entropy(model(tensor_features(offsets_to_coords(base, delta), onehot, encoder)), labels)
        gradient = torch.autograd.grad(loss, delta)[0]
        delta = project_l2(gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12) * budget, budget)
        return offsets_to_coords(base, delta.detach()), np.full(len(base), 1)

    if method == 'cw_l2':
        delta = torch.zeros_like(base, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=float(budget) / max(steps // 3, 1))
        best = delta.detach().clone()
        best_quality = torch.full((len(base),), -torch.inf, device=base.device)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor_features(offsets_to_coords(base, delta), onehot, encoder))
            margin = true_margin(logits, labels)
            distance = delta.norm(dim=1) / max(budget, 1e-12)
            with torch.no_grad():
                # Score and save the same pre-update state.  The previous implementation
                # scored this state and stored the post-optimizer delta under its quality.
                quality = torch.where(margin.gt(0), 1000 - distance, margin)
                improved = quality > best_quality
                best[improved] = delta.detach()[improved]
                best_quality[improved] = quality[improved]
            loss = (F.relu(-margin + cfg['attacks'][method]['kappa']) +
                    cfg['attacks'][method]['distance_weight'] * distance).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(project_l2(delta, budget))
        return offsets_to_coords(base, best), np.full(len(base), steps)

    delta = torch.zeros_like(base)
    previous = delta.clone()
    best = delta.clone()
    best_loss = torch.full((len(base),), -torch.inf, device=base.device)
    step = torch.full((len(base), 1),
                      2 * float(budget) / max(steps, 1), device=base.device)
    stagnant = torch.zeros(len(base), dtype=torch.int64, device=base.device)
    if method == 'pgd':
        direction = torch.randn_like(delta)
        radii = torch.rand((len(base), 1), device=base.device).sqrt() * budget
        delta = project_l2(direction, 1.0) * radii
        previous = delta.clone()
    if method == 'capgd':
        direction = torch.randn_like(delta)
        radii = torch.rand((len(base), 1), device=base.device).sqrt() * budget
        delta = project_l2(direction, 1.0) * radii
        previous = delta.clone()
        step.fill_(2 * float(budget))
    for iteration in range(steps):
        delta = delta.detach().requires_grad_(True)
        logits = model(tensor_features(offsets_to_coords(base, delta), onehot, encoder))
        losses = F.cross_entropy(logits, labels, reduction='none')
        gradient = torch.autograd.grad(losses.sum(), delta)[0]
        direction = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
        candidate = delta.detach() + step * direction
        if method == 'capgd':
            candidate = candidate + .75 * (delta.detach() - previous)
        candidate = project_l2(candidate, budget)
        improved = losses.detach() > best_loss
        best[improved] = delta.detach()[improved]
        best_loss[improved] = losses.detach()[improved]
        stagnant = torch.where(improved, torch.zeros_like(stagnant), stagnant + 1)
        if method == 'capgd' and (iteration + 1) % max(steps // 5, 1) == 0:
            reduce = stagnant.ge(max(steps // 10, 2))
            step[reduce] *= .5
            candidate[reduce] = best[reduce]
            stagnant[reduce] = 0
        previous, delta = delta.detach(), candidate.detach()
    logits = model(tensor_features(offsets_to_coords(base, delta), onehot, encoder))
    losses = F.cross_entropy(logits, labels, reduction='none')
    improved = losses > best_loss
    best[improved] = delta[improved]
    # One candidate forward occurs per optimization step plus the final candidate
    # search evaluation above.  Common final reporting inference is excluded.
    return offsets_to_coords(base, best), np.full(len(base), steps + 1)


def numeric_attack(method, coords, categories, labels, model, encoder, budget, cfg, device):
    outputs, queries = [], []
    batch_size = cfg['attack_batch_size']
    for start in range(0, len(coords), batch_size):
        stop = min(start + batch_size, len(coords))
        base = torch.from_numpy(coords[start:stop]).to(device=device, dtype=torch.float32)
        cats = torch.from_numpy(categories[start:stop]).to(device)
        y = torch.from_numpy(labels[start:stop]).to(device)
        attacked, used = numeric_attack_batch(
            method, base, cats, y, model, encoder, budget, cfg)
        outputs.append(attacked.detach().cpu().numpy())
        queries.append(used)
    return np.concatenate(outputs), np.concatenate(queries)


def valid_tuples(train, encoder):
    tuples = (train[CATEGORICAL].drop_duplicates().sort_values(CATEGORICAL)
              .to_numpy(dtype=np.float32))
    return tuples, category_onehot(tuples, encoder)


def categorical_exact(coords, categories, labels, model, encoder, tuples, tuple_vectors,
                      budget, cfg, device):
    budget, _ = _validate_budget(
        budget, 'categorical_budget_l0', integer=True)
    selected, queries = [], []
    batch_size = max(16, cfg['attack_batch_size'] // 4)
    tuple_tensor = torch.from_numpy(tuple_vectors).to(device)
    for start in range(0, len(coords), batch_size):
        stop = min(start + batch_size, len(coords))
        base_coords = torch.from_numpy(coords[start:stop]).to(device=device, dtype=torch.float32)
        base_categories = categories[start:stop]
        y = torch.from_numpy(labels[start:stop]).to(device)
        changes = (base_categories[:, None, :] != tuples[None, :, :]).sum(axis=2)
        allowed = changes <= budget
        b, candidates = allowed.shape
        expanded_coords = base_coords[:, None, :].expand(b, candidates, 2)
        expanded_vectors = tuple_tensor[None, :, :].expand(b, candidates, -1)
        with torch.no_grad():
            logits = model(tensor_features(
                expanded_coords.reshape(-1, 2), expanded_vectors.reshape(-1, tuple_tensor.shape[1]),
                encoder)).reshape(b, candidates, -1)
            repeated_y = y[:, None].expand(b, candidates).reshape(-1)
            margins = true_margin(logits.reshape(-1, logits.shape[-1]), repeated_y).reshape(b, candidates)
            margins[~torch.from_numpy(allowed).to(device)] = -torch.inf
            choice = margins.argmax(1).cpu().numpy()
        selected.append(tuples[choice])
        # The dense implementation evaluates every tuple and masks disallowed results
        # afterwards, so the exact forward-evaluation count is the full tuple count.
        queries.append(np.full(b, candidates, dtype=np.int64))
    return np.concatenate(selected), np.concatenate(queries)


def categorical_pcaa(coords, categories, labels, model, encoder, tuples, tuple_vectors,
                     budget, cfg, device):
    budget, _ = _validate_budget(
        budget, 'categorical_budget_l0', integer=True)
    selected, queries = [], []
    batch_size = max(16, cfg['attack_batch_size'] // 4)
    tuple_tensor = torch.from_numpy(tuple_vectors).to(device)
    steps = cfg['attacks']['pcaa']['steps']
    draws = cfg['attacks']['pcaa']['candidate_draws']
    for start in range(0, len(coords), batch_size):
        stop = min(start + batch_size, len(coords))
        base_coords = torch.from_numpy(coords[start:stop]).to(device=device, dtype=torch.float32)
        base_categories = categories[start:stop]
        y = torch.from_numpy(labels[start:stop]).to(device)
        changes = (base_categories[:, None, :] != tuples[None, :, :]).sum(axis=2)
        allowed = torch.from_numpy(changes <= budget).to(device)
        logits_parameters = torch.zeros(allowed.shape, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([logits_parameters], lr=cfg['attacks']['pcaa']['learning_rate'])
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            probabilities = torch.softmax(logits_parameters.masked_fill(~allowed, -1e9), dim=1)
            soft_categories = probabilities @ tuple_tensor
            output = model(tensor_features(base_coords, soft_categories, encoder))
            entropy = -(probabilities.clamp_min(1e-9).log() * probabilities).sum(1).mean()
            loss = -F.cross_entropy(output, y) + cfg['attacks']['pcaa']['entropy_weight'] * entropy
            loss.backward()
            optimizer.step()
        probabilities = torch.softmax(logits_parameters.masked_fill(~allowed, -1e9), dim=1)
        k = min(draws, tuple_tensor.shape[0])
        choices = probabilities.topk(k, dim=1).indices
        candidate_vectors = tuple_tensor[choices]
        candidate_coords = base_coords[:, None, :].expand(-1, k, -1)
        with torch.no_grad():
            output = model(tensor_features(candidate_coords.reshape(-1, 2),
                                            candidate_vectors.reshape(-1, tuple_tensor.shape[1]),
                                            encoder)).reshape(len(base_coords), k, -1)
            margins = true_margin(output.reshape(-1, output.shape[-1]),
                                  y[:, None].expand(-1, k).reshape(-1)).reshape(len(base_coords), k)
            hard_allowed = allowed.gather(1, choices)
            margins = margins.masked_fill(~hard_allowed, -torch.inf)
            best = margins.argmax(1)
            final_choices = choices[torch.arange(len(base_coords), device=device), best].cpu().numpy()
        selected.append(tuples[final_choices])
        queries.append(np.full(len(base_coords), steps + k))
    return np.concatenate(selected), np.concatenate(queries)


def random_valid_choices(allowed, population, generator):
    random_values = torch.rand((allowed.shape[0], population, allowed.shape[1]),
                               device=allowed.device, generator=generator)
    random_values.masked_fill_(~allowed[:, None, :], -1)
    return random_values.argmax(2)


def mixed_evolution(coords, categories, labels, model, encoder, tuples, tuple_vectors,
                    numeric_budget, categorical_budget, cfg, device, seed):
    numeric_budget, _ = _validate_budget(numeric_budget, 'numeric_budget_m')
    categorical_budget, _ = _validate_budget(
        categorical_budget, 'categorical_budget_l0', integer=True)
    if numeric_budget == 0 or categorical_budget == 0:
        raise ValueError('Mixed evolution requires positive numeric and categorical budgets')
    outputs_coords, outputs_categories, query_rows = [], [], []
    settings = cfg['attacks']['moeva']
    population, generations = settings['population'], settings['generations']
    batch_size = settings['batch_size']
    tuple_tensor = torch.from_numpy(tuple_vectors).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, len(coords), batch_size):
        stop = min(start + batch_size, len(coords))
        base = torch.from_numpy(coords[start:stop]).to(device=device, dtype=torch.float32)
        base_categories = categories[start:stop]
        y = torch.from_numpy(labels[start:stop]).to(device)
        change_counts = torch.from_numpy(
            (base_categories[:, None, :] != tuples[None, :, :]).sum(axis=2)).to(device)
        allowed = change_counts <= categorical_budget
        b = len(base)
        tuple_indices = random_valid_choices(allowed, population, generator)
        original = torch.from_numpy((change_counts.cpu().numpy() == 0).argmax(axis=1)).to(device)
        tuple_indices[:, 0] = original
        offsets = torch.randn((b, population, 2), device=device, generator=generator)
        offsets = project_l2(offsets, 1.0)
        radii = torch.rand((b, population, 1), device=device, generator=generator).sqrt()
        offsets = offsets * radii * numeric_budget
        offsets[:, 0] = 0
        best_quality = torch.full((b,), -torch.inf, device=device)
        best_offsets = torch.zeros((b, 2), device=device)
        best_tuples = original.clone()
        keep = max(2, population // 2)
        for generation in range(generations):
            attacked_coords = offsets_to_coords(base[:, None, :].expand(-1, population, -1), offsets)
            category_vectors = tuple_tensor[tuple_indices]
            with torch.no_grad():
                logits = model(tensor_features(attacked_coords.reshape(-1, 2),
                                               category_vectors.reshape(-1, tuple_tensor.shape[1]),
                                               encoder)).reshape(b, population, -1)
                margins = true_margin(logits.reshape(-1, logits.shape[-1]),
                                      y[:, None].expand(-1, population).reshape(-1)).reshape(b, population)
            distance_cost = offsets.norm(dim=2) / float(numeric_budget)
            category_cost = change_counts.gather(1, tuple_indices) / float(categorical_budget)
            combined_cost = (distance_cost + category_cost) / 2
            quality = torch.where(margins.gt(0), 1000 - combined_cost, margins)
            generation_quality, generation_best = quality.max(1)
            improved = generation_quality > best_quality
            rows = torch.arange(b, device=device)
            best_offsets[improved] = offsets[rows, generation_best][improved]
            best_tuples[improved] = tuple_indices[rows, generation_best][improved]
            best_quality[improved] = generation_quality[improved]

            tradeoff = torch.rand((b, population), device=device, generator=generator) * .2
            selection_score = margins - tradeoff * combined_cost
            parents = selection_score.topk(keep, dim=1).indices
            parent_offsets = offsets.gather(1, parents[:, :, None].expand(-1, -1, 2))
            parent_tuples = tuple_indices.gather(1, parents)
            first = torch.randint(0, keep, (b, population), device=device, generator=generator)
            second = torch.randint(0, keep, (b, population), device=device, generator=generator)
            gather_offset = lambda source, index: source.gather(1, index[:, :, None].expand(-1, -1, 2))
            child_offsets = (gather_offset(parent_offsets, first) +
                             gather_offset(parent_offsets, second)) / 2
            noise_scale = numeric_budget * settings['mutation_scale'] * (1 - generation / generations)
            child_offsets += torch.randn(child_offsets.shape, device=device, generator=generator) * noise_scale
            offsets = project_l2(child_offsets, numeric_budget)
            parent_choices = parent_tuples.gather(1, first)
            mutations = random_valid_choices(allowed, population, generator)
            mutate = torch.rand((b, population), device=device, generator=generator) < settings['category_mutation']
            tuple_indices = torch.where(mutate, mutations, parent_choices)
            offsets[:, 0] = best_offsets
            tuple_indices[:, 0] = best_tuples
        outputs_coords.append(offsets_to_coords(base, best_offsets).cpu().numpy())
        outputs_categories.append(tuples[best_tuples.cpu().numpy()])
        query_rows.append(np.full(b, population * generations))
    return (np.concatenate(outputs_coords), np.concatenate(outputs_categories),
            np.concatenate(query_rows))


def predict(model, coords, categories, encoder, device, batch_size):
    predictions, margins = [], []
    for start in range(0, len(coords), batch_size):
        stop = min(start + batch_size, len(coords))
        x = torch.from_numpy(encoded_features(coords[start:stop], categories[start:stop], encoder)).to(device)
        with torch.no_grad():
            logits = model(x)
        predictions.append(logits.argmax(1).cpu().numpy())
        margins.append(logits.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(margins)


def distance_metres(clean, attacked):
    clean = np.asarray(clean, dtype=np.float64)
    attacked = np.asarray(attacked, dtype=np.float64)
    if clean.shape != attacked.shape or clean.ndim != 2 or clean.shape[1] != 2:
        raise ValueError('Coordinate arrays must be matching N x 2 matrices')
    mean_latitude = np.deg2rad((clean[:, 1] + attacked[:, 1]) / 2)
    dx = (attacked[:, 0] - clean[:, 0]) * METRES_PER_LON_DEGREE * np.cos(mean_latitude)
    dy = (attacked[:, 1] - clean[:, 1]) * METRES_PER_LAT_DEGREE
    return np.sqrt(dx ** 2 + dy ** 2)


def enforce_numeric_budget(clean, attacked, budget):
    """Project final floating-point coordinates inside the geodesic approximation."""
    budget, _ = _validate_budget(budget, 'numeric_budget_m')
    result = attacked.astype(np.float64, copy=True)
    origin = clean.astype(np.float64, copy=False)
    if result.shape != origin.shape or result.ndim != 2 or result.shape[1] != 2:
        raise ValueError('Coordinate arrays must be matching N x 2 matrices')
    if not np.isfinite(result).all() or not np.isfinite(origin).all():
        raise ValueError('Cannot project non-finite coordinates')
    for _ in range(3):
        distance = distance_metres(origin, result)
        scale = np.minimum(1.0, (budget * (1 - 1e-7)) /
                           np.maximum(distance, 1e-12))
        result = origin + (result - origin) * scale[:, None]
    return result


def _quarantine_reasons(clean_boundary, attack_boundary, numeric_range_valid,
                        category_codebook_valid, clean_combo_valid,
                        attack_combo_valid, within_budget):
    reasons = np.full(len(numeric_range_valid), '', dtype=object)
    checks = [
        (~clean_boundary['matches'],
         np.char.add('clean_label_', clean_boundary['status'].astype(str))),
        (~attack_boundary['matches'],
         np.char.add('attack_label_', attack_boundary['status'].astype(str))),
        (~numeric_range_valid, np.full(len(reasons), 'numeric_outside_train_range')),
        (~category_codebook_valid, np.full(len(reasons), 'category_outside_train_codebook')),
        (~clean_combo_valid,
         np.full(len(reasons), 'clean_category_tuple_not_observed_in_train')),
        (~attack_combo_valid,
         np.full(len(reasons), 'attack_category_tuple_not_observed_in_train')),
        (~within_budget, np.full(len(reasons), 'attack_budget_exceeded')),
    ]
    for failed, label in checks:
        append = failed & (reasons != '')
        reasons[append] = np.char.add(np.char.add(reasons[append].astype(str), ';'),
                                      label[append].astype(str))
        first = failed & (reasons == '')
        reasons[first] = label[first]
    return reasons


def evaluate_constraints(frame, attacked_coords, attacked_categories, numeric_budget,
                         categorical_budget, tuples, numeric_bounds, encoder,
                         boundary_validator, clean_boundary, distance_validation=None):
    """Recompute every final-state eligibility field from train-fitted constraints."""
    contract = distance_contract(distance_validation)
    numeric_limit, implicit_zero = _validate_budget(
        numeric_budget, 'numeric_budget_m', none_means_zero=True)
    categorical_limit, _ = _validate_budget(
        categorical_budget, 'categorical_budget_l0', integer=True,
        none_means_zero=True)
    clean_coords = frame[NUMERIC].to_numpy(dtype=np.float64)
    clean_categories = frame[CATEGORICAL].to_numpy(dtype=np.float32)
    attacked_coords = np.asarray(attacked_coords, dtype=np.float64)
    attacked_categories = np.asarray(attacked_categories, dtype=np.float32)
    if attacked_coords.shape != clean_coords.shape:
        raise ValueError('Attacked coordinates do not align with source rows')
    if attacked_categories.shape != clean_categories.shape:
        raise ValueError('Attacked categories do not align with source rows')
    finite_coordinates = np.isfinite(attacked_coords).all(axis=1)
    finite_categories = np.isfinite(attacked_categories).all(axis=1)
    distances = distance_metres(clean_coords, attacked_coords)
    changed = clean_categories != attacked_categories
    changed_l0 = changed.sum(axis=1)
    valid_set = {tuple(row) for row in np.asarray(tuples, dtype=np.float32).tolist()}
    # Every final tuple, including an unchanged validation tuple, must have been
    # observed in clean train.  There is no official hierarchy rule that permits an
    # unseen combination to bypass the train-only tuple constraint.
    clean_combo_valid = np.array(
        [tuple(row) in valid_set for row in clean_categories])
    attack_combo_valid = finite_categories & np.array(
        [tuple(row) in valid_set for row in attacked_categories])
    combo_valid = clean_combo_valid & attack_combo_valid
    lower = np.asarray(numeric_bounds['min'], dtype=np.float64)
    upper = np.asarray(numeric_bounds['max'], dtype=np.float64)
    if (lower.shape != (2,) or upper.shape != (2,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or (lower > upper).any()):
        raise ValueError('Invalid train-only numeric bounds')
    numeric_range_valid = finite_coordinates & (
        (attacked_coords >= lower) & (attacked_coords <= upper)).all(axis=1)
    category_codebook_valid = finite_categories.copy()
    for index, feature in enumerate(CATEGORICAL):
        vocabulary = np.asarray(encoder['category_values'][feature], dtype=float)
        if not len(vocabulary) or not np.isfinite(vocabulary).all():
            raise ValueError(f'Invalid train category codebook: {feature}')
        category_codebook_valid &= np.isin(attacked_categories[:, index], vocabulary)
    tolerance = (contract['categorical_only_tolerance_m'] if implicit_zero else
                 contract['numeric_budget_tolerance_m'])
    distance_valid = np.isfinite(distances) & (distances <= numeric_limit + tolerance)
    categorical_valid = changed_l0 <= categorical_limit
    within_budget = distance_valid & categorical_valid
    attack_boundary = boundary_validator.validate(
        attacked_coords, frame.region.to_numpy())
    label_preserved = clean_boundary['matches'] & attack_boundary['matches']
    quarantine_reason = _quarantine_reasons(
        clean_boundary, attack_boundary, numeric_range_valid,
        category_codebook_valid, clean_combo_valid, attack_combo_valid,
        within_budget)
    constraints_valid = (numeric_range_valid & category_codebook_valid & combo_valid &
                         within_budget & label_preserved)
    return {
        'clean_coords': clean_coords,
        'clean_categories': clean_categories,
        'attacked_coords': attacked_coords,
        'attacked_categories': attacked_categories,
        'distances': distances,
        'changed': changed,
        'changed_l0': changed_l0,
        'numeric_budget_m': numeric_limit,
        'numeric_budget_was_implicit_zero': implicit_zero,
        'categorical_budget_l0': categorical_limit,
        'distance_tolerance_m': tolerance,
        'numeric_range_valid': numeric_range_valid,
        'category_codebook_valid': category_codebook_valid,
        'clean_combo_valid': clean_combo_valid,
        'attack_combo_valid': attack_combo_valid,
        'combo_valid': combo_valid,
        'within_budget': within_budget,
        'attack_boundary': attack_boundary,
        'label_preserved': label_preserved,
        'quarantine_reason': quarantine_reason,
        'constraints_valid': constraints_valid,
        'distance_contract': contract,
    }


def records(frame, attacked_coords, attacked_categories, labels, clean_pred, attack_pred,
            method, condition, numeric_budget, categorical_budget, queries, tuples,
            generation_seed, surrogate_seed, numeric_bounds, encoder, boundary_validator,
            clean_boundary, distance_validation=None, generation_identity_sha256='',
            surrogate_checkpoint_sha256=''):
    evaluated = evaluate_constraints(
        frame, attacked_coords, attacked_categories, numeric_budget,
        categorical_budget, tuples, numeric_bounds, encoder, boundary_validator,
        clean_boundary, distance_validation)
    clean_coords = evaluated['clean_coords']
    clean_categories = evaluated['clean_categories']
    attacked_coords = evaluated['attacked_coords']
    attacked_categories = evaluated['attacked_categories']
    distances = evaluated['distances']
    changed = evaluated['changed']
    changed_l0 = evaluated['changed_l0']
    numeric_range_valid = evaluated['numeric_range_valid']
    category_codebook_valid = evaluated['category_codebook_valid']
    combo_valid = evaluated['combo_valid']
    within_budget = evaluated['within_budget']
    attack_boundary = evaluated['attack_boundary']
    label_preserved = evaluated['label_preserved']
    quarantine_reason = evaluated['quarantine_reason']
    constraints_valid = evaluated['constraints_valid']
    queries = np.asarray(queries)
    if (queries.shape != (len(frame),) or not np.isfinite(queries).all() or
            (queries < 0).any() or not np.equal(queries, np.floor(queries)).all()):
        raise ValueError('query_count must be an aligned nonnegative integer vector')
    predicted_regions = np.asarray(encoder['regions'], dtype=object)
    output = pd.DataFrame({
        'POI_ID': frame.POI_ID.to_numpy(), 'split': frame.split.to_numpy(),
        'region': frame.region.to_numpy(), 'attack_method': method,
        'attack_condition': condition, 'source_model': SURROGATE_ID,
        'victim_model': SURROGATE_ID,
        'victim_evaluation_scope': 'direct_white_box_source_model',
        'transfer_evaluated': False, 'targeted': False, 'target_region': '',
        'numeric_budget_m': evaluated['numeric_budget_m'],
        'numeric_budget_was_implicit_zero': evaluated['numeric_budget_was_implicit_zero'],
        'categorical_budget_l0': evaluated['categorical_budget_l0'],
        'distance_m': distances, 'categorical_l0': changed_l0,
        'distance_metric': evaluated['distance_contract']['metric'],
        'distance_tolerance_m': evaluated['distance_tolerance_m'],
        'attack_attempted': True,
        'attack_changed_any': (distances > 1e-6) | (changed_l0 > 0),
        'valid_range': numeric_range_valid & category_codebook_valid,
        'valid_numeric_train_range': numeric_range_valid,
        'valid_category_codebook': category_codebook_valid,
        'valid_clean_category_combo': evaluated['clean_combo_valid'],
        'valid_attack_category_combo': evaluated['attack_combo_valid'],
        'valid_category_combo': combo_valid, 'within_budget': within_budget,
        'constraints_valid': constraints_valid,
        'primary_evaluation_eligible': constraints_valid,
        'label_preserved': label_preserved,
        'clean_boundary_status': clean_boundary['status'],
        'attack_boundary_status': attack_boundary['status'],
        'clean_polygon_region': clean_boundary['consensus_region'],
        'attack_polygon_region': attack_boundary['consensus_region'],
        'quarantine_reason': quarantine_reason,
        'clean_source_pred': clean_pred,
        'clean_source_pred_region': predicted_regions[clean_pred],
        'attack_source_pred': attack_pred, 'clean_source_correct': clean_pred == labels,
        'attack_success_source': (clean_pred == labels) & (attack_pred != labels),
        'clean_victim_pred': clean_pred,
        'clean_victim_pred_region': predicted_regions[clean_pred],
        'attack_victim_pred': attack_pred,
        'attack_victim_pred_region': predicted_regions[attack_pred],
        'clean_victim_correct': clean_pred == labels,
        'attack_success_victim': (clean_pred == labels) & (attack_pred != labels),
        'query_count': queries.astype(np.int64), 'query_count_scope': QUERY_COUNT_SCOPE,
        'generation_seed': generation_seed,
        'attack_rng_seed': condition_rng_seed(
            generation_seed, str(frame.split.iloc[0]), condition),
        'surrogate_seed': surrogate_seed,
        'attack_schema_version': ATTACK_SCHEMA_VERSION,
        'generation_identity_sha256': generation_identity_sha256,
        'surrogate_checkpoint_sha256': surrogate_checkpoint_sha256,
    })
    for source_id, values in clean_boundary['source_labels'].items():
        output[f'clean_polygon_region_{source_id}'] = values
    for source_id, values in attack_boundary['source_labels'].items():
        output[f'attack_polygon_region_{source_id}'] = values
    for index, feature in enumerate(NUMERIC):
        output[f'{feature}_clean'] = clean_coords[:, index]
        output[f'{feature}_adv'] = attacked_coords[:, index]
        output[f'changed_{feature.lower()}'] = np.abs(clean_coords[:, index] - attacked_coords[:, index]) > 1e-12
    for index, feature in enumerate(CATEGORICAL):
        output[f'{feature}_clean'] = clean_categories[:, index]
        output[f'{feature}_adv'] = attacked_categories[:, index]
        output[f'changed_{feature.lower()}'] = changed[:, index]
    return output


def save_figure(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def attack_condition_specs(cfg):
    specs = []
    for method in ('fgsm', 'pgd', 'cw_l2', 'capgd'):
        for value in cfg['numeric_budgets_m']:
            budget, _ = _validate_budget(value, 'numeric_budget_m')
            specs.append({'method': method, 'condition': f'{method}_m{value}',
                          'numeric_budget_m': budget, 'categorical_budget_l0': 0})
    for method in ('category_exact', 'pcaa'):
        for value in cfg['categorical_budgets_l0']:
            budget, _ = _validate_budget(
                value, 'categorical_budget_l0', integer=True)
            specs.append({'method': method, 'condition': f'{method}_l{value}',
                          'numeric_budget_m': None, 'categorical_budget_l0': budget})
    for level in cfg['mixed_budgets']:
        numeric, _ = _validate_budget(level['numeric_m'], 'numeric_budget_m')
        categorical, _ = _validate_budget(
            level['categorical_l0'], 'categorical_budget_l0', integer=True)
        if numeric == 0 or categorical == 0:
            raise ValueError('Mixed budgets must be positive')
        for method in ('moeva', 'caa'):
            specs.append({'method': method, 'condition': f"{method}_{level['name']}",
                          'numeric_budget_m': numeric,
                          'categorical_budget_l0': categorical})
    names = [spec['condition'] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError('Attack condition names must be unique')
    return specs


def generation_request(config_path, cfg, data_dir, clean, boundary_config_path,
                       boundary_validator):
    """Build the immutable pre-checkpoint identity before any shard is generated."""
    code_files = [Path(__file__), Path(__file__).with_name('data.py'),
                  Path(__file__).with_name('geospatial.py')]
    dataset_files = ['poi_data_region.csv', 'sample_manifest.csv', 'report.json']
    dataset_hashes = {
        name: sha256(Path(data_dir) / name)
        for name in dataset_files if (Path(data_dir) / name).is_file()
    }
    split_identity = {}
    for split in cfg['splits']:
        rows = clean.loc[clean.split.eq(split), ['POI_ID', 'region'] + FEATURES]
        split_identity[split] = {
            'rows': len(rows),
            'ordered_ids_sha256': _canonical_sha256(rows.POI_ID.astype(str).tolist()),
            'ordered_id_region_feature_sha256': hashlib.sha256(
                rows.to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest(),
        }
    provenance = boundary_validator.provenance()
    specs = attack_condition_specs(cfg)
    return {
        'format': 'poi-generation-request-v2',
        'attack_schema_version': ATTACK_SCHEMA_VERSION,
        'config_path': str(Path(config_path).resolve()),
        'config_sha256': sha256(config_path),
        'config_canonical_sha256': _canonical_sha256(cfg),
        'data_dir': str(Path(data_dir).resolve()),
        'dataset_sha256': dataset_hashes,
        'split_identity': split_identity,
        'features': FEATURES,
        'splits': list(cfg['splits']),
        'surrogate_seed': int(cfg['surrogate_seed']),
        'generation_seeds': [int(value) for value in cfg['generation_seeds']],
        'conditions': specs,
        'expected_shards': len(specs) * len(cfg['splits']) * len(cfg['generation_seeds']),
        'expected_rows': len(clean) * len(specs) * len(cfg['generation_seeds']),
        'distance_validation': distance_contract(cfg.get('distance_validation')),
        'boundary_config_path': str(Path(boundary_config_path).resolve()),
        'boundary_config_sha256': sha256(boundary_config_path),
        'boundary_validation': provenance,
        'code_sha256': {str(path.resolve()): sha256(path) for path in code_files},
        'query_count_scope': QUERY_COUNT_SCOPE,
        'test_feature_access': {
            'current_generation_run_materializes_test_features': False,
            'known_prior_exploratory_whole_34000_coordinate_access': True,
            'historically_pristine_test_claim_permitted': False,
            'note': (
                'A prior exploratory process materialized coordinates for all 34,000 rows, '
                'including test. This run is prospectively train/validation-only and does '
                'not restore historical test isolation.'
            ),
        },
        'boundary_limitations': {
            'original_poi_label_boundary_vintage': 'unknown_not_supplied',
            'source_status': (
                'two checksummed third-party boundary derivatives; official original bytes, '
                'source independence, transformation chain, and label-vintage match are not '
                'independently authenticated'
            ),
        },
    }


def _ensure_immutable_json(path, expected, description):
    path = Path(path)
    if path.exists():
        observed = json.loads(path.read_text())
        if _canonical_sha256(observed) != _canonical_sha256(expected):
            raise ValueError(f'Existing {description} differs from the current run')
        return observed
    _atomic_json(path, expected)
    return expected


def _generation_identity(request, checkpoint, encoder_path, history_path):
    value = {
        'format': 'poi-generation-identity-v2',
        'generation_request_sha256': _canonical_sha256(request),
        'surrogate_checkpoint_sha256': sha256(checkpoint),
        'surrogate_encoder_sha256': sha256(encoder_path),
        'surrogate_training_history_sha256': sha256(history_path),
    }
    value['generation_identity_sha256'] = _canonical_sha256(value)
    return value


def _load_shard_manifest(path, identity_sha256, shards_directory):
    path = Path(path)
    existing_shards = sorted(Path(shards_directory).glob('*.parquet'))
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get('format') != 'poi-shard-manifest-v2' or
                manifest.get('generation_identity_sha256') != identity_sha256 or
                not isinstance(manifest.get('shards'), dict)):
            raise ValueError('Shard manifest differs from the immutable generation identity')
        unregistered = [item.name for item in existing_shards
                        if item.name not in manifest['shards']]
        missing = [name for name in manifest['shards']
                   if not (Path(shards_directory) / name).is_file()]
        if unregistered or missing:
            raise ValueError(
                f'Shard manifest/file mismatch: unregistered={unregistered}, missing={missing}')
        return manifest
    if existing_shards:
        raise ValueError('Existing attack shards have no immutable shard manifest')
    manifest = {
        'format': 'poi-shard-manifest-v2',
        'generation_identity_sha256': identity_sha256,
        'shards': {},
    }
    _atomic_json(path, manifest)
    return manifest


def _query_count_is_valid(frame, method, cfg, tuples):
    counts = frame['query_count'].to_numpy(dtype=np.int64)
    if method == 'fgsm':
        expected = 1
    elif method == 'cw_l2':
        expected = int(cfg['attacks']['cw_l2']['steps'])
    elif method in {'pgd', 'capgd'}:
        expected = int(cfg['attacks'][method]['steps']) + 1
    elif method == 'category_exact':
        expected = len(tuples)
    elif method == 'pcaa':
        expected = (int(cfg['attacks']['pcaa']['steps']) +
                    min(int(cfg['attacks']['pcaa']['candidate_draws']), len(tuples)))
    elif method == 'moeva':
        expected = (int(cfg['attacks']['moeva']['population']) *
                    int(cfg['attacks']['moeva']['generations']))
    elif method == 'caa':
        base = int(cfg['attacks']['capgd']['steps']) + 2  # final search + routing
        evolution = (int(cfg['attacks']['moeva']['population']) *
                     int(cfg['attacks']['moeva']['generations']))
        return np.isin(counts, [base, base + evolution]).all()
    else:
        return False
    return np.equal(counts, expected).all()


def validate_shard_frame(shard, source_frame, spec, generation_seed, cfg, tuples,
                         numeric_bounds, encoder, boundary_validator, clean_boundary,
                         identity):
    required = {
        'POI_ID', 'region', 'split', 'attack_method', 'attack_condition',
        'generation_seed', 'generation_identity_sha256',
        'surrogate_checkpoint_sha256', 'attack_schema_version', 'query_count',
        'query_count_scope', 'constraints_valid', 'quarantine_reason',
        'numeric_budget_m', 'numeric_budget_was_implicit_zero',
        'categorical_budget_l0', 'distance_metric', 'distance_tolerance_m',
        'source_model', 'victim_model', 'victim_evaluation_scope',
        'transfer_evaluated', 'attack_attempted',
    }
    required.update(f'{feature}_clean' for feature in FEATURES)
    required.update(f'{feature}_adv' for feature in FEATURES)
    if not required.issubset(shard.columns):
        raise ValueError(f'Shard schema missing: {sorted(required - set(shard.columns))}')
    if len(shard) != len(source_frame):
        raise ValueError('Shard row count differs from its frozen split')
    if shard.POI_ID.astype(str).tolist() != source_frame.POI_ID.astype(str).tolist():
        raise ValueError('Shard ordered POI IDs differ from its frozen split')
    scalar_contract = {
        'split': str(source_frame.split.iloc[0]),
        'attack_method': spec['method'],
        'attack_condition': spec['condition'],
        'generation_seed': int(generation_seed),
        'generation_identity_sha256': identity['generation_identity_sha256'],
        'surrogate_checkpoint_sha256': identity['surrogate_checkpoint_sha256'],
        'attack_schema_version': ATTACK_SCHEMA_VERSION,
        'query_count_scope': QUERY_COUNT_SCOPE,
    }
    for column, expected in scalar_contract.items():
        if set(shard[column]) != {expected}:
            raise ValueError(f'Shard {column} differs from generation identity')
    expected_numeric = 0.0 if spec['numeric_budget_m'] is None else float(
        spec['numeric_budget_m'])
    expected_implicit = spec['numeric_budget_m'] is None
    if (not np.equal(shard['numeric_budget_m'].to_numpy(float), expected_numeric).all() or
            set(shard['numeric_budget_was_implicit_zero'].astype(bool)) != {
                expected_implicit} or
            not np.equal(
                shard['categorical_budget_l0'].to_numpy(float),
                float(spec['categorical_budget_l0'])).all()):
        raise ValueError('Shard budgets differ from attack condition')
    contract = distance_contract(cfg.get('distance_validation'))
    expected_tolerance = (
        contract['categorical_only_tolerance_m'] if expected_implicit else
        contract['numeric_budget_tolerance_m'])
    if (set(shard['distance_metric']) != {contract['metric']} or
            not np.equal(
                shard['distance_tolerance_m'].to_numpy(float), expected_tolerance).all()):
        raise ValueError('Shard distance metric/tolerance differs from configuration')
    if (set(shard['source_model']) != {SURROGATE_ID} or
            set(shard['victim_model']) != {SURROGATE_ID} or
            set(shard['victim_evaluation_scope']) != {
                'direct_white_box_source_model'} or
            shard['transfer_evaluated'].astype(bool).any() or
            not shard['attack_attempted'].astype(bool).all()):
        raise ValueError('Shard source/victim provenance is inconsistent')
    if shard.region.astype(str).tolist() != source_frame.region.astype(str).tolist():
        raise ValueError('Shard regions differ from frozen split')
    for feature in FEATURES:
        expected = source_frame[feature].to_numpy(dtype=float)
        observed = shard[f'{feature}_clean'].to_numpy(dtype=float)
        if not np.array_equal(expected, observed):
            raise ValueError(f'Shard clean anchor differs: {feature}')
    attacked_coords = shard[[f'{name}_adv' for name in NUMERIC]].to_numpy(float)
    attacked_categories = shard[[f'{name}_adv' for name in CATEGORICAL]].to_numpy(np.float32)
    evaluated = evaluate_constraints(
        source_frame, attacked_coords, attacked_categories,
        spec['numeric_budget_m'], spec['categorical_budget_l0'], tuples,
        numeric_bounds, encoder, boundary_validator, clean_boundary,
        cfg.get('distance_validation'))
    exact_fields = {
        'categorical_l0': evaluated['changed_l0'],
        'valid_numeric_train_range': evaluated['numeric_range_valid'],
        'valid_category_codebook': evaluated['category_codebook_valid'],
        'valid_clean_category_combo': evaluated['clean_combo_valid'],
        'valid_attack_category_combo': evaluated['attack_combo_valid'],
        'valid_category_combo': evaluated['combo_valid'],
        'within_budget': evaluated['within_budget'],
        'label_preserved': evaluated['label_preserved'],
        'constraints_valid': evaluated['constraints_valid'],
        'primary_evaluation_eligible': evaluated['constraints_valid'],
        'quarantine_reason': evaluated['quarantine_reason'],
        'clean_boundary_status': clean_boundary['status'],
        'attack_boundary_status': evaluated['attack_boundary']['status'],
    }
    for column, expected in exact_fields.items():
        if column not in shard or not np.array_equal(shard[column].to_numpy(), expected):
            raise ValueError(f'Shard final-state validation differs: {column}')
    for source_id, expected in clean_boundary['source_labels'].items():
        column = f'clean_polygon_region_{source_id}'
        if column not in shard or not np.array_equal(shard[column].to_numpy(), expected):
            raise ValueError(f'Shard clean polygon source differs: {source_id}')
    for source_id, expected in evaluated['attack_boundary']['source_labels'].items():
        column = f'attack_polygon_region_{source_id}'
        if column not in shard or not np.array_equal(shard[column].to_numpy(), expected):
            raise ValueError(f'Shard attack polygon source differs: {source_id}')
    if not np.allclose(
            shard['distance_m'].to_numpy(float), evaluated['distances'],
            rtol=0, atol=1e-9, equal_nan=False):
        raise ValueError('Shard recorded distance differs from recomputation')
    if not _query_count_is_valid(shard, spec['method'], cfg, tuples):
        raise ValueError('Shard query counts differ from exact candidate evaluations')
    return shard


def _shard_entry(path, frame):
    return {
        'sha256': sha256(path),
        'rows': len(frame),
        'ordered_poi_ids_sha256': _canonical_sha256(frame.POI_ID.astype(str).tolist()),
        'key_sha256': _canonical_sha256(
            frame[['POI_ID', 'split', 'attack_condition', 'generation_seed']]
            .astype(str).to_dict(orient='records')),
    }


def load_resumed_shard(path, entry, source_frame, spec, generation_seed, cfg, tuples,
                       numeric_bounds, encoder, boundary_validator, clean_boundary,
                       identity):
    path = Path(path)
    if not path.is_file() or not entry:
        raise ValueError(f'Cannot resume unregistered shard: {path.name}')
    if sha256(path) != entry.get('sha256'):
        raise ValueError(f'Shard checksum mismatch: {path.name}')
    frame = pd.read_parquet(path)
    validate_shard_frame(
        frame, source_frame, spec, generation_seed, cfg, tuples, numeric_bounds,
        encoder, boundary_validator, clean_boundary, identity)
    if _shard_entry(path, frame) != entry:
        raise ValueError(f'Shard manifest metadata mismatch: {path.name}')
    return frame


def persist_shard(path, frame, manifest, manifest_path, source_frame, spec,
                  generation_seed, cfg, tuples, numeric_bounds, encoder,
                  boundary_validator, clean_boundary, identity):
    path = Path(path)
    if path.exists() or path.name in manifest['shards']:
        raise FileExistsError(f'Refusing to overwrite registered shard: {path.name}')
    validate_shard_frame(
        frame, source_frame, spec, generation_seed, cfg, tuples, numeric_bounds,
        encoder, boundary_validator, clean_boundary, identity)
    _atomic_parquet(frame, path)
    # Validate the bytes that will be resumed, not only the in-memory frame.
    written = pd.read_parquet(path)
    validate_shard_frame(
        written, source_frame, spec, generation_seed, cfg, tuples, numeric_bounds,
        encoder, boundary_validator, clean_boundary, identity)
    manifest['shards'][path.name] = _shard_entry(path, written)
    _atomic_json(manifest_path, manifest)


def validate_full_key_coverage(attacks, clean, specs, generation_seeds):
    expected = {}
    for seed in generation_seeds:
        for split in ('train', 'validation'):
            ids = clean.loc[clean.split.eq(split), 'POI_ID'].astype(str).tolist()
            for spec in specs:
                expected[(split, int(seed), spec['condition'])] = ids
    grouped = {
        (str(split), int(seed), str(condition)): group.POI_ID.astype(str).tolist()
        for (split, seed, condition), group in attacks.groupby(
            ['split', 'generation_seed', 'attack_condition'], sort=False)
    }
    if set(grouped) != set(expected):
        missing = sorted(set(expected) - set(grouped))
        extra = sorted(set(grouped) - set(expected))
        raise ValueError(f'Full shard-key coverage differs: missing={missing}, extra={extra}')
    for key, ids in expected.items():
        if grouped[key] != ids:
            raise ValueError(f'Full ordered POI coverage differs for {key}')


def run(config_path, output_override=None):
    started = time.time()
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    if tuple(cfg.get('splits', ())) != ('train', 'validation'):
        raise ValueError('Attack generation must load exactly train and validation; test is locked')
    generation_seeds = list(cfg.get('generation_seeds', ()))
    if generation_seeds != [42, 202, 340]:
        raise ValueError('generation_seeds must be [42, 202, 340]')
    if len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError('generation_seeds must be unique')
    specs = attack_condition_specs(cfg)
    if len(specs) != 32:
        raise ValueError(f'Attack configuration must define 32 conditions, got {len(specs)}')
    distance_validation = distance_contract(cfg.get('distance_validation'))
    surrogate_seed = int(cfg['surrogate_seed'])
    output = Path(output_override or cfg['output'])
    markers = [output / name for name in
               ('_SUCCESS.json', '_PARTIAL.json', '_PENDING_REVIEW.json')]
    if any(path.exists() for path in markers):
        raise FileExistsError(f'Refusing to overwrite a finalized generation artifact: {output}')
    tables, figures = output / 'tables', output / 'figures'
    work, shards = output / 'work', output / 'work' / 'shards'
    checkpoints = Path(os.environ.get(
        'CHECKPOINT_DIR', output / 'work' / 'surrogate'))
    for directory in [tables, figures, work, shards, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    data_dir = Path(os.environ.get('DATA_DIR', cfg['data_dir']))
    clean_raw, _ = load_clean_splits(data_dir, splits=tuple(cfg['splits']))
    clean = numeric_frame(clean_raw)
    if set(clean.split) != {'train', 'validation'}:
        raise RuntimeError('Held-out split entered attack generation')
    train = clean.loc[clean.split.eq('train')].reset_index(drop=True)
    boundary_config_path = Path(cfg['boundary_config'])
    boundary_validator = AdministrativeBoundaryValidator.from_config(
        boundary_config_path)
    request = generation_request(
        config_path, cfg, data_dir, clean, boundary_config_path,
        boundary_validator)
    request_path = work / 'generation_request.json'
    checkpoint = checkpoints / 'surrogate.pt'
    if not request_path.exists() and (
            checkpoint.exists() or any(shards.glob('*.parquet'))):
        raise ValueError(
            'Checkpoint/shards exist without an immutable generation request; refusing resume')
    _ensure_immutable_json(request_path, request, 'generation request')
    request_sha256 = _canonical_sha256(request)
    set_seed(surrogate_seed)
    if checkpoint.exists():
        model, encoder, surrogate_result, checkpoint = load_surrogate(
            train, cfg, checkpoints, device, request_sha256)
    else:
        if any(shards.glob('*.parquet')):
            raise ValueError('Cannot retrain a missing checkpoint while cached shards exist')
        model, encoder, surrogate_result, checkpoint = train_surrogate(
            train, cfg, checkpoints, device, request_sha256)
    identity = _generation_identity(
        request, checkpoint, checkpoints / 'surrogate_encoder.json',
        checkpoints / 'surrogate_training.csv')
    identity_path = work / 'generation_identity.json'
    _ensure_immutable_json(identity_path, identity, 'generation identity')
    manifest_path = work / 'shard_manifest.json'
    manifest = _load_shard_manifest(
        manifest_path, identity['generation_identity_sha256'], shards)
    tuples, tuple_vectors = valid_tuples(train, encoder)
    numeric_bounds = {
        'min': train[NUMERIC].min().to_numpy(dtype=np.float64),
        'max': train[NUMERIC].max().to_numpy(dtype=np.float64),
    }
    tuple_table = pd.DataFrame(tuples, columns=CATEGORICAL)
    tuple_path = tables / 'valid_category_tuples_train.csv'
    tuple_text = tuple_table.to_csv(index=False, lineterminator='\n')
    if tuple_path.exists() and tuple_path.read_text() != tuple_text:
        raise ValueError('Existing train tuple table differs from generation identity')
    _atomic_text(tuple_path, tuple_text)
    numeric_table = pd.DataFrame({
        'feature': NUMERIC,
        'train_min': numeric_bounds['min'],
        'train_max': numeric_bounds['max'],
    })
    numeric_path = tables / 'numeric_constraints_train.csv'
    numeric_text = numeric_table.to_csv(index=False, lineterminator='\n')
    if numeric_path.exists() and numeric_path.read_text() != numeric_text:
        raise ValueError('Existing numeric constraint table differs from generation identity')
    _atomic_text(numeric_path, numeric_text)
    boundary_snapshot = output / 'boundary_config.yaml'
    if (boundary_snapshot.exists() and
            boundary_snapshot.read_text() != boundary_config_path.read_text()):
        raise ValueError('Existing boundary snapshot differs from generation identity')
    _atomic_text(boundary_snapshot, boundary_config_path.read_text())
    region_lookup = {region: index for index, region in enumerate(encoder['regions'])}
    spec_lookup = {spec['condition']: spec for spec in specs}
    produced = []
    for generation_seed in generation_seeds:
        set_seed(generation_seed)
        for split in cfg['splits']:
            frame = clean.loc[clean.split.eq(split)].reset_index(drop=True)
            coords = frame[NUMERIC].to_numpy(dtype=np.float64)
            categories = frame[CATEGORICAL].to_numpy(dtype=np.float32)
            labels = np.array([region_lookup[value] for value in frame.region], dtype=np.int64)
            clean_boundary = boundary_validator.validate(coords, frame.region.to_numpy())
            clean_pred, _ = predict(
                model, coords, categories, encoder, device, cfg['attack_batch_size'])

            for method in ['fgsm', 'pgd', 'cw_l2', 'capgd']:
                for budget in cfg['numeric_budgets_m']:
                    condition = f'{method}_m{budget}'
                    spec = spec_lookup[condition]
                    path = shards / f'{split}-g{generation_seed}-{condition}.parquet'
                    if path.exists():
                        load_resumed_shard(
                            path, manifest['shards'].get(path.name), frame, spec,
                            generation_seed, cfg, tuples, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, identity)
                    else:
                        set_seed(condition_rng_seed(generation_seed, split, condition))
                        attacked_coords, queries = numeric_attack(
                            method, coords, categories, labels, model, encoder,
                            budget, cfg, device)
                        attacked_coords = enforce_numeric_budget(coords, attacked_coords, budget)
                        attack_pred, _ = predict(
                            model, attacked_coords, categories, encoder, device,
                            cfg['attack_batch_size'])
                        part = records(
                            frame, attacked_coords, categories.copy(), labels, clean_pred,
                            attack_pred, method, condition, budget, 0, queries, tuples,
                            generation_seed, surrogate_seed, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, distance_validation,
                            identity['generation_identity_sha256'],
                            identity['surrogate_checkpoint_sha256'])
                        persist_shard(
                            path, part, manifest, manifest_path, frame, spec,
                            generation_seed, cfg, tuples, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, identity)
                    produced.append(path)
                    print(f'completed seed={generation_seed} {split} {condition}', flush=True)

            for method in ['category_exact', 'pcaa']:
                for budget in cfg['categorical_budgets_l0']:
                    condition = f'{method}_l{budget}'
                    spec = spec_lookup[condition]
                    path = shards / f'{split}-g{generation_seed}-{condition}.parquet'
                    if path.exists():
                        load_resumed_shard(
                            path, manifest['shards'].get(path.name), frame, spec,
                            generation_seed, cfg, tuples, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, identity)
                    else:
                        set_seed(condition_rng_seed(generation_seed, split, condition))
                        if method == 'category_exact':
                            attacked_categories, queries = categorical_exact(
                                coords, categories, labels, model, encoder, tuples,
                                tuple_vectors, budget, cfg, device)
                        else:
                            attacked_categories, queries = categorical_pcaa(
                                coords, categories, labels, model, encoder, tuples,
                                tuple_vectors, budget, cfg, device)
                        attack_pred, _ = predict(
                            model, coords, attacked_categories, encoder, device,
                            cfg['attack_batch_size'])
                        part = records(
                            frame, coords.copy(), attacked_categories, labels, clean_pred,
                            attack_pred, method, condition, None, budget, queries, tuples,
                            generation_seed, surrogate_seed, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, distance_validation,
                            identity['generation_identity_sha256'],
                            identity['surrogate_checkpoint_sha256'])
                        persist_shard(
                            path, part, manifest, manifest_path, frame, spec,
                            generation_seed, cfg, tuples, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, identity)
                    produced.append(path)
                    print(f'completed seed={generation_seed} {split} {condition}', flush=True)

            for level in cfg['mixed_budgets']:
                numeric_budget = level['numeric_m']
                categorical_budget = level['categorical_l0']
                for method in ['moeva', 'caa']:
                    condition = f"{method}_{level['name']}"
                    spec = spec_lookup[condition]
                    path = shards / f'{split}-g{generation_seed}-{condition}.parquet'
                    if path.exists():
                        load_resumed_shard(
                            path, manifest['shards'].get(path.name), frame, spec,
                            generation_seed, cfg, tuples, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, identity)
                    else:
                        rng_seed = condition_rng_seed(generation_seed, split, condition)
                        set_seed(rng_seed)
                        if method == 'moeva':
                            attacked_coords, attacked_categories, queries = mixed_evolution(
                                coords, categories, labels, model, encoder, tuples,
                                tuple_vectors, numeric_budget, categorical_budget, cfg,
                                device, rng_seed)
                            attacked_coords = enforce_numeric_budget(
                                coords, attacked_coords, numeric_budget)
                        else:
                            attacked_coords, capgd_queries = numeric_attack(
                                'capgd', coords, categories, labels, model, encoder,
                                numeric_budget, cfg, device)
                            attacked_coords = enforce_numeric_budget(
                                coords, attacked_coords, numeric_budget)
                            attacked_categories = categories.copy()
                            capgd_pred, _ = predict(
                                model, attacked_coords, attacked_categories, encoder,
                                device, cfg['attack_batch_size'])
                            failed = capgd_pred == labels
                            # CAA routing is an attack-specific forward evaluation and is
                            # included for every row.  The common final record prediction
                            # below remains excluded for every attack method.
                            queries = capgd_queries.copy() + 1
                            if failed.any():
                                evo_coords, evo_categories, evo_queries = mixed_evolution(
                                    coords[failed], categories[failed], labels[failed], model,
                                    encoder, tuples, tuple_vectors, numeric_budget,
                                    categorical_budget, cfg, device,
                                    rng_seed + 10_000)
                                attacked_coords[failed] = evo_coords
                                attacked_categories[failed] = evo_categories
                                queries[failed] += evo_queries
                            attacked_coords = enforce_numeric_budget(
                                coords, attacked_coords, numeric_budget)
                        attack_pred, _ = predict(
                            model, attacked_coords, attacked_categories, encoder,
                            device, cfg['attack_batch_size'])
                        part = records(
                            frame, attacked_coords, attacked_categories, labels, clean_pred,
                            attack_pred, method, condition, numeric_budget,
                            categorical_budget, queries, tuples, generation_seed,
                            surrogate_seed, numeric_bounds, encoder, boundary_validator,
                            clean_boundary, distance_validation,
                            identity['generation_identity_sha256'],
                            identity['surrogate_checkpoint_sha256'])
                        persist_shard(
                            path, part, manifest, manifest_path, frame, spec,
                            generation_seed, cfg, tuples, numeric_bounds, encoder,
                            boundary_validator, clean_boundary, identity)
                    produced.append(path)
                    print(f'completed seed={generation_seed} {split} {condition}', flush=True)

    attacks = pd.concat([pd.read_parquet(path) for path in produced], ignore_index=True)
    expected_conditions = len(specs)
    expected_rows = len(clean) * expected_conditions * len(generation_seeds)
    if len(attacks) != expected_rows:
        raise RuntimeError(f'Expected {expected_rows} attack rows, got {len(attacks)}')
    key = ['POI_ID', 'split', 'attack_condition', 'generation_seed']
    if attacks.duplicated(key).any():
        raise RuntimeError('Duplicate POI/attack condition/generation seed rows')
    validate_full_key_coverage(attacks, clean, specs, generation_seeds)
    if len(manifest['shards']) != request['expected_shards']:
        raise RuntimeError(
            f"Expected {request['expected_shards']} registered shards, got "
            f"{len(manifest['shards'])}")
    _atomic_parquet(attacks, tables / 'attacks.parquet')
    summary = (attacks.groupby(
        ['split', 'generation_seed', 'attack_method', 'attack_condition'], as_index=False)
               .agg(rows=('POI_ID', 'size'), clean_correct=('clean_source_correct', 'sum'),
                    attack_successes=('attack_success_source', 'sum'),
                    attack_success_rate_all_rows=('attack_success_source', 'mean'),
                    primary_eligible_rows=('primary_evaluation_eligible', 'sum'),
                    primary_valid_coverage=('primary_evaluation_eligible', 'mean'),
                    constraints_valid_rate=('constraints_valid', 'mean'),
                    label_preserved_rate=('label_preserved', 'mean'),
                    numeric_range_valid_rate=('valid_numeric_train_range', 'mean'),
                    clean_category_tuple_valid_rate=('valid_clean_category_combo', 'mean'),
                    attack_category_tuple_valid_rate=('valid_attack_category_combo', 'mean'),
                    mean_distance_m=('distance_m', 'mean'),
                    median_distance_m=('distance_m', 'median'),
                    mean_categorical_l0=('categorical_l0', 'mean'),
                    total_search_queries=('query_count', 'sum'),
                    mean_queries=('query_count', 'mean'),
                    min_queries=('query_count', 'min'),
                    max_queries=('query_count', 'max')))
    grouping = ['split', 'generation_seed', 'attack_method', 'attack_condition']
    intent = (attacks.loc[attacks.clean_source_correct]
              .groupby(grouping).attack_success_source.mean()
              .rename('intent_asr_clean_correct_all_rows').reset_index())
    eligible = (attacks.loc[attacks.clean_source_correct & attacks.constraints_valid]
                .groupby(['split', 'generation_seed', 'attack_method', 'attack_condition'])
                .attack_success_source.mean().rename('primary_valid_asr').reset_index())
    summary = summary.merge(intent, on=grouping, how='left')
    summary = summary.merge(eligible, on=grouping, how='left')
    _atomic_text(
        tables / 'attack_validity_summary.csv',
        summary.to_csv(index=False, lineterminator='\n'))
    fig, ax = plt.subplots(figsize=(14, 7))
    validation_summary = summary.loc[summary.split.eq('validation')]
    sns.barplot(data=validation_summary, x='attack_condition', y='primary_valid_asr',
                hue='attack_method', dodge=False, ax=ax)
    ax.tick_params(axis='x', rotation=75)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Attack success rate among clean-correct samples')
    ax.set_title('Surrogate attack success by valid attack condition')
    save_figure(fig, figures, 'attack_success_by_condition')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=validation_summary, x='mean_distance_m', y='primary_valid_asr',
                    hue='attack_method', size='mean_categorical_l0', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Attack success, coordinate distance and categorical changes')
    save_figure(fig, figures, 'attack_cost_vs_success')
    invalid_path = tables / 'quarantined_attacks.parquet'
    invalid = attacks.loc[~attacks.constraints_valid]
    if len(invalid):
        _atomic_parquet(invalid, invalid_path)
    elif invalid_path.exists():
        invalid_path.unlink()
    status = PENDING_STATUS
    metadata = {
        'status': status, 'scope': cfg['splits'], 'test_used': False,
        'completion_gate': (
            'Generated artifacts are never complete before a new fresh-context independent '
            'review passes every attack-generation requirement.'
        ),
        'loaded_feature_splits': sorted(clean.split.unique().tolist()),
        'test_feature_access': request['test_feature_access'],
        'surrogate_seed': surrogate_seed, 'generation_seeds': generation_seeds,
        'device': str(device), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'python': platform.python_version(), 'source_model': SURROGATE_ID,
        'surrogate_last_epoch': surrogate_result,
        'attack_conditions': expected_conditions, 'attack_rows': len(attacks),
        'valid_category_tuples': len(tuples), 'quarantined_rows': len(invalid),
        'invalid_rows_do_not_change_status': True,
        'numeric_constraints_train_only': {
            feature: {'min': float(numeric_bounds['min'][index]),
                      'max': float(numeric_bounds['max'][index])}
            for index, feature in enumerate(NUMERIC)},
        'boundary_validation': boundary_validator.provenance(),
        'boundary_limitations': request['boundary_limitations'],
        'boundary_config': {
            'path': str(boundary_config_path),
            'sha256': sha256(boundary_config_path),
            'snapshot': str(output / 'boundary_config.yaml'),
        },
        'algorithm_provenance': {
            'capgd_caa_paper': 'NeurIPS 2024, DOI 10.52202/079017-0873',
            'pcaa_paper': 'ICML 2023, PMLR 202:38428-38442',
            'moeva_framework': 'IJCAI 2022, DOI 10.24963/ijcai.2022/183',
            'reference_repository_commit': 'tabularbench bfb75415a6a31a41ddfeef34478eea1da227d19c',
            'implementation': 'POI-adapted PyTorch implementation; not byte-identical to reference',
            'query_count_scope': QUERY_COUNT_SCOPE,
        },
        'distance_validation': distance_validation,
        'budgets': {'numeric_m': cfg['numeric_budgets_m'],
                    'categorical_l0': cfg['categorical_budgets_l0'],
                    'mixed': cfg['mixed_budgets']},
        'dataset_sha256': {name: sha256(data_dir / name) for name in
                           ['poi_data_region.csv', 'sample_manifest.csv']},
        'checkpoint': {'path': str(checkpoint), 'sha256': file_hash(checkpoint)},
        'generation_request': {
            'path': str(request_path), 'sha256': _canonical_sha256(request)},
        'generation_identity': {
            'path': str(identity_path), **identity},
        'shard_manifest': {
            'path': str(manifest_path), 'sha256': sha256(manifest_path),
            'registered_shards': len(manifest['shards'])},
        'tables': {
            'attacks': {
                'path': str(tables / 'attacks.parquet'),
                'sha256': sha256(tables / 'attacks.parquet'),
                'rows': len(attacks),
            },
            'validity_summary': {
                'path': str(tables / 'attack_validity_summary.csv'),
                'sha256': sha256(tables / 'attack_validity_summary.csv'),
                'rows': len(summary),
            },
        },
        'seconds': time.time() - started,
    }
    _atomic_json(output / 'metadata.json', metadata)
    _atomic_json(output / 'metrics.json', {
        'status': PENDING_STATUS,
        'summary': summary.to_dict(orient='records')})
    _atomic_text(output / 'config.yaml', yaml.safe_dump(cfg, sort_keys=False))
    _atomic_text(output / 'report.md', f'''# 01. 적대적 공격 생성 및 유효성 검증

지역·주소·ID를 입력에서 제외한 5개 피처를 대상으로 train과 validation에 총
{expected_conditions}개 공격 조건, {len(attacks):,}개 결과 행을 만들었다. test는 열지 않았다.
수치 공격은 좌표 이동을 미터 단위 L2 거리와 train 좌표 범위로 제한하고, 범주 공격은
train에서 label과 무관하게 고정한 {len(tuples)}개 대·중·소분류 조합만 허용했다.
현재 generation run은 test feature를 materialize하지 않았지만, 이전 탐색 과정에서 test를
포함한 34,000개 전체 좌표가 한 차례 materialize된 사실이 있다. 따라서 historical pristine
test라고 주장하지 않으며 이번 격리는 prospective isolation으로만 해석한다.

FGSM, PGD, CW-L2는 표준 정의를 좌표에 적용했다. CAPGD, PCAA, MOEVA와 CAA는 현재
PyTorch/TabPFN 환경에서 동작하도록 POI 제약에 맞춰 이식한 구현이며 원 논문 구현과
byte-identical하지 않다. CAA는 CAPGD 후 실패 표본에 MOEVA를 적용한다.

모든 행에는 공격 방법·예산·generation seed·실제로 평가한 source/victim model·split·
이동거리·변경 mask·query 수·제약 충족·source 성공 여부를 기록했다. 두 실제 행정경계
source가 모두 원 지역을 확인한 경우에만 label_preserved=true다. 총 {len(invalid):,}개
격리 행이 있다. 원 POI label을 만든 행정경계 vintage는 알려지지 않았고, 사용한 두 경계는
checksum을 고정한 third-party derivative다. 공식 원본 bytes, 변환 chain, source 독립성은
독립 인증되지 않았다.

query_count는 attack search candidate forward evaluation만 센다. CAA routing은 포함하고,
모든 방법에 공통인 clean baseline과 final record prediction은 제외한다. 거리 검사는
{DISTANCE_METRIC} 근사를 쓰며 numeric tolerance는
{distance_validation['numeric_budget_tolerance_m']}m, categorical-only 좌표 불변 tolerance는
{distance_validation['categorical_only_tolerance_m']}m이다.

invalid row가 0개여도 이 단계는 완료가 아니다. 새 fresh-context 독립 검수가 모든 요구사항을
PASS하기 전까지 status는 `{PENDING_STATUS}`이며 success marker를 만들지 않는다.
''')
    _atomic_json(output / '_PENDING_REVIEW.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stages/01_attack_generation.yaml')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == '__main__':
    main()
