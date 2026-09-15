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

from .data import CATEGORICAL, FEATURES, NUMERIC, load_dataset, sha256


SURROGATE_ID = 'poi_mlp_surrogate_v1'
METRES_PER_LAT_DEGREE = 110_574.0
METRES_PER_LON_DEGREE = 111_320.0


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


def train_surrogate(train, cfg, output, device):
    encoder = encoder_from_train(train)
    coords = train[NUMERIC].to_numpy(dtype=np.float32)
    categories = train[CATEGORICAL].to_numpy(dtype=np.float32)
    x = encoded_features(coords, categories, encoder)
    region_lookup = {region: index for index, region in enumerate(encoder['regions'])}
    y = np.array([region_lookup[region] for region in train.region], dtype=np.int64)

    rng = np.random.default_rng(cfg['seed'])
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
    generator = torch.Generator().manual_seed(cfg['seed'])
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
    torch.save({'state_dict': best_state, 'input_dim': x.shape[1],
                'classes': len(encoder['regions'])}, checkpoint)
    pd.DataFrame(history).to_csv(output / 'surrogate_training.csv', index=False)
    serializable = {
        'category_values': encoder['category_values'], 'regions': encoder['regions'],
        'numeric_mean': encoder['numeric_mean'].tolist(),
        'numeric_std': encoder['numeric_std'].tolist(),
    }
    (output / 'surrogate_encoder.json').write_text(json.dumps(serializable, indent=2) + '\n')
    return model, encoder, history[-1], checkpoint


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
    norm = offsets.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return offsets * torch.clamp(float(budget) / norm, max=1.0)


def numeric_attack_batch(method, base, categories, labels, model, encoder, budget, cfg):
    onehot = torch.from_numpy(category_onehot(categories.cpu().numpy(), encoder)).to(base.device)
    steps = int(cfg['attacks'][method]['steps'])
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
            distance = delta.norm(dim=1) / float(budget)
            loss = (F.relu(-margin + cfg['attacks'][method]['kappa']) +
                    cfg['attacks'][method]['distance_weight'] * distance).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(project_l2(delta, budget))
                quality = torch.where(margin.gt(0), 1000 - distance, margin)
                improved = quality > best_quality
                best[improved] = delta.detach()[improved]
                best_quality[improved] = quality[improved]
        return offsets_to_coords(base, best), np.full(len(base), steps)

    delta = torch.zeros_like(base)
    previous = delta.clone()
    best = delta.clone()
    best_loss = torch.full((len(base),), -torch.inf, device=base.device)
    step = torch.full((len(base), 1),
                      2 * float(budget) / max(steps, 1), device=base.device)
    stagnant = torch.zeros(len(base), dtype=torch.int64, device=base.device)
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
    return offsets_to_coords(base, best), np.full(len(base), steps)


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
        queries.append(allowed.sum(axis=1))
    return np.concatenate(selected), np.concatenate(queries)


def categorical_pcaa(coords, categories, labels, model, encoder, tuples, tuple_vectors,
                     budget, cfg, device):
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
    mean_latitude = np.deg2rad((clean[:, 1] + attacked[:, 1]) / 2)
    dx = (attacked[:, 0] - clean[:, 0]) * METRES_PER_LON_DEGREE * np.cos(mean_latitude)
    dy = (attacked[:, 1] - clean[:, 1]) * METRES_PER_LAT_DEGREE
    return np.sqrt(dx ** 2 + dy ** 2)


def enforce_numeric_budget(clean, attacked, budget):
    """Project final floating-point coordinates inside the geodesic approximation."""
    result = attacked.astype(np.float64, copy=True)
    origin = clean.astype(np.float64, copy=False)
    for _ in range(3):
        distance = distance_metres(origin, result)
        scale = np.minimum(1.0, (float(budget) * (1 - 1e-7)) /
                           np.maximum(distance, 1e-12))
        result = origin + (result - origin) * scale[:, None]
    return result


def records(frame, attacked_coords, attacked_categories, labels, clean_pred, attack_pred,
            method, condition, numeric_budget, categorical_budget, queries, tuples, seed):
    clean_coords = frame[NUMERIC].to_numpy(dtype=np.float64)
    clean_categories = frame[CATEGORICAL].to_numpy(dtype=np.float32)
    distances = distance_metres(clean_coords, attacked_coords)
    changed = clean_categories != attacked_categories
    changed_l0 = changed.sum(axis=1)
    valid_set = {tuple(row) for row in tuples.tolist()}
    unchanged_category_tuple = np.all(clean_categories == attacked_categories, axis=1)
    combo_valid = unchanged_category_tuple | np.array(
        [tuple(row) in valid_set for row in attacked_categories])
    numeric_valid = np.ones(len(frame), dtype=bool) if numeric_budget is None else distances <= numeric_budget + 5e-2
    categorical_valid = (changed_l0 <= (categorical_budget or 0))
    output = pd.DataFrame({
        'POI_ID': frame.POI_ID.to_numpy(), 'split': frame.split.to_numpy(),
        'region': frame.region.to_numpy(), 'attack_method': method,
        'attack_condition': condition, 'source_model': SURROGATE_ID,
        'victim_model': '', 'targeted': False, 'target_region': '',
        'numeric_budget_m': numeric_budget, 'categorical_budget_l0': categorical_budget,
        'distance_m': distances, 'categorical_l0': changed_l0,
        'valid_range': numeric_valid, 'valid_category_combo': combo_valid,
        'within_budget': numeric_valid & categorical_valid,
        'constraints_valid': numeric_valid & categorical_valid & combo_valid,
        'label_preserved': True, 'clean_source_pred': clean_pred,
        'attack_source_pred': attack_pred, 'clean_source_correct': clean_pred == labels,
        'attack_success_source': (clean_pred == labels) & (attack_pred != labels),
        'query_count': queries, 'seed': seed,
    })
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


def run(config_path, output_override=None):
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    output = Path(output_override or cfg['output'])
    if (output / '_SUCCESS.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed {output}')
    tables, figures, shards = output / 'tables', output / 'figures', output / 'work' / 'shards'
    experiment_root = output.parent.parent if output.parent.name == 'reports' else output
    checkpoints = Path(os.environ.get('CHECKPOINT_DIR', experiment_root / 'checkpoints'))
    for directory in [tables, figures, shards, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    data_dir = Path(os.environ.get('DATA_DIR', cfg['data_dir']))
    frames, _ = load_dataset(data_dir)
    clean = numeric_frame(frames['clean'])
    train = clean.loc[clean.split.eq('train')].reset_index(drop=True)
    model, encoder, surrogate_result, checkpoint = train_surrogate(
        train, cfg, checkpoints, device)
    constraint_scope = clean.loc[clean.split.isin(cfg['splits'])]
    tuples, tuple_vectors = valid_tuples(constraint_scope, encoder)
    region_lookup = {region: index for index, region in enumerate(encoder['regions'])}
    produced = []
    for split in cfg['splits']:
        frame = clean.loc[clean.split.eq(split)].reset_index(drop=True)
        coords = frame[NUMERIC].to_numpy(dtype=np.float64)
        categories = frame[CATEGORICAL].to_numpy(dtype=np.float32)
        labels = np.array([region_lookup[value] for value in frame.region], dtype=np.int64)
        clean_pred, _ = predict(model, coords, categories, encoder, device, cfg['attack_batch_size'])

        for method in ['fgsm', 'pgd', 'cw_l2', 'capgd']:
            for budget in cfg['numeric_budgets_m']:
                condition = f'{method}_m{budget}'
                path = shards / f'{split}-{condition}.parquet'
                if not path.exists():
                    attacked_coords, queries = numeric_attack(
                        method, coords, categories, labels, model, encoder,
                        budget, cfg, device)
                    attacked_coords = enforce_numeric_budget(coords, attacked_coords, budget)
                    attack_pred, _ = predict(model, attacked_coords, categories, encoder,
                                             device, cfg['attack_batch_size'])
                    part = records(frame, attacked_coords, categories.copy(), labels,
                                   clean_pred, attack_pred, method, condition, budget,
                                   0, queries, tuples, cfg['seed'])
                    part.to_parquet(path, index=False)
                produced.append(path)
                print(f'completed {split} {condition}', flush=True)

        for method in ['category_exact', 'pcaa']:
            for budget in cfg['categorical_budgets_l0']:
                condition = f'{method}_l{budget}'
                path = shards / f'{split}-{condition}.parquet'
                if not path.exists():
                    if method == 'category_exact':
                        attacked_categories, queries = categorical_exact(
                            coords, categories, labels, model, encoder, tuples,
                            tuple_vectors, budget, cfg, device)
                    else:
                        attacked_categories, queries = categorical_pcaa(
                            coords, categories, labels, model, encoder, tuples,
                            tuple_vectors, budget, cfg, device)
                    attack_pred, _ = predict(model, coords, attacked_categories, encoder,
                                             device, cfg['attack_batch_size'])
                    part = records(frame, coords.copy(), attacked_categories, labels,
                                   clean_pred, attack_pred, method, condition, None,
                                   budget, queries, tuples, cfg['seed'])
                    part.to_parquet(path, index=False)
                produced.append(path)
                print(f'completed {split} {condition}', flush=True)

        for level in cfg['mixed_budgets']:
            numeric_budget = level['numeric_m']
            categorical_budget = level['categorical_l0']
            for method in ['moeva', 'caa']:
                condition = f"{method}_{level['name']}"
                path = shards / f'{split}-{condition}.parquet'
                if not path.exists():
                    if method == 'moeva':
                        attacked_coords, attacked_categories, queries = mixed_evolution(
                            coords, categories, labels, model, encoder, tuples, tuple_vectors,
                            numeric_budget, categorical_budget, cfg, device,
                            cfg['seed'] + numeric_budget + categorical_budget)
                        attacked_coords = enforce_numeric_budget(
                            coords, attacked_coords, numeric_budget)
                    else:
                        attacked_coords, capgd_queries = numeric_attack(
                            'capgd', coords, categories, labels, model, encoder,
                            numeric_budget, cfg, device)
                        attacked_coords = enforce_numeric_budget(
                            coords, attacked_coords, numeric_budget)
                        attacked_categories = categories.copy()
                        capgd_pred, _ = predict(model, attacked_coords, attacked_categories,
                                                encoder, device, cfg['attack_batch_size'])
                        failed = capgd_pred == labels
                        queries = capgd_queries.copy()
                        if failed.any():
                            evo_coords, evo_categories, evo_queries = mixed_evolution(
                                coords[failed], categories[failed], labels[failed], model,
                                encoder, tuples, tuple_vectors, numeric_budget,
                                categorical_budget, cfg, device,
                                cfg['seed'] + 10_000 + numeric_budget + categorical_budget)
                            attacked_coords[failed] = evo_coords
                            attacked_categories[failed] = evo_categories
                            queries[failed] += evo_queries
                        attacked_coords = enforce_numeric_budget(
                            coords, attacked_coords, numeric_budget)
                    attack_pred, _ = predict(model, attacked_coords, attacked_categories,
                                             encoder, device, cfg['attack_batch_size'])
                    part = records(frame, attacked_coords, attacked_categories, labels,
                                   clean_pred, attack_pred, method, condition,
                                   numeric_budget, categorical_budget, queries,
                                   tuples, cfg['seed'])
                    part.to_parquet(path, index=False)
                produced.append(path)
                print(f'completed {split} {condition}', flush=True)

    attacks = pd.concat([pd.read_parquet(path) for path in produced], ignore_index=True)
    expected_conditions = 4 * len(cfg['numeric_budgets_m']) + 2 * len(cfg['categorical_budgets_l0']) + 2 * len(cfg['mixed_budgets'])
    expected_rows = len(clean.loc[clean.split.isin(cfg['splits'])]) * expected_conditions
    if len(attacks) != expected_rows:
        raise RuntimeError(f'Expected {expected_rows} attack rows, got {len(attacks)}')
    if attacks.duplicated(['POI_ID', 'split', 'attack_condition']).any():
        raise RuntimeError('Duplicate POI/attack condition rows')
    attacks.to_parquet(tables / 'attacks.parquet', index=False)
    summary = (attacks.groupby(['split', 'attack_method', 'attack_condition'], as_index=False)
               .agg(rows=('POI_ID', 'size'), clean_correct=('clean_source_correct', 'sum'),
                    attack_successes=('attack_success_source', 'sum'),
                    attack_success_rate=('attack_success_source', 'mean'),
                    constraints_valid_rate=('constraints_valid', 'mean'),
                    label_preserved_rate=('label_preserved', 'mean'),
                    mean_distance_m=('distance_m', 'mean'),
                    median_distance_m=('distance_m', 'median'),
                    mean_categorical_l0=('categorical_l0', 'mean'),
                    mean_queries=('query_count', 'mean')))
    eligible = (attacks.loc[attacks.clean_source_correct]
                .groupby(['split', 'attack_method', 'attack_condition'])
                .attack_success_source.mean().rename('eligible_asr').reset_index())
    summary = summary.merge(eligible, on=['split', 'attack_method', 'attack_condition'], how='left')
    summary.to_csv(tables / 'attack_validity_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    validation_summary = summary.loc[summary.split.eq('validation')]
    sns.barplot(data=validation_summary, x='attack_condition', y='eligible_asr',
                hue='attack_method', dodge=False, ax=ax)
    ax.tick_params(axis='x', rotation=75)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Attack success rate among clean-correct samples')
    ax.set_title('Surrogate attack success by valid attack condition')
    save_figure(fig, figures, 'attack_success_by_condition')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=validation_summary, x='mean_distance_m', y='eligible_asr',
                    hue='attack_method', size='mean_categorical_l0', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Attack success, coordinate distance and categorical changes')
    save_figure(fig, figures, 'attack_cost_vs_success')
    invalid_path = tables / 'invalid_attacks.csv'
    invalid = attacks.loc[~attacks.constraints_valid]
    if len(invalid):
        invalid.to_csv(invalid_path, index=False)
        raise RuntimeError(f'Generated {len(invalid)} constraint-invalid attacks')
    if invalid_path.exists():
        invalid_path.unlink()
    metadata = {
        'status': 'complete', 'scope': cfg['splits'], 'test_used': False,
        'seed': cfg['seed'], 'device': str(device), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'python': platform.python_version(), 'source_model': SURROGATE_ID,
        'surrogate_last_epoch': surrogate_result,
        'attack_conditions': expected_conditions, 'attack_rows': len(attacks),
        'valid_category_tuples': len(tuples),
        'algorithm_provenance': {
            'capgd_caa_paper': 'NeurIPS 2024, DOI 10.52202/079017-0873',
            'pcaa_paper': 'ICML 2023, PMLR 202:38428-38442',
            'moeva_framework': 'IJCAI 2022, DOI 10.24963/ijcai.2022/183',
            'reference_repository_commit': 'tabularbench bfb75415a6a31a41ddfeef34478eea1da227d19c',
            'implementation': 'POI-adapted PyTorch implementation; not byte-identical to reference',
        },
        'budgets': {'numeric_m': cfg['numeric_budgets_m'],
                    'categorical_l0': cfg['categorical_budgets_l0'],
                    'mixed': cfg['mixed_budgets']},
        'dataset_sha256': {name: sha256(data_dir / name) for name in
                           ['poi_data_region.csv', 'sample_manifest.csv']},
        'checkpoint': {'path': str(checkpoint), 'sha256': file_hash(checkpoint)},
        'seconds': time.time() - started,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'metrics.json').write_text(json.dumps({
        'summary': summary.to_dict(orient='records')}, indent=2) + '\n')
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'report.md').write_text(f'''# 01. 적대적 공격 생성 및 유효성 검증

지역·주소·ID를 입력에서 제외한 5개 피처를 대상으로 train과 validation에 총
{expected_conditions}개 공격 조건, {len(attacks):,}개 결과 행을 만들었다. test는 열지 않았다.
수치 공격은 좌표 이동을 미터 단위 L2 거리로 제한하고, 범주 공격은 공격 대상 clean
split에서 label과 무관하게 고정한 {len(tuples)}개 대·중·소분류 조합만 허용했다.
test 조합은 사용하지 않았고 실패 표본도 원본과 함께 저장했다.

FGSM, PGD, CW-L2는 표준 정의를 좌표에 적용했다. CAPGD, PCAA, MOEVA와 CAA는 현재
PyTorch/TabPFN 환경에서 동작하도록 POI 제약에 맞춰 이식한 구현이며 원 논문 구현과
byte-identical하지 않다. CAA는 CAPGD 후 실패 표본에 MOEVA를 적용한다.

모든 행에는 공격 방법·예산·source model·split·이동거리·변경 mask·query 수·제약 충족·
source 성공 여부를 기록했다. 유효성 검사에서 제약 위반이 한 건이라도 있으면 이 단계는
성공 마커를 만들지 않는다.
''')
    (output / '_SUCCESS.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/adversarial.yaml')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == '__main__':
    main()
