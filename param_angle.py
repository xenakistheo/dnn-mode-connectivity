import argparse
import math
import json
import csv
import os
import re
import torch
from collections import OrderedDict


def load_state(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        return ckpt['model_state']
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        return ckpt['state_dict']
    if isinstance(ckpt, dict):
        return ckpt
    raise SystemExit(f"Unsupported checkpoint format: {path}")


def looks_like_curve_state(state):
    # Heuristic: presence of keys starting with 'net.' and at least one key ending with _0
    has_net_prefix = any(k.startswith('net.') for k in state.keys())
    has_bend_suffix = any(re.search(r'_(\d+)$', k) for k in state.keys())
    return has_net_prefix and has_bend_suffix


def detect_num_bends_from_state(state):
    # Prefer coeff_layer.range or coeff_layer.binom if present
    for key in ('coeff_layer.range', 'coeff_layer.binom', 'coeff_layer.rev_range'):
        if key in state and torch.is_tensor(state[key]):
            return int(state[key].numel())
    # fallback: search for suffix _<n> and infer max index
    max_idx = -1
    for k in state.keys():
        m = re.search(r'_(\d+)$', k)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    if max_idx >= 0:
        return max_idx + 1
    return None


def export_bend_from_curve_state(curve_state, bend_index):
    """
    Convert a curve-format state dict into a base-style state dict by extracting
    the tensors corresponding to the given bend_index.

    Rules:
    - For keys starting with 'net.' :
        - strip leading 'net.' prefix
        - if key ends with _<i>, remove that suffix and take tensor for _<bend_index>
          (e.g. 'net.conv1.weight_1' -> 'conv1.weight')
        - if key does not end with _<i>, keep as-is after stripping 'net.' (buffers like running_mean)
    - Ignore coeff_layer.* keys (they are not part of base model)
    """
    exported = {}
    num_bends = detect_num_bends_from_state(curve_state)
    if num_bends is None:
        raise RuntimeError("Could not detect num_bends in curve checkpoint.")

    if bend_index is None:
        # default to middle
        bend_index = num_bends // 2

    if bend_index < 0 or bend_index >= num_bends:
        raise ValueError(f"bend_index {bend_index} out of range for num_bends={num_bends}")

    for k, v in curve_state.items():
        # Skip coeff_layer entries
        if k.startswith('coeff_layer.'):
            continue
        if not k.startswith('net.'):
            # not under net., keep only if it looks like a base key (rare)
            exported[k] = v
            continue
        # strip net. prefix
        key_no_net = k[len('net.'):]
        m = re.search(r'_(\d+)$', key_no_net)
        if m:
            idx = int(m.group(1))
            key_base = key_no_net[:-(len(m.group(0)))]
            if idx == bend_index:
                exported[key_base] = v
            # else skip other bends
        else:
            # no suffix: probably a shared buffer like running_mean/var or coeff_layer buffers
            exported[key_no_net] = v
    return exported


def valid_tensor(t, include_buffers=False):
    if not torch.is_tensor(t):
        return False
    if t.dtype.is_floating_point:
        return True
    if include_buffers and t.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
        return True
    return False


def build_flat_vector(state, keys, include_buffers=False):
    parts = []
    per_key_shapes = {}
    for k in keys:
        t = state[k]
        if not valid_tensor(t, include_buffers=include_buffers):
            raise ValueError(f"Key {k} is not a valid tensor for concatenation (dtype={getattr(t,'dtype',None)})")
        per_key_shapes[k] = tuple(t.shape)
        parts.append(t.reshape(-1).to(torch.get_default_dtype()))
    if len(parts) == 0:
        return torch.tensor([], dtype=torch.get_default_dtype()), per_key_shapes
    flat = torch.cat(parts).double()
    return flat, per_key_shapes


def per_key_contributions(stateA, stateB, stateC, keys, include_buffers=False):
    contributions = OrderedDict()
    for k in keys:
        a = stateA[k].double()
        b = stateB[k].double()
        c = stateC[k].double()
        v1 = (a - c).reshape(-1)
        v2 = (b - c).reshape(-1)
        contributions[k] = float((v1 * v2).sum().item())
    return contributions


def canonicalize_checkpoint_for_vectorization(path, curve_bend_index=None, include_buffers=False):
    """
    Load checkpoint and, if it's a curve-format checkpoint, export the specified bend
    into base-key style dict. Returns a dict mapping base-style keys -> tensors.
    """
    state_raw = load_state(path)
    if looks_like_curve_state(state_raw):
        num_bends = detect_num_bends_from_state(state_raw)
        print(f"{path}: detected curve-format checkpoint with num_bends={num_bends}")
        if curve_bend_index is None:
            curve_bend_index = num_bends // 2
            print(f"{path}: no --curve-bend-index provided, defaulting to middle index {curve_bend_index}")
        else:
            print(f"{path}: extracting bend index {curve_bend_index}")
        exported = export_bend_from_curve_state(state_raw, curve_bend_index)
        # The exported keys are base-style (strip 'net.' and suffixes)
        return exported
    else:
        # already base-style or other; return as loaded
        return state_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_a', '-a', required=True, help='checkpoint for endpoint A')
    parser.add_argument('--ckpt_b', '-b', required=True, help='checkpoint for endpoint B')
    parser.add_argument('--ckpt_c', '-c', required=True, help='checkpoint for checkpoint C (apex)')
    parser.add_argument('--curve-bend-index', type=int, default=None,
                        help='bend index to extract from curve-format checkpoints (default: middle)')
    parser.add_argument('--include-buffers', action='store_true',
                        help='include non-parameter buffers that are floating tensors (default: exclude)')
    parser.add_argument('--show_counts', action='store_true',
                        help='print counts of keys considered/skipped')
    parser.add_argument('--topk', type=int, default=0,
                        help='show top-K keys by absolute contribution to dot((A-C),(B-C))')
    parser.add_argument('--eps', type=float, default=1e-12, help='small epsilon to avoid division by zero')
    parser.add_argument('--save_csv', type=str, default=None, help='path to save a one-line CSV summary')
    parser.add_argument('--save_json', type=str, default=None, help='path to save JSON with detailed fields')
    parser.add_argument('--save_contribs', type=str, default=None,
                        help='path to save per-key contributions CSV (writes all considered keys)')
    args = parser.parse_args()

    stateA = canonicalize_checkpoint_for_vectorization(args.ckpt_a, args.curve_bend_index, include_buffers=args.include_buffers)
    stateB = canonicalize_checkpoint_for_vectorization(args.ckpt_b, args.curve_bend_index, include_buffers=args.include_buffers)
    stateC = canonicalize_checkpoint_for_vectorization(args.ckpt_c, args.curve_bend_index, include_buffers=args.include_buffers)

    if not (isinstance(stateA, dict) and isinstance(stateB, dict) and isinstance(stateC, dict)):
        raise SystemExit("Error: expected all checkpoints to be state dicts or dicts containing 'model_state'.")

    keys_all = set(stateA.keys()) & set(stateB.keys()) & set(stateC.keys())
    keys_all = sorted(keys_all)

    used_keys = []
    skipped = []
    for k in keys_all:
        a = stateA[k]
        b = stateB[k]
        c = stateC[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b) and torch.is_tensor(c)):
            skipped.append((k, 'not_tensor'))
            continue
        if a.shape != b.shape or a.shape != c.shape:
            skipped.append((k, f'shape_mismatch {a.shape}/{b.shape}/{c.shape}'))
            continue
        if not valid_tensor(a, include_buffers=args.include_buffers):
            skipped.append((k, f'bad_dtype {a.dtype}'))
            continue
        used_keys.append(k)

    if args.show_counts:
        print(f"Total keys common to all checkpoints: {len(keys_all)}")
        print(f"Keys used for vectorization: {len(used_keys)}")
        print(f"Keys skipped: {len(skipped)}")
        if len(skipped) > 0:
            print("Examples of skipped keys (first 10):")
            for k, reason in skipped[:10]:
                print(" ", k, "->", reason)

    if len(used_keys) == 0:
        raise SystemExit("No compatible keys found to build parameter vectors. Try --include-buffers or check checkpoint formats.")

    vecA, shapes = build_flat_vector(stateA, used_keys, include_buffers=args.include_buffers)
    vecB, _ = build_flat_vector(stateB, used_keys, include_buffers=args.include_buffers)
    vecC, _ = build_flat_vector(stateC, used_keys, include_buffers=args.include_buffers)

    v1 = (vecA - vecC).double()
    v2 = (vecB - vecC).double()

    norm1 = v1.norm().item()
    norm2 = v2.norm().item()
    dot = float((v1 * v2).sum().item())

    denom = max(norm1 * norm2, args.eps)
    cos = dot / denom
    cos = max(min(cos, 1.0), -1.0)
    angle_rad = math.acos(cos)
    angle_deg = math.degrees(angle_rad)

    summary = {
        'num_parameters': int(vecA.numel()),
        'norm_A_minus_C': norm1,
        'norm_B_minus_C': norm2,
        'dot': dot,
        'cosine': cos,
        'angle_rad': angle_rad,
        'angle_deg': angle_deg,
    }

    print("=== Angle at C between A and B ===")
    print(f"number of parameters concatenated: {summary['num_parameters']:,}")
    print(f"||A - C||        = {summary['norm_A_minus_C']:.6e}")
    print(f"||B - C||        = {summary['norm_B_minus_C']:.6e}")
    print(f"dot((A-C),(B-C)) = {summary['dot']:.6e}")
    print(f"cosine similarity = {summary['cosine']:.6e}")
    print(f"angle (rad)      = {summary['angle_rad']:.6e}")
    print(f"angle (deg)      = {summary['angle_deg']:.6f}")

    if norm2 > args.eps:
        proj_len = dot / norm2
        proj_frac = proj_len / norm1 if norm1 > args.eps else float('nan')
        print(f"scalar projection length of (A-C) onto (B-C): {proj_len:.6e}")
        print(f"fraction of ||A-C|| explained by projection onto (B-C): {proj_frac:.6e}")
    else:
        print("||B - C|| is zero (or extremely small); can't compute projection.")

    contribs = None
    if args.topk > 0 or args.save_contribs:
        contribs = per_key_contributions(stateA, stateB, stateC, used_keys, include_buffers=args.include_buffers)
        items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        if args.topk > 0:
            print(f"\nTop {args.topk} keys by absolute contribution to dot((A-C),(B-C)):")
            for k, v in items[:args.topk]:
                frac = v / (dot if abs(dot) > 0 else 1.0)
                print(f"  {k:60s}  contrib={v: .6e}  frac={frac: .6e}")

    # Save CSV summary if requested
    if args.save_csv:
        header = ['num_parameters', 'norm_A_minus_C', 'norm_B_minus_C', 'dot', 'cosine', 'angle_rad', 'angle_deg']
        write_header = not os.path.exists(args.save_csv)
        with open(args.save_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([summary[h] for h in header])
        print(f"Wrote summary CSV to {args.save_csv}")

    # Save JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON summary to {args.save_json}")

    # Save per-key contributions if requested
    if args.save_contribs and contribs is not None:
        with open(args.save_contribs, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'contribution'])
            for k, v in contribs.items():
                writer.writerow([k, v])
        print(f"Wrote per-key contributions to {args.save_contribs}")

    return summary


if __name__ == '__main__':
    main()