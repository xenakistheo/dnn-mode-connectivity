import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Computes values for plane visualization')
    parser.add_argument('--dir', type=str, default='/tmp/plane', metavar='DIR',
                        help='training directory (default: /tmp/plane)')

    parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                        help='number of points in the grid (default: 21)')
    parser.add_argument('--margin_left', type=float, default=0.2, metavar='M',
                        help='left margin (default: 0.2)')
    parser.add_argument('--margin_right', type=float, default=0.2, metavar='M',
                        help='right margin (default: 0.2)')
    parser.add_argument('--margin_bottom', type=float, default=0.2, metavar='M',
                        help='bottom margin (default: 0.)')
    parser.add_argument('--margin_top', type=float, default=0.2, metavar='M',
                        help='top margin (default: 0.2)')

    parser.add_argument('--curve_points', type=int, default=61, metavar='N',
                        help='number of points on the curve (default: 61)')

    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                        help='model name (default: None)')
    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')

    parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                        help='checkpoint to eval (default: None)')

    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--device', type=str, default=None,
                        help='device to run on, e.g., "cpu", "cuda", or "mps". If not set, auto-detects (default: auto)')

    parser.add_argument('--inspect_ckpt', action='store_true',
                        help='If set, only inspect the checkpoint keys and exit (useful for debugging state_dict mismatches)')

    parser.add_argument('--allow_partial_load', action='store_true',
                        help='If set, load checkpoint with strict=False when exact match fails (may leave some params at default).')

    return parser.parse_args()


def update_bn_to_device(loader, model, device, max_batches=None, **kwargs):
    """
    Update batch-norm statistics by doing forward passes on the data.
    This wrapper moves the input batches to `device` before calling model,
    preventing device-type mismatches (e.g., MPS vs CPU).
    """
    model.train()
    i = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            inp = batch[0]
        else:
            inp = batch
        inp = inp.to(device)
        model(inp, **kwargs)
        i += 1
        if max_batches is not None and i >= max_batches:
            break


def main():
    args = parse_args()

    os.makedirs(args.dir, exist_ok=True)

    # Select device: user override or auto-detect CUDA / MPS / CPU
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        shuffle_train=False
    )

    architecture = getattr(models, args.model)
    curve = getattr(curves, args.curve)

    curve_model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    curve_model.to(device)

    # Load checkpoint onto the chosen device
    if args.ckpt is None:
        raise RuntimeError("No checkpoint provided. Use --ckpt /path/to/checkpoint.pt")

    checkpoint = torch.load(args.ckpt, map_location=device)

    # If user only wants to inspect checkpoint contents, print some info and exit.
    if args.inspect_ckpt:
        print("Checkpoint type:", type(checkpoint))
        if isinstance(checkpoint, dict):
            print("Top-level keys:", list(checkpoint.keys()))
            candidate = checkpoint.get('model_state', checkpoint.get('state_dict', None))
            if isinstance(candidate, dict):
                print("Model/state_dict keys (sample up to 50):")
                for k in list(candidate.keys())[:50]:
                    v = candidate[k]
                    print("  ", k, type(v), getattr(v, 'shape', None))
            else:
                print("No nested model_state/state_dict found; checkpoint may itself be a state_dict or contain different keys.")
        else:
            print("Checkpoint is not a dict; can't inspect keys further.")
        return

    # Try to obtain a state_dict from common checkpoint layouts:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    # First try strict=True, fall back to strict=False if requested/needed
    try:
        curve_model.load_state_dict(state_dict)
        print("Loaded checkpoint into curve_model with strict=True")
    except RuntimeError as e:
        print("Strict load failed with RuntimeError:", e)
        if args.allow_partial_load:
            print("Trying to load with strict=False (partial load)...")
            res = curve_model.load_state_dict(state_dict, strict=False)
            missing = getattr(res, 'missing_keys', None)
            unexpected = getattr(res, 'unexpected_keys', None)
            print("Partial load completed.")
            if missing:
                print("Missing keys (these were not found in checkpoint and remain as initialized):")
                for k in missing:
                    print(" ", k)
            if unexpected:
                print("Unexpected keys (present in checkpoint but not used by the model):")
                for k in unexpected:
                    print(" ", k)
        else:
            print("To allow partial loads set --allow_partial_load, or inspect the checkpoint with --inspect_ckpt")
            raise

    criterion = F.cross_entropy
    regularizer = utils.l2_regularizer(args.wd)

    def get_xy(point, origin, vector_x, vector_y):
        return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

    w = list()
    curve_parameters = list(curve_model.net.parameters())
    for i in range(args.num_bends):
        w.append(np.concatenate([
            p.data.cpu().numpy().ravel() for p in curve_parameters[i::args.num_bends]
        ]))

    # sanity check
    if not isinstance(w, (list, tuple)) or len(w) < 3:
        raise RuntimeError("Unexpected `w` layout: expected a list/tuple of at least 3 bend vectors. "
                           f"Found type={type(w)} len={len(w) if hasattr(w,'__len__') else 'unknown'}")

    # Ensure w arrays are float32 initially (they usually are, but normalization below can upcast)
    w = [arr.astype(np.float32, copy=False) for arr in w]

    print('Weight space dimensionality: %d' % w[0].shape[0])

    # compute orthonormal basis u, v and keep them float32 to avoid float64 on MPS
    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u = (u / dx).astype(np.float32)

    v = w[1] - w[0]
    v = v - np.dot(u.astype(np.float64), v) * u  # dot may use float64 temporarily
    v = (v / np.linalg.norm(v)).astype(np.float32)
    dy = np.linalg.norm(w[1] - w[0])  # keep dy as float64 or float32 as needed

    bend_coordinates = np.stack([get_xy(p, w[0], u, v) for p in w])

    ts = np.linspace(0.0, 1.0, args.curve_points)
    curve_coordinates = []
    for t in np.linspace(0.0, 1.0, args.curve_points):
        t_input = torch.tensor([t], device=device, dtype=torch.float32)
        weights = curve_model.weights(t_input)
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy().ravel()
        else:
            weights = np.asarray(weights).ravel()
        weights = weights.astype(np.float32, copy=False)
        curve_coordinates.append(get_xy(weights, w[0], u, v))
    curve_coordinates = np.stack(curve_coordinates)

    G = args.grid_points
    alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
    betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

    tr_loss = np.zeros((G, G))
    tr_nll = np.zeros((G, G))
    tr_acc = np.zeros((G, G))
    tr_err = np.zeros((G, G))

    te_loss = np.zeros((G, G))
    te_nll = np.zeros((G, G))
    te_acc = np.zeros((G, G))
    te_err = np.zeros((G, G))

    grid = np.zeros((G, G, 2))

    base_model = architecture.base(num_classes, **architecture.kwargs)
    base_model.to(device)

    columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

    # initial BN update
    update_bn_to_device(loaders['train'], base_model, device)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # build p using float32 arrays so value below will be float32
            p = (w[0] + alpha * dx * u + beta * dy * v).astype(np.float32, copy=False)

            offset = 0
            for parameter in base_model.parameters():
                size = np.prod(parameter.size())
                value = p[offset:offset+size].reshape(parameter.size())
                # ensure the numpy array is float32 (MPS doesn't support float64)
                if value.dtype != np.float32:
                    value = value.astype(np.float32)
                # create a tensor with the same dtype as the parameter and move it to device
                tensor = torch.from_numpy(value).to(device=device, dtype=parameter.data.dtype)
                parameter.data.copy_(tensor)
                offset += size

            # update BN stats for this newly-loaded weight setting
            update_bn_to_device(loaders['train'], base_model, device)

            tr_res = utils.test(loaders['train'], base_model, criterion, regularizer)
            te_res = utils.test(loaders['test'], base_model, criterion, regularizer)

            tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
            te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res['accuracy']

            c = get_xy(p, w[0], u, v)
            grid[i, j] = [alpha * dx, beta * dy]

            tr_loss[i, j] = tr_loss_v
            tr_nll[i, j] = tr_nll_v
            tr_acc[i, j] = tr_acc_v
            tr_err[i, j] = 100.0 - tr_acc[i, j]

            te_loss[i, j] = te_loss_v
            te_nll[i, j] = te_nll_v
            te_acc[i, j] = te_acc_v
            te_err[i, j] = 100.0 - te_acc[i, j]

            values = [
                grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
                te_nll[i, j], te_err[i, j]
            ]
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            if j == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)

    np.savez(
        os.path.join(args.dir, 'plane.npz'),
        ts=ts,
        bend_coordinates=bend_coordinates,
        curve_coordinates=curve_coordinates,
        alphas=alphas,
        betas=betas,
        grid=grid,
        tr_loss=tr_loss,
        tr_acc=tr_acc,
        tr_nll=tr_nll,
        tr_err=tr_err,
        te_loss=te_loss,
        te_acc=te_acc,
        te_nll=te_nll,
        te_err=te_err
    )


if __name__ == '__main__':
    main()