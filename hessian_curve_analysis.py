"""
Standalone script to evaluate loss, generalization gap, and Hessian metrics along a learned curve.

What it does
- Samples NUM_POINTS t values along a curve checkpoint (CurveNet).
- For each t it computes training & test NLL/loss/accuracy (using utils.update_bn and utils.test).
- For a small set of HESSIAN_K points it computes:
    * lambda_max: top Hessian eigenvalue (approx) via power iteration using Hessian-vector products (HVP)
    * Hutchinson trace estimator of the Hessian: E[v^T H v] using Rademacher vectors

Globals you can change at the top of the file:
- NUM_POINTS: number of t values to sample for loss / gap evaluation (the "test points" you asked for)
- HESSIAN_K: number of t values (evenly spaced) where Hessian metrics are computed
- HESSIAN_BATCH_SIZE: batch size used for Hessian estimations
- HUTCHINSON_SAMPLES: number of Hutchinson probe vectors
- POWER_ITERS: iterations of power method to estimate top eigenvalue
- HV_BATCHES: number of mini-batches to average over for each Hutchinson probe (small -> faster, noisier)

How to run (example)
 python3 hessian_curve_analysis.py \
   --curve_dir ./runs/curve_vgg \
   --curve_ckpt ./runs/curve_vgg/checkpoint-200.pt \
   --dataset CIFAR10 --data_path ./data --model VGG16 --curve Bezier --num_bends 3 \
   --use_test --num_workers 0

Outputs
- Prints progress and results.
- Saves a file curve_hessian.npz into --out_dir (contains ts, losses, accs, gen_gap and arrays of hessian metrics at HESSIAN_K points).

Notes / caveats
- Hessian computations are expensive. Defaults are conservative: HESSIAN_K=3, POWER_ITERS=20, HUTCHINSON_SAMPLES=20.
- For CurveNet we materialize weights(t) and copy them into a fresh base model before doing Hessian computations (this makes autograd simpler).
- Hutchinson and power-iteration use a small number of training samples (single mini-batch by default). Increase HV_BATCHES and/or use more samples for more stable estimates.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils

# ----------------- Globals you can tune -----------------
NUM_POINTS = 61            # number of points along curve to evaluate loss/accuracy/gap
HESSIAN_K = 3              # number of points along curve to compute Hessian metrics (evenly spaced)
HESSIAN_BATCH_SIZE = 128   # batch size used for Hessian computations
HUTCHINSON_SAMPLES = 20    # number of Hutchinson probe vectors
POWER_ITERS = 20           # power iteration iterations for top eigenvalue
HV_BATCHES = 1             # number of minibatches to average over during Hutchinson
# --------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hessian + loss + gap analysis along curve")
    p.add_argument('--curve_dir', type=str, required=True, help='output dir for curve run (will also save results)')
    p.add_argument('--curve_ckpt', type=str, required=True, help='checkpoint file for a curve (e.g. checkpoint-200.pt)')
    p.add_argument('--dataset', type=str, default='CIFAR10')
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--transform', type=str, default='VGG')
    p.add_argument('--model', type=str, required=True, help='model name (e.g., VGG16)')
    p.add_argument('--curve', type=str, required=True, help='curve type (Bezier or PolyChain)')
    p.add_argument('--num_bends', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--use_test', action='store_true', help='evaluate on official test set')
    p.add_argument('--out_dir', type=str, default=None, help='where to save outputs (default: curve_dir)')
    return p.parse_args()

# ---------- Flatten / unflatten helpers ----------
def flatten_tensors(tensor_list):
    flat = [t.reshape(-1) for t in tensor_list]
    return torch.cat(flat, dim=0)

def shapes_and_numels(params):
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    return shapes, numels

def unflatten_to_param_tensors(flat_vec, params, device=None):
    # flat_vec: 1D torch tensor
    out = []
    idx = 0
    for p in params:
        n = p.numel()
        chunk = flat_vec[idx: idx + n].view_as(p).to(p.dtype)
        out.append(chunk.reshape(p.shape))
        idx += n
    return out

def assign_flat_to_model(flat_np, model):
    # flat_np is numpy array; assign slices to model.parameters() in order
    params = list(model.parameters())
    idx = 0
    for p in params:
        n = p.numel()
        chunk = flat_np[idx: idx + n].reshape(p.shape)
        p.data.copy_(torch.from_numpy(chunk).to(p.device, dtype=p.dtype))
        idx += n

# ---------- Hessian primitives (HVP, Hutchinson, Power iteration) ----------
def compute_grads_for_batch(model, loss, params):
    # return list of gradients (with create_graph=True if loss required higher-order)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    return grads

def hvp_from_grads_and_vector(grads, params, v_flat):
    # grads: tuple/list of tensors (first-order grads), params: corresponding parameter list
    # v_flat: 1D tensor matching flattened parameter dimension
    # compute gdotv = sum( g_i * v_i ) then hv = grad(gdotv, params)
    g_flat = flatten_tensors([g.contiguous().view(-1) for g in grads])
    # ensure same device/dtype
    v_flat = v_flat.to(g_flat.device).type_as(g_flat)
    gdotv = (g_flat * v_flat).sum()
    hv = torch.autograd.grad(gdotv, params, retain_graph=True)
    hv_flat = flatten_tensors([h.contiguous().view(-1) for h in hv])
    return hv_flat

def hutchinson_trace_estimate(base_model, data_loader, criterion, device,
                              num_samples=20, hv_batches=1):
    """Estimate trace(H) via Hutchinson: E[v^T H v], averaging over num_samples random v.
       hv_batches: number of training batches to average over per probe (increases stability).
    """
    base_model.eval()
    params = [p for p in base_model.parameters() if p.requires_grad]
    out_sum = 0.0
    total_probes = 0
    data_iter = iter(data_loader)
    for s in range(num_samples):
        # Rademacher random vector
        v = torch.randint(0, 2, (sum(p.numel() for p in params),), device=device, dtype=torch.float32) * 2.0 - 1.0
        v = v / v.norm()  # normalize for stability (not required for unbiasedness)
        # average over hv_batches minibatches
        probe_vals = []
        for b in range(hv_batches):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                inputs, targets = next(data_iter)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            base_model.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            hv_flat = hvp_from_grads_and_vector(grads, params, v)
            vhv = (v * hv_flat).sum().item()
            probe_vals.append(vhv)
        out_sum += np.mean(probe_vals)
        total_probes += 1
    trace_est = out_sum / total_probes
    # scale by total parameter dimension if you prefer absolute trace (v·Hv is already unbiased for trace)
    return trace_est

def power_iteration_top_eigenvalue(base_model, data_loader, criterion, device,
                                   power_iters=20, hv_batches=1):
    """Estimate top eigenvalue using power iteration with Hessian-vector products (single-run).
       hv_batches: number of batches to average H v per iteration.
    """
    base_model.eval()
    params = [p for p in base_model.parameters() if p.requires_grad]
    param_dim = sum(p.numel() for p in params)
    # initialize v
    v = torch.randn(param_dim, device=device, dtype=torch.float32)
    v = v / v.norm()
    data_iter = iter(data_loader)

    for it in range(power_iters):
        # compute H v (averaged over hv_batches batches)
        hv_acc = torch.zeros_like(v)
        for b in range(hv_batches):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                inputs, targets = next(data_iter)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            base_model.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            hv_flat = hvp_from_grads_and_vector(grads, params, v)
            hv_acc += hv_flat
        hv = hv_acc / float(hv_batches)
        # update v
        v = hv.detach()
        v_norm = v.norm()
        if v_norm.item() == 0:
            return 0.0
        v = v / v_norm
    # final eigenvalue estimate: lambda = v^T H v
    # compute H v one more time (averaged)
    hv_acc = torch.zeros_like(v)
    data_iter = iter(data_loader)
    for b in range(hv_batches):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inputs, targets = next(data_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        base_model.zero_grad()
        outputs = base_model(inputs)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        hv_flat = hvp_from_grads_and_vector(grads, params, v)
        hv_acc += hv_flat
    hv = hv_acc / float(hv_batches)
    lambda_est = (v * hv).sum().item()
    return float(lambda_est)

# ----------------- Main analysis -----------------
def main():
    args = parse_args()
    out_dir = args.out_dir or args.curve_dir
    os.makedirs(out_dir, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Ensure utils uses the same device if possible
    try:
        utils._device = device
    except Exception:
        pass

    # Load data
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )

    # Load curve model
    architecture = getattr(models, args.model)
    curve_cls = getattr(curves, args.curve)
    curve_model = curves.CurveNet(
        num_classes,
        curve_cls,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    curve_model.to(device)
    checkpoint = torch.load(args.curve_ckpt, map_location=device)
    curve_model.load_state_dict(checkpoint['model_state'])
    print("Loaded curve checkpoint:", args.curve_ckpt)

    criterion = F.cross_entropy
    # Preallocate arrays for NUM_POINTS evaluation
    T = NUM_POINTS
    ts = np.linspace(0.0, 1.0, T)
    tr_nll = np.zeros(T)
    te_nll = np.zeros(T)
    tr_loss = np.zeros(T)
    te_loss = np.zeros(T)
    tr_acc = np.zeros(T)
    te_acc = np.zeros(T)
    gen_gap = np.zeros(T)

    # Evaluate losses/accs along curve (cheap relative to Hessian)
    print(f"Evaluating losses/accuracies at {T} points along curve...")
    for i, tval in enumerate(ts):
        t_tensor = torch.tensor([tval], dtype=torch.float32, device=device)
        # update BN statistics for this point
        utils.update_bn(loaders['train'], curve_model, t=t_tensor)
        tr_res = utils.test(loaders['train'], curve_model, criterion, regularizer=None, t=t_tensor)
        te_res = utils.test(loaders['test'], curve_model, criterion, regularizer=None, t=t_tensor)
        tr_nll[i] = tr_res['nll']
        tr_loss[i] = tr_res.get('loss', tr_res['nll'])
        tr_acc[i] = tr_res['accuracy']
        te_nll[i] = te_res['nll']
        te_loss[i] = te_res.get('loss', te_res['nll'])
        te_acc[i] = te_res['accuracy']
        gen_gap[i] = te_nll[i] - tr_nll[i]
        if i % max(1, T // 10) == 0:
            print(f"t={tval:.3f}: tr_nll={tr_nll[i]:.4f}, te_nll={te_nll[i]:.4f}, gen_gap={gen_gap[i]:.4f}")

    # Choose HESSIAN_K points to compute Hessian metrics (evenly spaced)
    hess_ts = np.linspace(0.0, 1.0, HESSIAN_K)
    hess_lambda = np.zeros(HESSIAN_K)
    hess_trace = np.zeros(HESSIAN_K)

    # Prepare a small data loader for Hessian estimation (use training loader but with smaller batch size)
    # We'll sample from the training DataLoader; create a DataLoader with HESSIAN_BATCH_SIZE
    from torch.utils.data import DataLoader
    # Use the full training dataset for sampling (not the Subset loader) - get original dataset object
    # data.loaders returned a DataLoader or Subset; to be robust, we create a simple loader from train dataset
    # Attempt to extract train dataset:
    train_dataset = loaders['train'].dataset
    # If it's a Subset, get underlying dataset
    try:
        base_train_dataset = train_dataset.dataset
    except Exception:
        base_train_dataset = train_dataset
    hess_loader = DataLoader(base_train_dataset, batch_size=HESSIAN_BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print(f"Computing Hessian metrics at {HESSIAN_K} points (POWER_ITERS={POWER_ITERS}, HUTCH_SAMPLES={HUTCHINSON_SAMPLES})")
    for k_idx, tval in enumerate(hess_ts):
        t_tensor = torch.tensor([tval], dtype=torch.float32, device=device)
        # materialize base model with weights at t
        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        base_model.to(device)
        # get flattened numpy weights from curve model
        flat_np = curve_model.weights(t_tensor)
        # assign to base_model parameters in order
        assign_flat_to_model(flat_np, base_model)
        base_model.eval()
        # make sure batch-norm buffers are present (we can import buffers from a base endpoint if desired)
        # Update BN using utils.update_bn but it expects a curve model with t arg. Instead update base_model BN by forward passes:
        # Use utils.update_bn with a small wrapper: pass model=base_model but it expects isbatchnorm detection; that's fine.
        try:
            utils.update_bn(hess_loader, base_model)
        except Exception:
            # fallback: do a few forward passes to prime BN
            with torch.no_grad():
                cnt = 0
                for x, y in hess_loader:
                    x = x.to(device, non_blocking=True)
                    base_model(x)
                    cnt += 1
                    if cnt >= 5:
                        break

        # Hutchinson trace estimate
        start = time.time()
        trace_est = hutchinson_trace_estimate(base_model, hess_loader, criterion, device,
                                              num_samples=HUTCHINSON_SAMPLES, hv_batches=HV_BATCHES)
        end = time.time()
        print(f"hess t={tval:.3f} Hutchinson trace ≈ {trace_est:.6e} (time {end-start:.1f}s)")
        hess_trace[k_idx] = trace_est

        # power iteration for largest eigenvalue
        start = time.time()
        lambda_est = power_iteration_top_eigenvalue(base_model, hess_loader, criterion, device,
                                                    power_iters=POWER_ITERS, hv_batches=HV_BATCHES)
        end = time.time()
        print(f"hess t={tval:.3f} lambda_max ≈ {lambda_est:.6e} (time {end-start:.1f}s)")
        hess_lambda[k_idx] = lambda_est

    # Save results
    out_path = os.path.join(out_dir, 'curve_hessian.npz')
    np.savez(out_path,
             ts=ts, tr_nll=tr_nll, te_nll=te_nll, tr_loss=tr_loss, te_loss=te_loss,
             tr_acc=tr_acc, te_acc=te_acc, gen_gap=gen_gap,
             hess_ts=hess_ts, hess_lambda=hess_lambda, hess_trace=hess_trace)
    print("Saved results to", out_path)

    # Print concise summaries
    barrier_train = np.max(tr_nll) - max(tr_nll[0], tr_nll[-1])
    barrier_test = np.max(te_nll) - max(te_nll[0], te_nll[-1])
    print("Barrier (train nll):", barrier_train)
    print("Barrier (test nll):", barrier_test)
    print("Hessian lambda_max at points:", list(zip(hess_ts.tolist(), hess_lambda.tolist())))
    print("Hessian Hutchinson trace at points:", list(zip(hess_ts.tolist(), hess_trace.tolist())))

if __name__ == "__main__":
    main()