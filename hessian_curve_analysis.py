import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import data
import models
import curves
import utils

# ----------------- Globals you can tune -----------------
NUM_POINTS = 21            # number of points along curve to evaluate loss/accuracy/gap
HESSIAN_K = NUM_POINTS     # number of points along curve to compute Hessian metrics (evenly spaced)
HESSIAN_BATCH_SIZE = 128   # batch size used for Hessian computations
HUTCHINSON_SAMPLES = 20    # number of Hutchinson probe vectors
POWER_ITERS = 12           # power iteration iterations for top eigenvalue
HV_BATCHES = 8             # number of minibatches to average over during Hutchinson
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

def compute_eigenvalue_estimate(v, data_loader, base_model, params, criterion, hv_batches, device):
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

    # final eigenvalue estimate: lambda = v^T H v
    # compute H v one more time (averaged)
    hv = hv_acc / float(hv_batches)
    eigenvalue_estimate = (v * hv).sum().item()
    hv_norm = hv.norm()
    if hv_norm.item() == 0:
        return 0.0, 0.0
    v_next = hv.detach() / hv_norm

    return v_next, eigenvalue_estimate

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
        v, eigenvalue_estimate = compute_eigenvalue_estimate(v, data_loader, base_model, params, criterion, hv_batches, device)

    return float(eigenvalue_estimate)

def block_power_iteration_top_k_eigenvalues(base_model, data_loader, criterion, device,
                                           k=5, power_iters=20, hv_batches=1):
    """Estimate top k eigenvalues using Block Power Iteration with HVP.
    Returns: top_k_eigenvalues (list of floats)
    """
    base_model.eval()
    params = [p for p in base_model.parameters() if p.requires_grad]
    param_dim = sum(p.numel() for p in params)
    data_iter = iter(data_loader)

    # Initialize V: a block of k random vectors (D x k matrix)
    V = torch.randn(param_dim, k, device=device, dtype=torch.float32)

    for it in range(power_iters):
        # 1. Compute H V (Hessian applied to all k vectors)
        HV = torch.zeros_like(V)

        # We need to average H V over hv_batches mini-batches
        for b in range(hv_batches):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Compute H v for each of the k vectors in the block
            batch_HV = torch.zeros_like(V)
            for j in range(k):
                v_flat = V[:, j] # j-th column is the j-th vector
                base_model.zero_grad()
                outputs = base_model(inputs)
                loss = criterion(outputs, targets)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

                # hvp_from_grads_and_vector computes H @ v_flat
                hvp_flat = hvp_from_grads_and_vector(grads, params, v_flat)
                batch_HV[:, j] = hvp_flat.detach()

            HV += batch_HV

        HV /= float(hv_batches)

        # 2. Orthonormalize the block V using QR decomposition
        # This prevents the columns of V from collapsing onto the top eigenvector
        Q, _ = torch.linalg.qr(HV)
        V = Q

    # Final estimate of top k eigenvalues (Ritz values)
    # The eigenvalues are approximated by the eigenvalues of the k x k matrix V^T @ H @ V

    # Compute the final averaged H V using the orthonormal basis V
    Final_HV = torch.zeros_like(V)
    data_iter = iter(data_loader) # Restart iterator for final average
    for b in range(hv_batches):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inputs, targets = next(data_iter)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_Final_HV = torch.zeros_like(V)
        for j in range(k):
            v_flat = V[:, j]
            base_model.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            hvp_flat = hvp_from_grads_and_vector(grads, params, v_flat)
            batch_Final_HV[:, j] = hvp_flat.detach()

        Final_HV += batch_Final_HV

    Final_HV /= float(hv_batches)

    # Calculate the k x k matrix A_k = V^T @ H @ V
    A_k = V.transpose(0, 1) @ Final_HV

    # Find the eigenvalues of the small k x k matrix A_k
    eigenvalues = torch.linalg.eigvalsh(A_k) # eigvalsh for symmetric matrix

    # Return the top k (largest) eigenvalues
    top_k_eigenvalues = eigenvalues.flip(0).tolist() # Sort descending

    return top_k_eigenvalues

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
    hess_topk = np.zeros((HESSIAN_K, 5))

    # Prepare a small data loader for Hessian estimation (use training loader but with smaller batch size)
    # We'll sample from the training DataLoader; create a DataLoader with HESSIAN_BATCH_SIZE
    
    
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

        # power iteration for top k eigenvalues
        start = time.time()
        topk_est = block_power_iteration_top_k_eigenvalues(base_model, hess_loader, criterion, device,
                                                             power_iters=POWER_ITERS, hv_batches=HV_BATCHES)
        end = time.time()
        print(f"hess t={tval:.3f} topk[0] ≈ {topk_est[0]:.6e} (time {end-start:.1f}s)")
        hess_topk[k_idx, :] = topk_est

    # Save results
    out_path = os.path.join(out_dir, 'curve_hessian.npz')
    np.savez(out_path,
             ts=ts, tr_nll=tr_nll, te_nll=te_nll, tr_loss=tr_loss, te_loss=te_loss,
             tr_acc=tr_acc, te_acc=te_acc, gen_gap=gen_gap,
             hess_ts=hess_ts, hess_lambda=hess_lambda, hess_trace=hess_trace, hess_topk=hess_topk)
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