import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as pl

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.utils.image import load_images


def parse_args():
    root = Path(__file__).resolve().parent
    return argparse.ArgumentParser(description="DUSt3R two-image match visualization").parse_args()


def main():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="DUSt3R two-image match visualization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--n-viz", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--model", default=str(root / "models" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"))
    parser.add_argument("--image1", default=str(root / "test_images" / "dust3r1.jpg"))
    parser.add_argument("--image2", default=str(root / "test_images" / "dust3r2.jpg"))
    parser.add_argument("--output", default=str(root / "outputs" / "dust3r_matches.png"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AsymmetricCroCo3DStereo.from_pretrained(args.model).to(args.device)

    images = load_images([args.image1, args.image2], size=args.size)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=args.batch_size)

    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(
        init="mst",
        niter=args.niter,
        schedule=args.schedule,
        lr=args.lr,
    )

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    print("alignment loss:", loss)
    print("focals:", focals)
    print("poses shape:", poses.shape if hasattr(poses, "shape") else type(poses))

    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])

    reciprocal_in_p2, nn2_in_p1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f"found {num_matches} matches")

    matches_im1 = pts2d_list[1][reciprocal_in_p2]
    matches_im0 = pts2d_list[0][nn2_in_p1][reciprocal_in_p2]

    n_viz = min(args.n_viz, num_matches)
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0 = matches_im0[match_idx_to_viz]
    viz_matches_im1 = matches_im1[match_idx_to_viz]

    h0, w0, h1, w1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(h1 - h0, 0)), (0, 0), (0, 0)), "constant", constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(h0 - h1, 0)), (0, 0), (0, 0)), "constant", constant_values=0)
    merged = np.concatenate((img0, img1), axis=1)

    pl.figure(figsize=(14, 8))
    pl.imshow(merged)
    cmap = pl.get_cmap("jet")
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + w0], [y0, y1], "-+", color=cmap(i / max(n_viz - 1, 1)), scalex=False, scaley=False)

    pl.tight_layout()
    pl.savefig(output_path, dpi=200, bbox_inches="tight")
    print("saved match visualization to:", output_path)


if __name__ == "__main__":
    main()
