import argparse
from pathlib import Path

import numpy as np
import trimesh

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images


def main():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="DUSt3R two-image point cloud export")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--model", default=str(root / "models" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"))
    parser.add_argument("--image1", default=str(root / "test_images" / "dust3r1.jpg"))
    parser.add_argument("--image2", default=str(root / "test_images" / "dust3r2.jpg"))
    parser.add_argument("--output", default=str(root / "outputs" / "dust3r_pointcloud.ply"))
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
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    print("alignment loss:", loss)

    all_points = []
    all_colors = []
    for i in range(2):
        mask = confidence_masks[i].cpu().numpy().astype(bool)
        points = pts3d[i].detach().cpu().numpy()[mask]
        colors = np.clip(imgs[i][mask] * 255.0, 0, 255).astype(np.uint8)
        all_points.append(points)
        all_colors.append(colors)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    cloud.export(output_path)

    print("exported point cloud to:", output_path)
    print("point count:", len(points))


if __name__ == "__main__":
    main()
