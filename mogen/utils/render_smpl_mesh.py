import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse

import imageio
import numpy as np
import pyrender
import smplx
import torch
import trimesh
from PIL import Image, ImageDraw


def load_smpl_mesh_sequence(npz_path, bm_path, device='cuda'):
    data = np.load(npz_path, allow_pickle=True)
    poses = data['poses']  # [T, 72]
    trans = data['trans']  # [T, 3]
    betas = data['betas'] if 'betas' in data.files else np.zeros(10, dtype=np.float32)
    gender = str(data['gender']) if 'gender' in data.files else 'neutral'
    fps = float(data['mocap_framerate']) if 'mocap_framerate' in data.files else 30.0
    caption = str(data['caption']) if 'caption' in data.files else ''

    rots = poses[:, :66].reshape(-1, 22, 3)  # root + 21 body joints, axis-angle
    num_frames = rots.shape[0]

    bm = smplx.create(bm_path, model_type='smplx', num_betas=10, gender=gender,
                       use_pca=False, flat_hand_mean=True,
                       batch_size=num_frames).to(device)

    bparam = {
        'transl': torch.from_numpy(trans).float().to(device),
        'global_orient': torch.from_numpy(rots[:, 0]).float().to(device),
        'body_pose': torch.from_numpy(rots[:, 1:]).float().to(device).reshape(-1, 63),
        'betas': torch.from_numpy(betas).float().to(device).unsqueeze(0).expand(num_frames, -1),
    }
    with torch.no_grad():
        body = bm(return_verts=True, **bparam)

    vertices = body.vertices.detach().cpu().numpy()  # [T, V, 3]
    joints = body.joints[:, :22].detach().cpu().numpy()  # [T, 22, 3]
    faces = bm.faces  # [F, 3]
    return vertices, joints, faces, fps, caption


def render_mesh_video(vertices, faces, root_xy, save_path, fps=30.0,
                       resolution=(768, 768), cam_dist=3.0, cam_height=1.0,
                       caption=''):
    verts = vertices.copy()
    verts[:, :, 2] -= verts[:, :, 2].min()  # ground the feet (data axis 2 is height, not 1)
    verts[:, :, 0] -= root_xy[:, 0:1]  # keep the character centered each frame
    verts[:, :, 1] -= root_xy[:, 1:2]

    # This SMPL-X output is Z-up; rotate -90 deg about X so it matches
    # pyrender/OpenGL's Y-up camera convention (x, z, -y) -> (x, y, z).
    verts = np.stack([verts[..., 0], verts[..., 2], -verts[..., 1]], axis=-1)

    body_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, roughnessFactor=0.7, baseColorFactor=(0.55, 0.65, 0.85, 1.0))
    floor_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.85, 0.85, 0.85, 1.0))

    floor_size = 6.0
    floor_verts = np.array([
        [-floor_size, 0, -floor_size],
        [-floor_size, 0, floor_size],
        [floor_size, 0, floor_size],
        [floor_size, 0, -floor_size],
    ])
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]])
    floor_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=floor_verts, faces=floor_faces, process=False),
        material=floor_material, smooth=False)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=resolution[0] / resolution[1])
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, cam_height],
        [0.0, 0.0, 1.0, cam_dist],
        [0.0, 0.0, 0.0, 1.0],
    ])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.4, 0.4, 0.4))
    scene.add(floor_mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(*resolution)
    frames = []
    body_node = None
    for t in range(verts.shape[0]):
        if body_node is not None:
            scene.remove_node(body_node)
        body_mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices=verts[t], faces=faces, process=False),
            material=body_material, smooth=True)
        body_node = scene.add(body_mesh)

        color, _ = renderer.render(scene)
        if caption:
            frame = Image.fromarray(color)
            ImageDraw.Draw(frame).text((10, 10), caption, fill=(0, 0, 0))
            color = np.asarray(frame)
        frames.append(color)
    renderer.delete()

    imageio.mimwrite(save_path, frames, fps=fps, quality=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render a full SMPL-X mesh video from a MoLingo demo .npz")
    parser.add_argument("-i", "--npz", type=str, required=True, help="Path to a molingo_sample*_repeat*.npz")
    parser.add_argument("-b", "--bm_path", type=str, default='mogen/body_models',
                        help="Folder that contains the smplx/ body-model subfolder")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output mp4 path (default: <npz path>_mesh.mp4)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--resolution", type=int, nargs=2, default=(768, 768))
    args = parser.parse_args()

    out_path = args.out or os.path.splitext(args.npz)[0] + '_mesh.mp4'

    vertices, joints, faces, fps, caption = load_smpl_mesh_sequence(args.npz, args.bm_path, device=args.device)
    root_xy = joints[:, 0, [0, 1]]  # the two horizontal axes; axis 2 is height
    render_mesh_video(vertices, faces, root_xy, out_path, fps=fps,
                       resolution=tuple(args.resolution), caption=caption)
    print(f"Saved full SMPL mesh video -> {out_path} (caption: {caption})")
