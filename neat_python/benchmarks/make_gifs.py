"""
Build animated GIFs from genome graph snapshots and convert MP4 videos to GIFs.
"""
import os
import sys
import imageio.v2 as imageio
from pathlib import Path

SHOWCASE_DIR = "/home/z/my-project/download/showcase"


def make_genome_evolution_gif(env_dir: str, output_path: str, fps: int = 2) -> None:
    """Combine genome_gen*.png snapshots into an animated GIF."""
    plots_dir = Path(env_dir)
    genome_imgs = sorted(plots_dir.glob("genome_gen*.png"),
                          key=lambda p: int(p.stem.replace("genome_gen", "")))
    if not genome_imgs:
        return
    frames = []
    for img_path in genome_imgs:
        frames.append(imageio.imread(str(img_path)))
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"  {output_path}: {len(frames)} frames")


def make_video_gif(mp4_dir: str, output_path: str, fps: int = 10) -> None:
    """Convert MP4 video to GIF (smaller for embedding)."""
    mp4s = sorted(Path(mp4_dir).glob("*.mp4"))
    if not mp4s:
        return
    # pick the longest video (most informative)
    longest = max(mp4s, key=lambda p: p.stat().st_size)
    try:
        reader = imageio.get_reader(str(longest))
        frames = []
        for frame in reader:
            frames.append(frame)
        reader.close()
        # downsample if too many frames
        max_frames = 60
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step]
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"  {output_path}: {len(frames)} frames (from {longest.name})")
    except Exception as e:
        print(f"  FAILED {mp4_dir}: {e}")


def main():
    gif_dir = os.path.join(SHOWCASE_DIR, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    plots_root = Path(SHOWCASE_DIR) / "plots"
    videos_root = Path(SHOWCASE_DIR) / "videos"

    print("Building genome evolution GIFs...")
    for env_dir in sorted(plots_root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name
        make_genome_evolution_gif(env_dir, os.path.join(gif_dir, f"{env_name}_genome_evolution.gif"))

    print("\nBuilding agent behavior GIFs from videos...")
    for vid_dir in sorted(videos_root.iterdir()):
        if not vid_dir.is_dir():
            continue
        env_name = vid_dir.name
        # find the actual video subfolder
        subdirs = [d for d in vid_dir.iterdir() if d.is_dir()]
        for sd in subdirs:
            make_video_gif(sd, os.path.join(gif_dir, f"{env_name}_agent.gif"))
            break  # only one subdir per env


if __name__ == "__main__":
    main()
