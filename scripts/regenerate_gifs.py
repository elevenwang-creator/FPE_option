"""Regenerate PDF evolution GIFs with slower playback and a hold at the final frame."""
import subprocess, os
from PIL import Image

def regenerate_gif(mp4_path, gif_path, fps=8, hold_sec=8.0, scale=1000):
    tmp_gif = gif_path + ".tmp.gif"
    palette = gif_path + ".palette.png"

    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path, "-vf",
        f"fps={fps},scale={scale}:-1:flags=lanczos,palettegen=stats_mode=full:max_colors=256",
        palette,
    ], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path, "-i", palette,
        "-filter_complex",
        f"[0:v]fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=floyd_steinberg",
        tmp_gif,
    ], check=True, capture_output=True)

    img = Image.open(tmp_gif)
    frames = []
    try:
        while True:
            frames.append(img.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    hold_frames = max(1, int(fps * hold_sec))
    last = frames[-1]
    for _ in range(hold_frames):
        frames.append(last.copy())

    frame_dur = int(1000 / fps)
    durations = [frame_dur] * len(frames)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    os.remove(tmp_gif)
    os.remove(palette)
    print(f"  -> {gif_path}  ({fps} fps, {hold_sec}s hold, {len(frames)} frames, {scale}px)")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "python", "examples")
    for name in ["pdf_evolution", "pdf_evolution_barrier"]:
        mp4 = os.path.join(base, f"{name}.mp4")
        gif = os.path.join(base, f"{name}.gif")
        if os.path.exists(mp4):
            print(f"Regenerating {name}...")
            regenerate_gif(mp4, gif)
        else:
            print(f"Skipping {name}: {mp4} not found")
