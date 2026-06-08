"""Regenerate PDF evolution GIFs with slower playback and a hold at the final frame."""
import subprocess, os, sys

def regenerate_gif(mp4_path, gif_path, fps=8, hold_sec=2.0):
    tmp_gif = gif_path + ".tmp.gif"
    palette = gif_path + ".palette.png"

    # Generate palette for better quality
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", f"fps={fps},scale=800:-1:flags=lanczos,palettegen=stats_mode=diff",
        palette
    ], check=True, capture_output=True)

    # Convert to GIF with palette, then add hold at end via tpad
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path, "-i", palette,
        "-filter_complex", (
            f"[0:v]fps={fps},scale=800:-1:flags=lanczos[x];"
            f"[x][1:v]paletteuse=dither=bayer[b];"
            f"[b]tpad=stop_mode=clone:stop_duration={hold_sec}"
        ),
        tmp_gif
    ], check=True, capture_output=True)

    os.replace(tmp_gif, gif_path)
    os.remove(palette)
    print(f"  -> {gif_path}  ({fps} fps, {hold_sec}s hold at end)")

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
