ffmpeg -i vis_049_half.gif -vf "scale=iw/2:ih/2,fps=10" -c:v libx264 -crf 28 -pix_fmt yuv420p vis_049_half_compressed.mp4
