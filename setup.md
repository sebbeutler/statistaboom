deno init --npm vite@latest web-platform--template solid-ts

micromamba create -n stats -c conda-forge python=3.11
micromamba activate stats
pip install -R requirements.txt
