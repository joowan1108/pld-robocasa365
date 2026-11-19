# Vast.ai template "On-start Script" for this project (paste the lines below, without code fences).
# Clones/updates the repo and starts the unattended run (log: tail -f /workspace/autorun.log).
cd /workspace
if [ ! -d residual ]; then git clone https://github.com/joowan1108/pld-robocasa365.git residual; fi
cd residual && git pull --ff-only || true
nohup bash scripts/autorun.sh >> /workspace/autorun.log 2>&1 &
