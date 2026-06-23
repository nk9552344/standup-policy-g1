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


https://github.com/nk9552344/standup-policy-g1.git

python -m mjlab.scripts.train Mjlab-StayStand-Flat-Unitree-G1 --gpu-ids None

 uv run play Mjlab-StayStand-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_staystand/2026-06-21_21-58-46/model_4000.pt \ 
  --num-envs 2

uv run train Mjlab-StayStand-Unitree-G1 --gpu-ids None

uv run play Mjlab-StayStand-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_stay_stand/2026-06-22_23-04-28/model_550.pt --num-envs 2

uv run play Mjlab-StayStand-Rough-Unitree-G1 \ 
  --checkpoint-file logs/rsl_rl/g1_staystand/2026-06-22_17-01-29/model_0.pt

https://github.com/InternRobotics/HoST

https://wandb.ai/nk9552344-infosys/mjlab/runs/rawno0s9?nw=nwusernk9552344

