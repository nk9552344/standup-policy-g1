"""Standup task on-policy runner.

Forked from the velocity locomotion runner.py. No behavioral changes were
needed: this class only handles checkpoint saving + ONNX export + metadata
attachment for whatever policy was trained, none of which inspects
observations, rewards, or commands -- it's training infrastructure, not
task logic. Only the class name changed, for consistency with the rest of
the mjlab.tasks.standup module naming.

One thing worth verifying on your end (not visible from this file alone):
get_base_metadata(self.env.unwrapped, run_name) reads from the unwrapped
env, and if its implementation in exporter_utils.py happens to assume a
specific command term name/shape (e.g. bakes a "twist" velocity command's
ranges into exported metadata for downstream deployment tooling), that
would be a real coupling point worth checking, since this runner file alone
can't reveal that.
"""

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class StandupOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")