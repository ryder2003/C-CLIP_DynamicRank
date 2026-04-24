"""
C-CLIP with MAB-driven Dynamic Rank Selection.

Key changes vs. the original cclip.py:
  1. CCLIP.__init__ now accepts a `rank_bandit` (LoRARankBandit) instead of a
     fixed `lora_r`.  The bandit lives here so the training loop can call
     `model.bandit.select_rank(task_idx)` and `model.bandit.update(rank, reward)`.

  2. inject_lora_for_new_task(task_idx, task_name) accepts the runtime rank
     chosen by the bandit instead of always using the constructor's lora_r.

  3. A new helper `compute_lora_utilisation()` measures how much each rank
     component actually contributed (Frobenius-norm based, inspired by DoRA).
     This can optionally be folded into the bandit reward.

Everything else (projectors, CKC loss, merge, checkpoint I/O) is unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import copy

from .clip_wrapper import CLIPWrapper
from .lora import (
    inject_lora,
    merge_all_lora_weights,
    get_lora_parameters,
    count_lora_parameters,
    LoRALayer,
    LoRAForAttn,
)
from .rank_bandit import LoRARankBandit


class Projector(nn.Module):
    """Near-identity projector for CKC knowledge consolidation."""

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CCLIPWithBandit(nn.Module):
    """
    C-CLIP that uses a Multi-Armed Bandit to choose LoRA rank per task.

    Args:
        clip_model_name   : OpenCLIP model string
        pretrained        : pretrained weights identifier
        rank_bandit       : LoRARankBandit instance (owns rank selection logic)
        lora_alpha        : LoRA scaling factor (kept fixed; only rank varies)
        lora_dropout      : dropout inside LoRA layers
        lora_target_modules : which modules to inject LoRA into
        integration_coeff : merging coefficient after each task
        device            : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        rank_bandit: Optional[LoRARankBandit] = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        integration_coeff: float = 0.7,
        device: str = "cuda",
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ['q_proj', 'v_proj', 'c_fc', 'c_proj']

        # --- Bandit setup ---
        if rank_bandit is None:
            rank_bandit = LoRARankBandit(
                rank_choices=[4, 8, 16, 32],
                algorithm="ucb1",
            )
        self.bandit = rank_bandit

        # Fixed LoRA hyper-params (rank is variable, these stay constant)
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.integration_coeff = integration_coeff
        self.device = device

        # Current task's chosen rank (set by inject_lora_for_new_task)
        self.current_lora_r: int = 0

        # Load base CLIP
        self.clip = CLIPWrapper(
            model_name=clip_model_name,
            pretrained=pretrained,
            device=device,
        )
        self.embed_dim = self.clip.embed_dim

        # Projectors for CKC
        self.vision_projector = Projector(self.embed_dim).to(device)
        self.text_projector = Projector(self.embed_dim).to(device)

        # Old frozen model for CKC
        self.old_clip = None

        # LoRA layer registry
        self.lora_layers = {}

        # Task counter
        self.current_task = 0

        print(f"Initialised CCLIPWithBandit | embed_dim={self.embed_dim}")
        print(f"Bandit: {self.bandit}")

    # ------------------------------------------------------------------ #
    #  Task lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def inject_lora_for_new_task(self, task_idx: int, task_name: str = "") -> int:
        """
        Ask the bandit which rank to use, then inject LoRA with that rank.

        Returns the chosen rank (so the training loop can store it for later
        reward feedback).
        """
        print(f"\n{'='*60}")
        print(f"Starting Task {task_idx + 1} — {task_name}")
        print(f"{'='*60}")

        # 1. Bandit picks rank
        chosen_rank = self.bandit.select_rank(task_idx, task_name)
        self.current_lora_r = chosen_rank

        # Dynamic alpha: scale alpha with rank so effective magnitude stays ~constant
        # (per DoRA insight: lora_alpha / r is the actual scaling factor)
        dynamic_alpha = 2 * chosen_rank   # keeps scaling_factor = 2.0 always

        print(f"[CCLIPWithBandit] Injecting LoRA with r={chosen_rank}, "
              f"alpha={dynamic_alpha}")

        # 2. Save old model for CKC (from task 1 onwards)
        if self.current_task > 0:
            self.old_clip = copy.deepcopy(self.clip)
            self.old_clip.eval()
            for param in self.old_clip.parameters():
                param.requires_grad = False
            print("  Saved frozen old model for CKC")

            # Re-init projectors to near-identity
            nn.init.eye_(self.vision_projector.projection.weight)
            nn.init.zeros_(self.vision_projector.projection.bias)
            nn.init.eye_(self.text_projector.projection.weight)
            nn.init.zeros_(self.text_projector.projection.bias)

        # 3. Freeze base model
        self.clip.freeze_base_model()

        # 4. Inject LoRA into vision encoder
        vision_lora = inject_lora(
            model=self.clip.model.visual,
            target_modules=self.lora_target_modules,
            r=chosen_rank,
            lora_alpha=dynamic_alpha,
            lora_dropout=self.lora_dropout,
        )

        # 5. Inject LoRA into text encoder
        text_lora = inject_lora(
            model=self.clip.model.transformer,
            target_modules=self.lora_target_modules,
            r=chosen_rank,
            lora_alpha=dynamic_alpha,
            lora_dropout=self.lora_dropout,
        )

        self.lora_layers = (
            {f"visual.{k}": v for k, v in vision_lora.items()}
            | {f"transformer.{k}": v for k, v in text_lora.items()}
        )

        n_params = count_lora_parameters(self.clip.model)
        print(f"  LoRA rank    : {chosen_rank}")
        print(f"  LoRA params  : {n_params:,}")

        self.current_task += 1
        return chosen_rank

    def merge_lora_after_task(self):
        """Merge LoRA weights into backbone after task completes."""
        print(f"\n=== Finishing Task {self.current_task} — merging LoRA ===")
        self.clip.model = merge_all_lora_weights(
            model=self.clip.model,
            lora_layers=self.lora_layers,
            integration_coeff=self.integration_coeff,
        )
        self.lora_layers = {}
        self.clip.visual = self.clip.model.visual
        self.clip.model = self.clip.model.to(self.device)
        print("LoRA merged.")

    def update_bandit(
        self,
        rank: int,
        task_idx: int,
        task_name: str,
        task_accuracy: float,
        baseline_accuracy: float,
        zeroshot_retention: float,
        zeroshot_baseline: float,
        extra_info: Optional[Dict] = None,
    ):
        """
        Compute composite reward and feed it back to the bandit.
        Call this AFTER training + evaluation for the task.

        Args:
            rank                : rank that was used (returned by inject_lora_for_new_task)
            task_accuracy       : val accuracy on current task after training (0-1)
            baseline_accuracy   : pretrained zero-shot on same task           (0-1)
            zeroshot_retention  : zero-shot on reference set after training   (0-1)
            zeroshot_baseline   : zero-shot on reference set before training  (0-1)
        """
        # Optionally incorporate LoRA utilisation (DoRA insight):
        # if some rank components are barely contributing, a smaller rank
        # might have been sufficient → penalise over-provisioning slightly
        utilisation = self.compute_lora_utilisation()
        if extra_info is None:
            extra_info = {}
        extra_info["lora_utilisation"] = utilisation
        extra_info["chosen_rank"] = rank

        reward = self.bandit.compute_reward(
            task_accuracy=task_accuracy,
            baseline_accuracy=baseline_accuracy,
            zeroshot_retention=zeroshot_retention,
            zeroshot_baseline=zeroshot_baseline,
        )

        # Optional: slight penalty if utilisation is low (rank wastage)
        # Uncomment if you want DoRA-style efficiency feedback:
        # reward = reward * (0.8 + 0.2 * utilisation)

        self.bandit.update(
            rank=rank,
            reward=reward,
            task_idx=task_idx,
            task_name=task_name,
            extra_info=extra_info,
        )

    # ------------------------------------------------------------------ #
    #  LoRA utilisation (DoRA-inspired)                                   #
    # ------------------------------------------------------------------ #

    def compute_lora_utilisation(self) -> float:
        """
        Measure what fraction of LoRA rank components are 'active'
        (Frobenius-norm of each rank slice normalised by the total).

        Returns a scalar in [0,1]:
          1.0 = all rank components contribute equally (efficient use of budget)
          0.0 = only one component dominates (could have used lower rank)

        Inspired by DoRA's importance scoring: s_i = ||A_i B_i||_F / ||sum_j A_j B_j||_F
        """
        importance_scores = []

        for module in self.clip.model.modules():
            if isinstance(module, LoRALayer):
                # LoRALayer: contribution of each rank slice
                # delta_W = B @ A,  B: (out, r), A: (r, in)
                r = module.r
                slice_norms = []
                for i in range(r):
                    # i-th rank-1 component: B[:,i] outer A[i,:]
                    bi = module.lora_B[:, i]          # (out,)
                    ai = module.lora_A[i, :]          # (in,)
                    norm = (bi.unsqueeze(1) * ai.unsqueeze(0)).norm().item()
                    slice_norms.append(norm)
                total = sum(slice_norms) + 1e-8
                # Entropy-like uniformity: max entropy = log(r), actual = -sum p*log(p)
                probs = [n / total for n in slice_norms]
                entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                max_entropy = math.log(r) if r > 1 else 1.0
                importance_scores.append(entropy / max_entropy)

            elif isinstance(module, LoRAForAttn):
                # LoRAForAttn: Q and V components separately
                r = module.r
                for lora_A, lora_B in [
                    (module.lora_q_A, module.lora_q_B),
                    (module.lora_v_A, module.lora_v_B),
                ]:
                    slice_norms = []
                    for i in range(r):
                        bi = lora_B[:, i]
                        ai = lora_A[i, :]
                        norm = (bi.unsqueeze(1) * ai.unsqueeze(0)).norm().item()
                        slice_norms.append(norm)
                    total = sum(slice_norms) + 1e-8
                    probs = [n / total for n in slice_norms]
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                    max_entropy = math.log(r) if r > 1 else 1.0
                    importance_scores.append(entropy / max_entropy)

        if not importance_scores:
            return 1.0

        utilisation = sum(importance_scores) / len(importance_scores)
        print(f"[CCLIPWithBandit] LoRA utilisation (rank entropy): {utilisation:.3f}")
        return utilisation

    # ------------------------------------------------------------------ #
    #  Standard forward / encode methods (unchanged from original)         #
    # ------------------------------------------------------------------ #

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.clip.encode_image(images, normalize=normalize)

    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.clip.encode_text(text, normalize=normalize)

    def forward(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        return_old_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        proj_img = F.normalize(self.vision_projector(image_features), p=2, dim=-1)
        proj_txt = F.normalize(self.text_projector(text_features), p=2, dim=-1)

        output = {
            'image_features': image_features,
            'text_features': text_features,
            'projected_image_features': proj_img,
            'projected_text_features': proj_txt,
        }

        if return_old_features and self.old_clip is not None:
            with torch.no_grad():
                output['old_image_features'] = self.old_clip.encode_image(images, normalize=True)
                output['old_text_features'] = self.old_clip.encode_text(text, normalize=True)

        return output

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        params.extend(get_lora_parameters(self.clip.model))
        params.extend(self.vision_projector.parameters())
        params.extend(self.text_projector.parameters())
        return params

    # ------------------------------------------------------------------ #
    #  Checkpoint I/O                                                      #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: str):
        checkpoint = {
            'clip_state_dict': self.clip.model.state_dict(),
            'vision_projector_state_dict': self.vision_projector.state_dict(),
            'text_projector_state_dict': self.text_projector.state_dict(),
            'current_task': self.current_task,
            'lora_layers': list(self.lora_layers.keys()),
            'bandit_summary': self.bandit.summary(),
            'current_lora_r': self.current_lora_r,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        has_lora = any('lora_A' in k or 'original_layer' in k
                       for k in checkpoint['clip_state_dict'])
        if has_lora:
            r = checkpoint.get('current_lora_r', 16)
            self.inject_lora_for_new_task(task_idx=0)
            self.clip.model.load_state_dict(checkpoint['clip_state_dict'], strict=False)
        else:
            self.clip.model.load_state_dict(checkpoint['clip_state_dict'])

        self.vision_projector.load_state_dict(checkpoint['vision_projector_state_dict'])
        self.text_projector.load_state_dict(checkpoint['text_projector_state_dict'])
        self.current_task = checkpoint.get('current_task', 0)
        self.clip.visual = self.clip.model.visual
        print(f"Loaded checkpoint from {path} (task={self.current_task})")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_cclip_with_bandit(config: Dict, device: str = "cuda") -> CCLIPWithBandit:
    """
    Build CCLIPWithBandit from the existing config dict format.
    Add a 'bandit' sub-dict to your YAML to control the bandit, e.g.:

        bandit:
          rank_choices: [4, 8, 16, 32]
          algorithm: ucb1          # ucb1 | epsilon_greedy | thompson
          plasticity_w: 0.6
          stability_w: 0.4
          ucb_c: 2.0
          epsilon: 0.3
          epsilon_decay: 0.85
          save_path: checkpoints/bandit_state.json
    """
    bandit_cfg = config.get('bandit', {})
    bandit = LoRARankBandit(
        rank_choices=bandit_cfg.get('rank_choices', [4, 8, 16, 32]),
        algorithm=bandit_cfg.get('algorithm', 'ucb1'),
        plasticity_w=bandit_cfg.get('plasticity_w', 0.6),
        stability_w=bandit_cfg.get('stability_w', 0.4),
        epsilon=bandit_cfg.get('epsilon', 0.3),
        epsilon_decay=bandit_cfg.get('epsilon_decay', 0.85),
        ucb_c=bandit_cfg.get('ucb_c', 2.0),
        save_path=bandit_cfg.get('save_path', None),
    )

    model_cfg = config['model']
    return CCLIPWithBandit(
        clip_model_name=model_cfg.get('clip_model_name', 'ViT-B-16'),
        pretrained=model_cfg.get('pretrained', 'openai'),
        rank_bandit=bandit,
        lora_alpha=model_cfg.get('lora_alpha', 32),
        lora_dropout=model_cfg.get('lora_dropout', 0.1),
        lora_target_modules=model_cfg.get('lora_target_modules', None),
        integration_coeff=model_cfg.get('integration_coeff', 0.7),
        device=device,
    )
