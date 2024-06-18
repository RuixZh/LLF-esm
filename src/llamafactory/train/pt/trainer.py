from types import MethodType
from typing import TYPE_CHECKING, Dict, Optional

from transformers import Trainer
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler

if TYPE_CHECKING:
    import torch
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.processor = processor
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def compute_rmsd(
        true_atom_pos: torch.Tensor,
        pred_atom_pos: torch.Tensor,
        atom_mask: torch.Tensor = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Function to calculate RMSD between predicted and ground truth atom position

        Args:
            true_atom_pos: a [nres*3] tensor
            pred_atom_pos: a [nres*3] tensor
            atom_mask: a [1*nres] tensor

        Return:
            RMSD value between true and predicted atom positions
        """
        sq_diff = torch.square(true_atom_pos - pred_atom_pos).sum(dim=-1, keepdim=False)
        if atom_mask is not None:
            sq_diff = torch.masked_select(sq_diff, atom_mask.to(sq_diff.device))
        msd = torch.mean(sq_diff)
        msd = torch.nan_to_num(msd, nan=1e8)
        return torch.sqrt(msd + eps)  # prevent sqrt 0

    def save_outputs_to_pdb(self, model, inputs):
        labels = inputs.pop("labels")
        linker_mask = inputs.pop("linker_mask")
        idx = inputs.pop("idx")

        outputs = model(**inputs)
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        output_dir = os.path.join(self.args.output_dir, 'pdb_out')
        os.makedirs(output_dir, exist_ok=True)
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            out = to_pdb(pred)
            with open(output_dir/f'{idx}.pdb','w') as f:
                f.write(out)
            # pdbs.append()
        # return pdbs

    def evaluation(self, eval_dataset, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and save output files.
        """
        results = super().evaluation(eval_dataset, ignore_keys, metric_key_prefix)

        # Save output files
        self.save_outputs_to_pdb(results)

        return results

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        linker_mask = inputs.pop("linker_mask")
        idx = inputs.pop("idx")

        outputs = model(**inputs)
        outputs['atom37_atom_exists'] = outputs['atom37_atom_exists'] * linker_mask
        atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        loss = compute_rmsd(labels, atom_positions)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

        if self.finetuning_args.link_latest:
            from pathlib import Path
            latest_path = Path('/'.join(output_dir.split('/')[:-1])) / "latest"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(output_dir.split('/')[-1], target_is_directory=True)
            except FileExistsError:
                if latest_path.resolve().name != output_dir:
                    raise
