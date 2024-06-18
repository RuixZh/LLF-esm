from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List
import pandas as pd
from io import StringIO
from Bio.PDB import PDBParser
import numpy as np

atom_types =[
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ...hparams import DataArguments

def convert_pdb_tensor(data_args, ID, FILE_PATH):

    atom_dict = {k:i for i,k in enumerate(atom_types)}
    # Create a PDBParser object
    parser = PDBParser()
    # Read the PDB file
    structure = parser.get_structure(ID, FILE_PATH)
    # Access the structure
    groundtruth = torch.zeros((data_args.max_length, len(atom_types), 3))
    model = structure[0]
    residue_idx = 0
    for chain in model:
        for residue in chain:
            residue_name = residue.get_resname()
            residue_number = residue.get_id()[1]
            if (residue_idx + 1 != residue_number) and (residue_number == 1):
                residue_idx +=  data_args.G_len
            for atom in residue:
                atom_name = atom.get_name()
                atom_coords = atom.get_coord()
                if atom_name not in atom_dict:
                    continue
                groundtruth[residue_idx,atom_dict[atom_name]] = torch.tensor(atom_coords)
            residue_idx += 1
    return groundtruth

def preprocess_esm_dataset(
    df, tokenizer, data_args
):
    x = df['sequence']
    # protein_length = len(x.split(':')[0]), len(x.split(':')[0])
    x = x.replace(':', 'G'*data_args.G_len)
    tokenized_homodimer = tokenizer(x, add_special_tokens=False, padding=True)
    label = convert_pdb_tensor(data_args, df['id'], df['path'])
    tokenized_homodimer['labels'] = label
    linker_mask = torch.tensor([1] * df['len1'] + [0] * data_args.G_len + [1] * df['len2'])[None, :, None]
    tokenized_homodimer['linker_mask'] = linker_mask
    tokenized_homodimer['idx'] = [df['id']]

    # position_ids = torch.arange(len(x), dtype=torch.long)
    # position_ids[protein_length + data_args.G_len:] += 512
    # tokenized_homodimer['position_ids'] = position_ids.unsqueeze(0)

    return tokenized_homodimer


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    text_examples = [messages[0]["content"] + eos_token for messages in examples["prompt"]]

    if not data_args.packing:
        if data_args.template == "gemma":
            text_examples = [tokenizer.bos_token + example for example in text_examples]

        result = tokenizer(text_examples, add_special_tokens=False, max_length=data_args.cutoff_len, truncation=True)
    else:
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if data_args.template == "gemma":
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id

    return result
