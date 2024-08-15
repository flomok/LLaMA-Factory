# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 功能主要用于模型的加载、比较和验证，特别是在涉及训练和推理时。
# compare_model: 比较两个模型的权重，检查指定的权重键是否在给定的容差范围内相等。可以用于验证模型的状态是否一致。
# check_lora_model: 验证 Lora 模型的参数，检查是否符合预期的类型和要求，区分了线性模块和额外模块，并检查参数的 requires_grad 属性和数据类型。
# load_train_model 和 load_infer_model: 加载用于训练和推理的模型，使用 get_train_args 和 get_infer_args 函数来获取模型和训练参数，然后加载相应的模型和分词器。load_train_model 函数还可以选择是否添加价值头（valuehead）。
# load_reference_model: 根据给定路径加载参考模型，支持使用 LoRA（Low-Rank Adaptation）和 PISSA（PISA Scale Adaptation）策略，并可选择是否添加价值头（valuehead）。
# load_train_dataset: 加载训练数据集，使用 get_dataset 函数和模型分词器来获取数据集。
# patch_valuehead_model: 为 AutoModelForCausalLMWithValueHead 模型添加一个 post_init 方法，以便正确加载模型的价值头（valuehead）的状态字典。
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Set, Tuple, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from ..data import get_dataset
from ..extras.misc import get_current_device
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraModel
    from transformers import PreTrainedModel


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module", diff_keys: Sequence[str] = []) -> None:
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        if any(key in name for key in diff_keys):
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is False
        else:
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is True


def check_lora_model(model: "LoraModel") -> Tuple[Set[str], Set[str]]:
    linear_modules, extra_modules = set(), set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            linear_modules.add(name.split(".lora_", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        elif "modules_to_save" in name:
            extra_modules.add(name.split(".modules_to_save", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    return linear_modules, extra_modules


def load_train_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    model_args, _, _, finetuning_args, _ = get_train_args(kwargs)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    return load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=add_valuehead)


def load_infer_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    model_args, _, finetuning_args, _ = get_infer_args(kwargs)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    return load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=add_valuehead)


def load_reference_model(
    model_path: str,
    lora_path: Optional[str] = None,
    use_lora: bool = False,
    use_pissa: bool = False,
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> Union["PreTrainedModel", "LoraModel"]:
    if add_valuehead:
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=get_current_device()
        )
        if not is_trainable:
            model.v_head = model.v_head.to(torch.float16)

        return model

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=get_current_device()
    )
    if use_lora or use_pissa:
        model = PeftModel.from_pretrained(
            model, lora_path, subfolder="pissa_init" if use_pissa else None, is_trainable=is_trainable
        )
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def load_train_dataset(**kwargs) -> "Dataset":
    model_args, data_args, training_args, _, _ = get_train_args(kwargs)
    tokenizer_module = load_tokenizer(model_args)
    dataset_module = get_dataset(model_args, data_args, training_args, stage=kwargs["stage"], **tokenizer_module)
    return dataset_module["train_dataset"]


def patch_valuehead_model():
    def post_init(self: "AutoModelForCausalLMWithValueHead", state_dict: Dict[str, "torch.Tensor"]) -> None:
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys() if k.startswith("v_head.")}
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

    AutoModelForCausalLMWithValueHead.post_init = post_init
