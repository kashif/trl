# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Optional

from .sft_config import SFTConfig


@dataclass
class GKDConfig(SFTConfig):
    temperature: float = 1.0
    lmbda: float = 1.0
    max_new_tokens_response: int = 128
    loss_type: str = "kl"
    teacher_model_name_or_path: Optional[str] = None
    teacher_model_init_kwargs: Optional[Dict] = None
    disable_dropout: bool = True