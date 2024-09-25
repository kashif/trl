# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class SCoREConfig(OnlineDPOConfig):
    r"""
    Configuration class for the [`SCoRETrainer`].

    Subclass of [`OnlineDPOConfig`] we can use all its arguments and add the following prompts:

    Parameters:
        correction_instruction (`str`, *optional*, defaults to `"The previous response may contain errors. Please review and correct any mistakes: "`):
            Instruction for self-correction in the second attempt.
        first_attempt_prefix (`str`, *optional*, defaults to `"First attempt: "`):
            Prefix for the first attempt in the second attempt prompt.
        second_attempt_prefix (`str`, *optional*, defaults to `"Improved response: "`):
            Prefix for the second attempt in the model output.
    """

    # Prompts
    correction_instruction: str = field(
        default="The previous response may contain errors. Please review and correct any mistakes: ",
        metadata={"help": "Instruction for self-correction in the second attempt"},
    )

    first_attempt_prefix: str = field(
        default="First attempt: ", metadata={"help": "Prefix for the first attempt in the second attempt prompt"}
    )

    second_attempt_prefix: str = field(
        default="Improved response: ", metadata={"help": "Prefix for the second attempt in the model output"}
    )

    def __post_init__(self):
        super().__post_init__()

        # Ensure that the correction instruction ends with a space
        if not self.correction_instruction.endswith(" "):
            self.correction_instruction += " "
        # Ensure that the prefixes end with a space
        if not self.first_attempt_prefix.endswith(" "):
            self.first_attempt_prefix += " "
        if not self.second_attempt_prefix.endswith(" "):
            self.second_attempt_prefix += " "
