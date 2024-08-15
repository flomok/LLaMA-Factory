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
# 处理基于 Pydantic 模型的数据，将其转换为 Python 字典或 JSON 字符串格式。代码中定义了两个函数 dictify 和 jsonify，它们分别用于将 Pydantic 模型实例转换为字典和 JSON 字符串。
# 这两个函数都考虑到了 Pydantic 的不同版本（v1 和 v2）可能带来的 API 差异，因此在代码中使用了异常处理机制来兼容这两种版本。

import json
from typing import TYPE_CHECKING, Any, Dict


if TYPE_CHECKING:
    from pydantic import BaseModel


def dictify(data: "BaseModel") -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)
