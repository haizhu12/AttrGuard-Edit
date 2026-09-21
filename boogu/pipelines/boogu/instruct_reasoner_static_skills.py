# Modified for the AttrGuard-Edit/Boogu runtime on 2026-09-21; see NOTICE.

"""Static instruction-rewriter prompts for reference-image editing only."""

from textwrap import dedent
from typing import Dict, List, Tuple


RewritePromptStep = Dict[Tuple[str, str], str]


class InstructionReasonerStaticRewriteSkills:
    """Provide built-in and custom prompts for image-edit instruction rewriting."""

    def __init__(self) -> None:
        self.REWRITE_SYSTEM_PROMPT_4_EDIT_EN = dedent(
            """
            You rewrite instructions for reference-image editing. Inspect the
            supplied image and turn the user's request into a precise, visually
            achievable edit while preserving the user's intent.

            Rules:
            - State exactly what should be added, removed, replaced, recolored,
              repaired, restyled, or moved, including position and visible traits
              when the request leaves them ambiguous.
            - Preserve all subjects, identity traits, layout, lighting, background,
              and style that the user did not ask to change.
            - For text edits, quote the exact visible text and keep its language and
              capitalization unless the user explicitly requests a replacement.
            - Resolve contradictions conservatively; never invent unrelated objects
              or turn the request into a new scene.
            - Return only the rewritten editing instruction, with no explanation.
            """
        ).strip()
        self.REWRITE_SYSTEM_PROMPT_4_EDIT_ZH = dedent(
            """
            你负责改写“基于参考图的图像编辑”指令。请观察输入图像，在不改变用户核心意图的
            前提下，把请求改写为精确、可视化且能够执行的编辑说明。

            规则：
            - 明确要添加、删除、替换、改色、修复、转换风格或移动的对象；用户描述含糊时，
              只补充完成该编辑所必需的位置和视觉属性。
            - 保持用户未要求改变的主体、身份特征、构图、光照、背景和风格。
            - 编辑图片文字时，用引号保留准确文字；除非用户明确要求替换，否则不改变语言和大小写。
            - 保守解决矛盾，不添加无关对象，也不把局部编辑扩展为全新场景。
            - 只输出改写后的编辑指令，不要解释或分析。
            """
        ).strip()

        self.rewrite_skills_dict: Dict[str, List[RewritePromptStep]] = {
            "default": [
                {
                    ("zh", "image-editing"): self.REWRITE_SYSTEM_PROMPT_4_EDIT_ZH,
                    ("en", "image-editing"): self.REWRITE_SYSTEM_PROMPT_4_EDIT_EN,
                }
            ]
        }

    def get_default_rewrite_system_prompt(
        self, task_type: str = "image-editing", language: str = "zh"
    ) -> str:
        """Return the built-in editing prompt for Chinese or English."""
        if task_type.lower() != "image-editing":
            raise ValueError("Only image-editing rewrite prompts are available.")
        return (
            self.REWRITE_SYSTEM_PROMPT_4_EDIT_EN
            if language.lower() == "en"
            else self.REWRITE_SYSTEM_PROMPT_4_EDIT_ZH
        )

    def set_custom_rewrite_system_prompts(
        self, custom_rewriter_system_prompts_list: List[str]
    ) -> None:
        """Install one or more custom prompts, all scoped to image editing."""
        if not custom_rewriter_system_prompts_list or not all(
            isinstance(prompt, str) and prompt.strip()
            for prompt in custom_rewriter_system_prompts_list
        ):
            raise ValueError("Custom image-editing rewrite prompts must be non-empty strings.")
        self.rewrite_skills_dict["custom"] = [
            {
                ("zh", "image-editing"): prompt,
                ("en", "image-editing"): prompt,
            }
            for prompt in custom_rewriter_system_prompts_list
        ]

    def get_rewrite_system_prompts_list(
        self, rewriter_system_prompt_type: str = "default"
    ) -> List[RewritePromptStep]:
        """Return an editing prompt chain for the selected preset."""
        prompt_type = rewriter_system_prompt_type.lower()
        if prompt_type not in self.rewrite_skills_dict:
            raise ValueError(
                "Invalid image-editing rewriter preset: "
                f"{rewriter_system_prompt_type!r}. Choose 'default' or configure 'custom'."
            )
        return self.rewrite_skills_dict[prompt_type]
