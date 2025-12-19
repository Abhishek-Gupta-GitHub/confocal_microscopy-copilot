# copilot/chat_explainer.py

import json
from dataclasses import dataclass
from typing import Dict

@dataclass
class ChatExplainer:
    client: object  # your LLM client (OpenAI, Groq, etc.)
    model: str = "gpt-4o-mini"

    def build_prompt(self, focus_question: str, summary: Dict) -> str:
        summary_json = json.dumps(summary, indent=2)

        return f"""
You are a microscopy and soft-matter physicist.

Task:
- Interpret the following analysis summary from a 3D confocal particle-tracking experiment.
- Write a concise, scientifically accurate explanation (2–4 short paragraphs).
- Comment on: signal quality (SNR), tracking quality, MSD (diffusive vs sub/superdiffusive),
  depth/bleaching trends, and any structure (RDF / crowding metrics).
- Use approximate quantitative numbers when present (e.g. D, alpha, plateau MSD, peaks in g(r)).
- Do not restate JSON keys; synthesise and interpret the physics.

User focus:
{focus_question or "Describe the main physics in this dataset for a soft-matter audience."}

Analysis summary (JSON, already pre-processed):
{summary_json}
"""
    def explain(self, focus_question: str, summary: Dict) -> str:
        prompt = self.build_prompt(focus_question, summary)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are a concise, rigorous microscopy and soft-matter physics explainer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()



    def call_llm(self, prompt: str) -> str:
        if self.llm is None:
            return (
                "MSD and the fitted alpha parameter indicate the overall diffusive "
                "behaviour under your current imaging conditions. Depth and bleaching "
                "diagnostics highlight where intensity and track quality degrade.\n\n"
                "Next experiments: (1) Restrict analysis to the depth range with "
                "stable intensity, (2) lower laser power or shorten acquisition to "
                "reduce bleaching, (3) adjust particle density or magnification to "
                "mitigate crowding and improve tracking."
            )
        return self.llm.complete(prompt)

    def explain(self, user_question: str, summary: dict) -> str:
        prompt = self.build_prompt(user_question, summary)
        return self.call_llm(prompt)

