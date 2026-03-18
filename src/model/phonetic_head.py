"""
Phonetic constraint head.
A lightweight 2-layer MLP trained to predict the phoneme of the last token.

At inference: used to rescore beam candidates — penalizes tokens that would
violate the target end-rhyme phoneme or beat stress pattern.

Architecture:
  hidden_state (d_model) → Linear(d_model, 512) → GELU → Linear(512, PHONEME_VOCAB_SIZE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.dual_tokenizer import PHONEME_VOCAB_SIZE, PHONEME_TO_ID


class PhoneticHead(nn.Module):
    def __init__(self, d_model: int = 4096, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, PHONEME_VOCAB_SIZE),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, d_model)
        Returns logits: (batch, seq_len, PHONEME_VOCAB_SIZE)
        """
        return self.net(hidden_states)

    def predict_last_phoneme(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, d_model)
        Returns predicted phoneme IDs for the last token: (batch,)
        """
        last = hidden_states[:, -1, :]        # (batch, d_model)
        logits = self.net(last)               # (batch, PHONEME_VOCAB_SIZE)
        return logits.argmax(dim=-1)


# ── Constraint scoring for beam search ────────────────────────────────────────

class PhoneticConstraintScorer:
    """
    Wraps PhoneticHead and provides beam candidate re-scoring.
    """

    def __init__(self, head: PhoneticHead, device: str = "cpu"):
        self.head = head.to(device)
        self.head.eval()
        self.device = device

    @torch.no_grad()
    def score_candidates(
        self,
        hidden_states: torch.Tensor,             # (beam, seq_len, d_model)
        target_end_phoneme_id: int,              # ID of the target rhyme phoneme
        stress_targets: list[int] | None = None, # per-position stress (0 or 1)
        rhyme_weight: float = 2.0,
        stress_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns a penalty tensor of shape (beam,) — higher penalty = worse candidate.
        Call this on fully-generated candidate sequences before ranking.
        """
        logits = self.head(hidden_states)  # (beam, seq_len, PH_VOCAB)

        # Rhyme penalty: how unlikely is the target end-phoneme at the last position?
        last_logits = logits[:, -1, :]  # (beam, PH_VOCAB)
        last_probs = F.softmax(last_logits, dim=-1)
        rhyme_prob = last_probs[:, target_end_phoneme_id]  # (beam,)
        rhyme_penalty = -rhyme_weight * torch.log(rhyme_prob.clamp(min=1e-9))

        penalty = rhyme_penalty

        # Stress penalty (if stress targets provided)
        if stress_targets is not None:
            stress_id_1 = PHONEME_TO_ID.get("AH1", 1)  # stressed vowel proxy
            for pos, target_stress in enumerate(stress_targets):
                if pos >= logits.shape[1]:
                    break
                if target_stress == 1:
                    pos_probs = F.softmax(logits[:, pos, :], dim=-1)
                    # Stressed tokens end with "1" in ARPAbet
                    stressed_ids = [
                        i for tok, i in PHONEME_TO_ID.items() if tok.endswith("1")
                    ]
                    if stressed_ids:
                        stress_prob = pos_probs[:, stressed_ids].sum(dim=-1)
                        penalty += stress_weight * (-torch.log(stress_prob.clamp(min=1e-9)))

        return penalty  # (beam,)

    def rerank_beams(
        self,
        beam_hidden_states: torch.Tensor,
        beam_scores: torch.Tensor,           # (beam,) from LLM beam search
        target_end_phoneme_id: int,
        alpha: float = 0.5,                  # blend: alpha * LLM score + (1-alpha) * phonetic
    ) -> list[int]:
        """
        Returns sorted beam indices (best first).
        """
        penalty = self.score_candidates(beam_hidden_states, target_end_phoneme_id)
        # Normalize LLM scores (higher = better), penalty (higher = worse)
        lm_norm = (beam_scores - beam_scores.mean()) / (beam_scores.std() + 1e-9)
        ph_norm = (penalty - penalty.mean()) / (penalty.std() + 1e-9)
        combined = alpha * lm_norm - (1 - alpha) * ph_norm
        return combined.argsort(descending=True).tolist()


# ── Training loss ─────────────────────────────────────────────────────────────

def phonetic_head_loss(
    logits: torch.Tensor,      # (batch, seq_len, PH_VOCAB)
    phoneme_ids: torch.Tensor, # (batch, seq_len) — ground truth phoneme IDs
    ignore_index: int = 0,     # PAD_PH id
) -> torch.Tensor:
    """Cross-entropy loss for phonetic head training."""
    batch, seq, vocab = logits.shape
    return F.cross_entropy(
        logits.reshape(batch * seq, vocab),
        phoneme_ids.reshape(batch * seq),
        ignore_index=ignore_index,
    )


if __name__ == "__main__":
    # Smoke test
    head = PhoneticHead(d_model=256, hidden=128)  # small for test
    dummy_hidden = torch.randn(4, 10, 256)  # batch=4, seq=10, d_model=256
    logits = head(dummy_hidden)
    print(f"Logits shape: {logits.shape}")  # (4, 10, ~162)

    scorer = PhoneticConstraintScorer(head)
    beam_scores = torch.tensor([-1.2, -1.5, -0.8, -2.1])
    target_id = PHONEME_TO_ID.get("EY1", 10)
    ranked = scorer.rerank_beams(dummy_hidden, beam_scores, target_id)
    print(f"Reranked beam order: {ranked}")
