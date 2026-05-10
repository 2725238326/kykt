from pathlib import Path
p = Path('/hdd3/kykt26/code/dream3r/dream3r/anchor_bank.py')
text = p.read_text()
replacements = [
('''                 utility_decay: float = 0.99,
                 spatial_bias_alpha: float = 1.0,
                 spatial_retrieval_mode: str = "latent_plus_3d"):
''', '''                 utility_decay: float = 0.99,
                 spatial_bias_alpha: float = 1.0,
                 spatial_retrieval_mode: str = "latent_plus_3d",
                 stability_prune_bonus: float = 1.0):
'''),
('''        self.spatial_bias_alpha = spatial_bias_alpha
        self.spatial_retrieval_mode = spatial_retrieval_mode
''', '''        self.spatial_bias_alpha = spatial_bias_alpha
        self.spatial_retrieval_mode = spatial_retrieval_mode
        self.stability_prune_bonus = stability_prune_bonus
'''),
('''        self.register_buffer("points3d_mean", torch.zeros(1, capacity, 3))
        self.register_buffer("_write_cursor", torch.zeros(1, dtype=torch.long))
''', '''        self.register_buffer("points3d_mean", torch.zeros(1, capacity, 3))
        self.register_buffer("stability_score", torch.zeros(1, capacity))
        self.register_buffer("_write_cursor", torch.zeros(1, dtype=torch.long))
'''),
('''        self.points3d_mean = torch.zeros(batch_size, self.capacity, 3, device=dev)
        self._write_cursor = torch.zeros(batch_size, dtype=torch.long, device=dev)
''', '''        self.points3d_mean = torch.zeros(batch_size, self.capacity, 3, device=dev)
        self.stability_score = torch.zeros(batch_size, self.capacity, device=dev)
        self._write_cursor = torch.zeros(batch_size, dtype=torch.long, device=dev)
'''),
('''                _, evict_order = self.utility[b].topk(n_evict, largest=False)
''', '''                eviction_score = self.utility[b] + self.stability_prune_bonus * self.stability_score[b]
                _, evict_order = eviction_score.topk(n_evict, largest=False)
'''),
('''            self.points3d_mean[b, write_positions] = points3d_mean[b, accepted_idx].float()

            quar_mask = quarantine_new[b, accepted_idx]
''', '''            self.points3d_mean[b, write_positions] = points3d_mean[b, accepted_idx].float()
            self.stability_score[b, write_positions] = 0

            quar_mask = quarantine_new[b, accepted_idx]
'''),
('''            self.utility[b, valid_idx] = scores
''', '''            scores = scores + self.stability_prune_bonus * self.stability_score[b, valid_idx]
            self.utility[b, valid_idx] = scores
'''),
('''            self.points3d_mean[b, prune_idx] = 0
            total_pruned += n_to_prune
''', '''            self.points3d_mean[b, prune_idx] = 0
            self.stability_score[b, prune_idx] = 0
            total_pruned += n_to_prune
'''),
('''    def tick(self):
        self._current_timestep += 1
        self.utility *= self.utility_decay
''', '''    def promote(self, state_tokens: torch.Tensor,
                confidence: Optional[torch.Tensor] = None,
                values: Optional[torch.Tensor] = None,
                points3d_mean: Optional[torch.Tensor] = None) -> WriteResult:
        values = state_tokens if values is None else values
        return self.write(
            keys=state_tokens,
            values=values,
            confidence=confidence,
            points3d_mean=points3d_mean,
        )

    def tick(self):
        self._current_timestep += 1
        self.utility *= self.utility_decay
        stable_mask = self.valid & ~self.quarantined
        self.stability_score = self.stability_score + stable_mask.float()
'''),
('''            "points3d_mean": self.points3d_mean.clone(),
        }
''', '''            "points3d_mean": self.points3d_mean.clone(),
            "stability_score": self.stability_score.clone(),
        }
'''),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f'missing anchor pattern: {old!r}')
    text = text.replace(old, new)
p.write_text(text)
