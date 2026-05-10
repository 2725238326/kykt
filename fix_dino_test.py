from pathlib import Path
p = Path('/hdd3/kykt26/code/dream3r/dream3r/tests/test_dinov2_backbone.py')
text = p.read_text()
text = text.replace('return tokens * base.unsqueeze(-1) * self.weight', 'return tokens * base * self.weight')
p.write_text(text)
