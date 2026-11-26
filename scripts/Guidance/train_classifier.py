from protgen.models.transformer_freeze import load_frozen_transformer
from protgen.models.classification_guidance import FamilyHead

frozen_transformer = load_frozen_transformer("weights/transformer_best.pt", config.hardware)
model = FamilyHead(frozen_transformer, config.model_layout.embedding_size, config.model_layout.hidden_units).to(config.hardware)