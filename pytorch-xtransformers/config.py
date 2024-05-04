from types import SimpleNamespace

def get_config(eos_token_id, eos_token):
    config = SimpleNamespace(
        total_samples = 10000,
        valid_size = 0.2,
        batch_size = 8,
        max_length = 512,
        epochs=10,
        lr=0.0001,
        max_grad_norm = 0.5,
        embedding_dimension = 256,
        number_of_heads=4,
        number_of_layers=3,
        dropout_rate=0.1,
        eos_token_id = eos_token_id,
        eos_token = eos_token,
    )
    return config