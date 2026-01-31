MODEL_SIZES = {
    "tiny": {
        "cnn_depth": 16,      # Filtros base: 16, 32, 32
        "fc_units": 128,      # Capa tras la CNN
        "gru_units": 128,     # Estado recurrente
        "rnd_units": 128      # Capas del RND
    },
    "small": {
        "cnn_depth": 32,      # Filtros base: 32, 64, 64
        "fc_units": 512,      
        "gru_units": 256,     
        "rnd_units": 512      
    },
    "large": {
        "cnn_depth": 64,      # Filtros base: 64, 128, 128
        "fc_units": 1024,     
        "gru_units": 512,     
        "rnd_units": 1024     
    }
}