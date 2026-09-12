from representation_extraction.constants import (
    EXTRACTOR_DEFAULTS,
    KNN_DEFAULTS,
    EXTRACTOR_PARAMS,
)

def build_folder_name(extractor_name: str, ext_params: dict) -> str:
    """Gera o nome da subpasta a partir dos hiperparametros do extrator."""
    norm = ext_params.get("normalization", "none")
    if extractor_name in ("tfidf", "bow"):
        mf = ext_params["max_features"]
        ng_min, ng_max = ext_params["ngram_range"]
        md = ext_params["min_df"]
        mxd = ext_params["max_df"]
        return f"mf{mf}_ng{ng_min}-{ng_max}_md{md}_mxd{mxd}_norm-{norm}"
    elif extractor_name == "bert":
        ml = ext_params["max_length"]
        mn = ext_params["model_name"]
        return f"ml{ml}_{mn}_norm-{norm}"
    else:
        raise ValueError(f"Extrator desconhecido: {extractor_name}")


def get_extractor_params(extractor_name: str, args) -> dict:
    """Extrai os hiperparametros do extrator a partir dos args do argparse."""
    defaults = EXTRACTOR_DEFAULTS[extractor_name]
    params = {}
    for key in defaults:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
        else:
            params[key] = defaults[key]
    # Converte ngram_range de lista para tupla se necessario
    if "ngram_range" in params and isinstance(params["ngram_range"], list):
        params["ngram_range"] = tuple(params["ngram_range"])
    return params


def get_knn_params(args) -> dict:
    """Extrai os hiperparametros do KNN a partir dos args do argparse."""
    params = {}
    for key in KNN_DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
        else:
            params[key] = KNN_DEFAULTS[key]
    return params


def validate_extractor_args(extractor_name: str, args):
    """Verifica se o usuario nao passou hiperparametros invalidos para o extrator."""
    if extractor_name == "all":
        return  # Nao valida quando e 'all' (usa defaults)

    valid_params = EXTRACTOR_PARAMS[extractor_name]
    all_extractor_params = set()
    for params in EXTRACTOR_PARAMS.values():
        all_extractor_params.update(params)

    for param in all_extractor_params:
        if param not in valid_params:
            val = getattr(args, param, None)
            if val is not None:
                raise ValueError(
                    f"Hiperparametro '--{param.replace('_', '-')}' nao se aplica ao extrator '{extractor_name}'. "
                    f"Hiperparametros validos para '{extractor_name}': {sorted(valid_params)}"
                )
