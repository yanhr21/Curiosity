"""Explicit full-span frame selection for the official causal Wan VAE."""


def prompt_frame_indices() -> list[int]:
    # Nearest integer to linspace(0, 63, 61), computed without floating ties.
    # Select first, then reverse this exact support (never reverse then select).
    return [(index * 63 + 30) // 60 for index in range(61)]


def prompt_coverage_record() -> dict:
    selected = prompt_frame_indices()
    return {"protocol": "wan22_explicit_61_of_64_full_span_v1",
            "source_frame_count": 64, "encoded_frame_count": 61,
            "matched_source_indices": selected,
            "reversed_source_indices": list(reversed(selected)),
            "omitted_source_indices": [i for i in range(64) if i not in selected],
            "same_source_support_under_reversal": True,
            "vae_consumes_every_selected_frame": 1 + 4 * ((len(selected) - 1) // 4) == len(selected)}
