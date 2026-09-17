"""Explicit, independently qualified controller interventions; no default change."""
from importlib import import_module


INTERVENTIONS = {
    'regional_bilateral_alignment_v1': (
        'regional_bilateral_alignment', 'RegionalBilateralAlignmentController',
        'Relative to bilateral_common_alignment_v1, only geometry support changes: fit each observed pad with original thresholds and select the greatest-area eligible pad per hand. Normal, pivot and readiness use that support; closure and gain-admission effects follow the unchanged parent formulas. Complete hand loads and original sensor recordings are retained.',
        ('common_alignment_valid', 'common_alignment_applied', 'common_alignment_axis_w',
         'common_alignment_target_error_deg', 'observed_normal_opposition_error_deg',
         'unilateral_alignment_suppressed', 'actual_alignment_active',
         'regional_selected_pad', 'regional_selected_area_m2', 'regional_total_area_m2',
         'regional_selected_count', 'regional_selected_rms_m', 'regional_selected_center_hand_m',
         'regional_selected_pca_normal_hand', 'regional_valid_candidate_count',
         'regional_selection_changed', 'regional_input_hand_load_n',
         'regional_candidate_area_m2', 'regional_candidate_count',
         'regional_candidate_rms_m', 'regional_candidate_valid')),
    'bilateral_common_alignment_v1': (
        'bilateral_common_alignment', 'BilateralCommonAlignmentController',
        'Relative to observed_normal_bisector_alignment_v1, only suppress unilateral fallback alignment; shared normal-bisector alignment requires both observed fits valid. Closure, gains, actual-plane readiness and v2 runtime motion unchanged.',
        ('common_alignment_valid', 'common_alignment_applied', 'common_alignment_axis_w',
         'common_alignment_target_error_deg', 'observed_normal_opposition_error_deg',
         'unilateral_alignment_suppressed', 'actual_alignment_active')),
    'freeze_postlift_alignment_v1': (
        'freeze_postlift_alignment', 'FreezePostliftAlignmentController',
        'Relative to legacy_runtime_path_v1, only freeze alignment after the initial lift-admission step. Preserve fixed four-second path, prescribed yaw/lift, closure, force feedback and all original final checks. Runtime can advance outside the readiness band.',
        ('sustained_ready', 'motion_rate', 'actualmotion_rate',
         'postlift_alignment_frozen', 'postlift_alignment_suppressed', 'actual_alignment_active')),
    'legacy_runtime_path_v1': (
        'fixed_path_surface_grip', 'FixedPathSurfaceGripController',
        'Only runtime motion returns to the original ungated four-second path; initial 1s admission, continued alignment, force feedback and final physical checks retained. Runtime may advance outside the load band.',
        ('sustained_ready', 'motion_rate')),
    'observed_contact_axis_closure_v1': (
        'coupled_surface_grip', 'CoupledSurfaceGripController',
        'Only closure direction follows the measured contact-center axis; all v2 orientation and motion rules retained', ()),
    'continuous_phase_governor_v1': (
        'phase_governor_surface_grip', 'PhaseGovernorSurfaceGripController',
        'Only runtime phase governor changes: same instantaneous readiness band, continuous load margin and 2/s maximum rate increase; original initial 1s admission and final criteria retained',
        ('sustained_ready', 'motion_rate', 'phase_governor_raw_rate', 'phase_governor_rate')),
    'observed_normal_bisector_alignment_v1': (
        'coupled_alignment_surface_grip', 'CoupledAlignmentSurfaceGripController',
        'Only orientation target uses the measured normal bisector; closure and actual local-plane readiness remain original v2',
        ('common_alignment_valid', 'common_alignment_applied', 'common_alignment_axis_w',
         'common_alignment_target_error_deg', 'observed_normal_opposition_error_deg')),
}


INTERVENTION_DEPENDENCIES = {
    'regional_bilateral_alignment_v1': ('bilateral_common_alignment',
        'coupled_alignment_surface_grip', 'alignment_rollback', 'inspect_contact_surface'),
    'bilateral_common_alignment_v1': ('coupled_alignment_surface_grip', 'alignment_rollback'),
    'freeze_postlift_alignment_v1': ('fixed_path_surface_grip', 'alignment_rollback',
                                    'coupled_alignment_surface_grip'),
}


def intervention_source_modules(name):
    return (INTERVENTIONS[name][0], 'controller_interventions',
            *INTERVENTION_DEPENDENCIES.get(name, ()))


def controller_class(name):
    module, cls, _, _ = INTERVENTIONS[name]
    result = getattr(import_module('.'+module, __package__), cls)
    if result.intervention_name != name:
        raise ValueError('Controller intervention name mismatch')
    return result
