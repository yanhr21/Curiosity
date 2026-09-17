"""Only failed whole-snapshot charts may use observed local-pad fallback."""
from .report_fixture_touch import snapshot_chart
from .active3d_local_charts import supported_local_chart


def snapshot_with_local_fallback(surface,hand_pose,template,faces):
    arrays,original=snapshot_chart(surface,hand_pose,template,faces)
    if original['available']:
        return arrays,original
    arrays,local=snapshot_chart(surface,hand_pose,template,faces,chart_builder=supported_local_chart)
    local.update(adapter='failed_snapshot_local_support_v1',original_snapshot_failure=original)
    return arrays,local
