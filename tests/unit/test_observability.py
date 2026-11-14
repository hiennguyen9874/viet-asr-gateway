"""Histogram rendering keeps cumulative buckets, count and sum consistent."""

from app.observability import Metrics


def test_histogram_named_state_preserves_rendered_values():
    metrics = Metrics()
    for value in (0.005, 0.01, 200.0):
        metrics.observe("duration_seconds", "Duration.", value, kind="file")
    rendered = metrics.render().splitlines()
    assert 'gateway_duration_seconds_bucket{kind="file",le="0.005"} 1' in rendered
    assert 'gateway_duration_seconds_bucket{kind="file",le="0.01"} 2' in rendered
    assert 'gateway_duration_seconds_bucket{kind="file",le="120.0"} 2' in rendered
    assert 'gateway_duration_seconds_bucket{kind="file",le="+Inf"} 3' in rendered
    assert 'gateway_duration_seconds_sum{kind="file"} 200.015' in rendered
    assert 'gateway_duration_seconds_count{kind="file"} 3' in rendered
