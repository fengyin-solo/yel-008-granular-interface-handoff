import pytest

CASE_IDS_BY_TEST = {'test_config_schema_and_numeric_validation': ['REQ-001-01', 'REQ-003-01', 'REQ-004-01', 'REQ-007-01', 'REQ-008-01', 'REQ-010-01', 'REQ-014-01', 'REQ-016-01', 'REQ-021-02', 'REQ-025-01', 'REQ-025-02', 'REQ-025-03'], 'test_dense_mass_checks_recover_physical_masses': ['REQ-001-02', 'REQ-003-02', 'REQ-004-02', 'REQ-005-01', 'REQ-006-01', 'REQ-007-02', 'REQ-008-02', 'REQ-009-01', 'REQ-010-02', 'REQ-012-01', 'REQ-014-02', 'REQ-016-02', 'REQ-017-01'], 'test_particle_mass_range_endpoints_are_accepted': ['REQ-027-01'], 'test_box_boundary_reflects_once_and_across_multiple_walls': ['REQ-002-01'], 'test_elastic_collision_conserves_energy_and_momentum': ['REQ-002-02', 'REQ-011-01', 'REQ-013-01'], 'test_mass_checks_require_full_rank': ['REQ-005-02', 'REQ-006-02', 'REQ-009-02', 'REQ-012-02', 'REQ-017-02'], 'test_periodic_seam_collision': ['REQ-011-02', 'REQ-013-02'], 'test_touching_but_not_approaching_particles_do_not_collide': ['REQ-011-03'], 'test_checkpoint_resume_matches_continuous_run': ['REQ-015-01', 'REQ-018-01', 'REQ-023-01'], 'test_invalid_checkpoint_falls_back_to_start': ['REQ-015-02', 'REQ-018-02', 'REQ-023-02'], 'test_entrypoint_and_output_contract': ['REQ-019-01', 'REQ-020-01', 'REQ-022-02', 'REQ-024-01', 'REQ-026-01', 'REQ-026-02', 'REQ-026-03'], 'test_repeat_run_is_byte_identical': ['REQ-019-02', 'REQ-020-02', 'REQ-022-01', 'REQ-024-02'], 'test_invalid_config_preserves_existing_targets': ['REQ-021-01']}

_RESULTS = {}


def pytest_runtest_logreport(report):
    if report.when == "call":
        _RESULTS[report.nodeid.split("::")[-1]] = report.passed


def pytest_sessionfinish(session, exitstatus):
    for test_name, case_ids in CASE_IDS_BY_TEST.items():
        passed = _RESULTS.get(test_name)
        if passed is None:
            continue
        for case_id in case_ids:
            print(("PASS " if passed else "FAIL ") + case_id, flush=True)
