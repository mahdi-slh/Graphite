"""Compatibility shims for newer Open3D / NumPy / SciPy versions.

The original Graphite codebase was written against Open3D ~0.9, NumPy 1.16
and SciPy 1.3.  We monkey-patch the small handful of APIs that have since
been renamed/removed so the rest of the code can run unmodified.

Apply by calling ``apply_compat()`` once - safe to call repeatedly.
"""
from __future__ import annotations

import numpy as np


_APPLIED = False


def _patch_open3d() -> None:
    import open3d as o3d

    pc_cls = o3d.geometry.PointCloud

    if not hasattr(pc_cls, 'select_down_sample'):
        pc_cls.select_down_sample = pc_cls.select_by_index  # type: ignore[attr-defined]

    original_rotate = pc_cls.rotate
    original_translate = pc_cls.translate

    def rotate_compat(self, R, center=None, *args, **kwargs):
        if isinstance(center, (bool, np.bool_)):
            center = (0.0, 0.0, 0.0) if not center else self.get_center()
        if center is None:
            center = (0.0, 0.0, 0.0)
        return original_rotate(self, R, center, *args, **kwargs)

    def translate_compat(self, t, relative=True, *args, **kwargs):
        t = np.asarray(t, dtype=np.float64).reshape(3)
        return original_translate(self, t, relative, *args, **kwargs)

    pc_cls.rotate = rotate_compat  # type: ignore[assignment]
    pc_cls.translate = translate_compat  # type: ignore[assignment]

    # In Open3D 0.10+ the ``registration`` submodule was moved under
    # ``pipelines.registration`` and several signatures changed.  We expose the
    # legacy name + wrap the affected calls so the original code keeps working.
    if not hasattr(o3d, 'registration'):
        import types

        legacy = types.ModuleType('open3d.registration')

        modern = o3d.pipelines.registration
        for name in dir(modern):
            if not name.startswith('_'):
                setattr(legacy, name, getattr(modern, name))

        original_ransac_criteria = modern.RANSACConvergenceCriteria

        def _ransac_criteria_compat(*args, **kwargs):
            # ``max_validation`` was removed in Open3D 0.10+; silently drop it.
            kwargs.pop('max_validation', None)
            return original_ransac_criteria(*args, **kwargs)

        legacy.RANSACConvergenceCriteria = _ransac_criteria_compat

        original_ransac = modern.registration_ransac_based_on_feature_matching

        def _ransac_feature_compat(source, target, source_feature, target_feature,
                                   *args, **kwargs):
            # Old signature did not have ``mutual_filter``; new versions require it.
            mutual_filter = kwargs.pop('mutual_filter', False)
            return original_ransac(source, target, source_feature, target_feature,
                                   mutual_filter, *args, **kwargs)

        legacy.registration_ransac_based_on_feature_matching = _ransac_feature_compat

        o3d.registration = legacy  # type: ignore[attr-defined]
        import sys as _sys
        _sys.modules['open3d.registration'] = legacy


def _patch_scipy_rotation() -> None:
    """``Rotation`` is a C extension and cannot be monkey-patched directly.

    Instead, we patch the few call sites in test_rt / test_match by registering
    a module-level helper which they re-import.  See ``rotation_compat`` below.
    """


def from_dcm_compat(matrix):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(matrix)


def _patch_numpy() -> None:
    if not hasattr(np, 'asscalar'):
        def asscalar(a):
            return np.asarray(a).item()
        np.asscalar = asscalar  # type: ignore[attr-defined]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for name, default in (('int', int), ('float', float), ('bool', bool)):
            if not hasattr(np, name):
                setattr(np, name, default)


def apply_compat() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _patch_numpy()
    try:
        _patch_open3d()
    except Exception as exc:  # pragma: no cover
        print('open3d compat shim skipped:', exc)
    try:
        _patch_scipy_rotation()
    except Exception as exc:  # pragma: no cover
        print('scipy rotation compat shim skipped:', exc)
    _APPLIED = True
