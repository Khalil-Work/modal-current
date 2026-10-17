"""Repository-root conftest: puts the project directory on ``sys.path``.

Keeping a conftest at the repository root makes ``import modal_current``
work when running ``pytest`` from a source checkout without installing the
package.
"""
