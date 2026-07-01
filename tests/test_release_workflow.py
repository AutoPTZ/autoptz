"""Release workflow invariants that protect update integrity."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _release_workflow() -> str:
    return RELEASE_WORKFLOW.read_text(encoding="utf-8")


def test_macos_intel_is_required_release_artifact() -> None:
    workflow = _release_workflow()
    assert "macos-intel:" in workflow
    assert "runs-on: macos-26-intel" in workflow
    assert "continue-on-error" not in workflow
    assert "needs: [macos, macos-intel, windows, linux]" in workflow
    assert "needs['macos-intel'].result == 'success'" in workflow


def test_checksums_generated_after_all_artifacts_downloaded() -> None:
    workflow = _release_workflow()
    download_idx = workflow.index("uses: actions/download-artifact@v4")
    checksum_idx = workflow.index("Generate SHA256SUMS")
    publish_idx = workflow.index("uses: softprops/action-gh-release@v2")
    assert download_idx < checksum_idx < publish_idx
    assert "SHA256SUMS" in workflow[publish_idx:]
