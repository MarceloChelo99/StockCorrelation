import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


def text_box_failures(layout_path: Path) -> list[str]:
    data = json.loads(layout_path.read_text())
    failures: list[str] = []
    for element in data.get("elements", []):
        if not element.get("textLayout"):
            continue
        bbox = element.get("bbox") or [0, 0, 0, 0]
        layout = element["textLayout"]
        height = float(layout.get("height") or 0)
        line_widths = [float(line.get("width") or 0) for line in layout.get("lines", [])]
        max_width = max(line_widths, default=0.0)
        box_w = float(bbox[2])
        box_h = float(bbox[3])
        name = element.get("name") or element.get("textPreview") or "text"
        if height > box_h + 1:
            failures.append(f"{layout_path.name}: {name} text height {height:.1f} > box {box_h:.1f}")
        if max_width > box_w + 1:
            failures.append(f"{layout_path.name}: {name} line width {max_width:.1f} > box {box_w:.1f}")
    return failures


def main() -> None:
    failures: list[str] = []
    for layout_path in sorted(SCRATCH.glob("pptx_slide_*.layout.json")):
        failures.extend(text_box_failures(layout_path))

    report = {"checked_layouts": len(list(SCRATCH.glob("pptx_slide_*.layout.json"))), "failures": failures}
    (SCRATCH / "layout-text-check.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("\n".join(failures))
    print(SCRATCH / "layout-text-check.json")


if __name__ == "__main__":
    main()
