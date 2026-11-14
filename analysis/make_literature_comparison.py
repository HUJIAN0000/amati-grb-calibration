"""Build the constant-slope comparison and parameter table from sourced values and fit summaries."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LATEX = ROOT.parent / "generated"
# Published marginal results. Source tables and distance conventions are in README.md.
PUBLISHED = (
    ("Cosmological background fitted", (
        (r"Khadka et al.\ \citep{Khadka2021}", r"Flat \(\Lambda\)CDM, GRBs alone", 118, r"1.207\pm0.091"),
        ("", r"Flat \(\Lambda\)CDM, GRBs + BAO + \(H(z)\)", 118, r"1.228\pm0.088"),
        (r"Cao et al.\ \citep{CaoKhadkaRatra2022}", r"Flat \(\Lambda\)CDM, GRBs alone", 118, r"1.110\pm0.090"),
    )),
    ("Distances reconstructed from observations", (
        (r"Liang et al.\ \citep{Liang2022}", r"Pantheon GP, \(z<1.4\)", 20, r"1.01\pm0.14"),
        (r"Li et al.\ \citep{LiZhangLiang2023}", r"\(H(z)\) GP, D'Agostini, \(z<1.4\)", 20, r"1.00\pm0.21"),
        ("", r"\(H(z)\) GP, Reichart, \(z<1.4\)", 20, r"1.57^{+0.22}_{-0.42}"),
        (r"Han et al.\ \citep{Han2024}", r"\(H(z)\) GP, \(z<1.965\)", 48, r"1.18\pm0.14"),
        ("", r"Pantheon+ GP, \(z<1.965\)", 48, r"1.20\pm0.14"),
    )),
)
SAMPLES = (("a118", "Full A118", 118), ("reference", "Matched GRBs", 83))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(param, digits=3):
    median, plus, minus = (f"{param[key]:.{digits}f}" for key in ("median", "plus", "minus"))
    if plus == minus:
        return rf"{median}\pm{plus}"
    return rf"{median}^{{+{plus}}}_{{-{minus}}}"


def row(study, calibration, count, slope):
    return rf"{study} & {calibration} & {count} & " + "$" + slope + "$" + r" \\"


def section(title):
    return rf"\addlinespace[2pt]\multicolumn{{4}}{{@{{}}l}}{{\itshape {title}}} \\"


def write_table(filename, content):
    (LATEX / filename).write_text("\n".join(content) + "\n", encoding="utf-8")


def main():
    LATEX.mkdir(exist_ok=True)
    comparison = [r"\begin{tabular}{@{}llrc@{}}", r"\toprule",
                  r"Study & Distance calibration / GRB selection & \(N\) & \(b\) \\",
                  r"\midrule"]
    for title, rows in PUBLISHED:
        comparison.append(section(title))
        comparison.extend(row(*entry) for entry in rows)

    comparison.append(section(r"This work: fixed flat \(\Lambda\)CDM"))
    parameters = [r"\begin{tabular}{@{}lrccc@{}}", r"\toprule",
                  r"Selection & \(N\) & \(a_{\rm G}\) & \(b\) & \(\sigma_{\rm G}\) (mag) \\",
                  r"\midrule"]
    for name, label, count in SAMPLES:
        folder = ROOT / "benchmarks" / name
        config = read(folder / "config.json")["model"]
        if (config["n_grb"], config["H0"], config["Omega_m"]) != (count, 70., .3):
            raise ValueError(f"Unexpected fixed-background configuration for {name}")
        post = read(folder / "summary.json")["parameters"]
        comparison.append(row("", label, count, interval(post["b"])))
        values = " & ".join("$" + interval(post[key]) + "$" for key in ("a_GRB", "b", "sigma_GRB"))
        parameters.append(rf"{label} & {count} & {values} \\")

    comparison.append(section("This work: relative pair likelihood"))
    for name, label, count in SAMPLES[1:]:
        post = read(ROOT / "runs" / name / "constant" / "summary.json")["parameters"]
        comparison.append(row("", label, count, interval(post["b"])))

    for filename, content in (("literature_table.tex", comparison), ("benchmark_parameters.tex", parameters)):
        write_table(filename, content + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    main()
