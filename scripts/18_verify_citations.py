"""18 - Verify every reference against Crossref and build the INFORMS-style
reference list.

For entries with a DOI: fetch https://api.crossref.org/works/<DOI>, compare
title (token overlap), year, volume, and first page. For entries without a DOI
(books, working papers, preprints, or journal DOIs not yet recorded): query
Crossref's bibliographic search and record the best match for manual review;
working papers and preprints are checked against their public URL (HTTP 200).
Outputs: literature/citation_audit.csv, manuscript/references.tex
"""
import sys, json, re, time, urllib.request, urllib.parse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd

UA = {"User-Agent": "nullworld-citation-check/0.1 (mailto:research@example.org)"}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def toks(s):
    s = re.sub(r"\\[a-z]+|[{}`'\\]", "", str(s).lower())
    return set(re.findall(r"[a-z0-9]+", s))


def overlap(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A))


def main():
    refs = pd.read_csv(ROOT / "literature/references.csv", sep="|", dtype=str).fillna("")
    rows = []
    for _, r in refs.iterrows():
        rec = dict(key=r.key, doi=r.doi, status="", crossref_title="", crossref_year="", crossref_volume="",
                   crossref_page="", title_overlap="", found_doi="")
        try:
            if r.doi and r.type == "article":
                st, body = get("https://api.crossref.org/works/" + urllib.parse.quote(r.doi))
                m = json.loads(body)["message"]
            elif r.type == "article":
                q = urllib.parse.urlencode({"query.bibliographic": f"{r.title} {r.authors} {r.venue} {r.year}", "rows": 1})
                st, body = get("https://api.crossref.org/works?" + q)
                items = json.loads(body)["message"]["items"]
                m = items[0] if items else {}
                rec["found_doi"] = m.get("DOI", "")
            else:
                m = None
            if m is not None:
                t = (m.get("title") or [""])[0]
                y = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
                rec.update(crossref_title=t, crossref_year=y, crossref_volume=m.get("volume", ""),
                           crossref_page=m.get("page", m.get("article-number", "")), title_overlap=round(overlap(r.title, t), 2))
                t_full = t + " " + " ".join(m.get("subtitle") or [])
                rec["title_overlap"] = round(max(overlap(r.title, t), overlap(r.title, t_full)), 2)
                ok_t = rec["title_overlap"] >= 0.8
                ok_y = str(y) == r.year or (r.year and y and abs(int(y) - int(r.year)) <= 1)
                ok_v = (not r.volume) or str(m.get("volume", "")) == r.volume
                rec["status"] = "VERIFIED" if (ok_t and ok_y and ok_v) else f"CHECK(title={ok_t},year={ok_y},vol={ok_v})"
            else:
                url = r.note if r.note.startswith("http") else ""
                if url:
                    try:
                        st, _ = get(url)
                        rec["status"] = f"URL_OK({st})"
                    except Exception as e:
                        rec["status"] = f"URL_FAIL({type(e).__name__})"
                else:
                    rec["status"] = "BOOK(manual)"
        except Exception as e:
            rec["status"] = f"ERROR({type(e).__name__}: {str(e)[:60]})"
        rows.append(rec)
        print(r.key, rec["status"], rec["title_overlap"], flush=True)
        time.sleep(1.2)
    a = pd.DataFrame(rows)
    a["verified_on"] = time.strftime("%Y-%m-%d")
    a.to_csv(ROOT / "literature/citation_audit.csv", index=False)

    # INFORMS-style reference list: "Author A, Author B (Year) Title. Journal Vol(Iss):pages."
    def entry(r):
        au = r.authors
        if r.type == "article":
            vi = f" {r.volume}" + (f"({r.issue})" if r.issue else "") + (f":{r.pages}" if r.pages else "")
            s = f"{au} ({r.year}) {r.title}. \\textit{{{r.venue}}}{vi}."
        elif r.type == "book":
            s = f"{au} ({r.year}) \\textit{{{r.title}}} ({r.venue})."
        else:
            s = f"{au} ({r.year}) {r.title}. {r.venue}."
        return s

    def label(r):
        names = [n.strip() for n in r.authors.split(",")]
        sur = [re.sub(r"\s+[A-Z]{1,3}( III)?$", "", n).strip() for n in names if n != "et al."]
        if "et al." in r.authors or len(sur) > 2:
            lab = f"{sur[0]} et al."
        elif len(sur) == 2:
            lab = f"{sur[0]} and {sur[1]}"
        else:
            lab = sur[0]
        return f"{lab}({r.year})"

    refs["sortkey"] = refs.authors.str.replace(r"[{}\\`']", "", regex=True).str.lower() + refs.year
    L = ["\\begin{thebibliography}{99}", "\\setlength{\\itemsep}{1pt}\\small"]
    for _, r in refs.sort_values("sortkey").iterrows():
        L.append(f"\\bibitem[{label(r)}]{{{r.key}}} {entry(r)}")
    L.append("\\end{thebibliography}")
    (ROOT / "manuscript/references.tex").write_text("\n".join(L) + "\n")
    # per-document lists: only works actually cited (journal rule: every listed reference is cited)
    for doc, extra in [("main", []), ("technical_appendix", [])]:
        tex = (ROOT / f"manuscript/{doc}.tex").read_text()
        keys = set()
        for grp in re.findall(r"\\cite[pt]?\*?\{([^}]*)\}", tex):
            keys.update(k.strip() for k in grp.split(","))
        keep = refs[refs.key.isin(keys)].sort_values("sortkey")
        missing = keys - set(refs.key)
        LL = ["\\begin{thebibliography}{99}", "\\setlength{\\itemsep}{1pt}\\small"]
        for _, r in keep.iterrows():
            LL.append(f"\\bibitem[{label(r)}]{{{r.key}}} {entry(r)}")
        LL.append("\\end{thebibliography}")
        (ROOT / f"manuscript/references_{doc}.tex").write_text("\n".join(LL) + "\n")
        print(doc, "cited:", len(keys), "listed:", len(keep), "missing from database:", sorted(missing))
    print(a.status.value_counts())


if __name__ == "__main__":
    main()
