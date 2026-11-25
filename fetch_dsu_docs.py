import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

TARGET_PAGES = [
    ("dsu_home.md", "Dayananda Sagar University (DSU) - Home", "https://www.dsu.edu.in/"),
    ("dsu_admissions_portal.md", "DSU Admissions Portal", "https://admissions.dsu.edu.in/"),
    ("dsu_helpline.md", "DSU Admissions Helpline", "https://www.dsu.edu.in/admission/helpline"),
    ("dsu_postal_address.md", "DSU Campuses & Postal Addresses", "https://www.dsu.edu.in/contact-us/postal-address"),
    ("dsu_academics.md", "DSU Academics & Schools Overview", "https://www.dsu.edu.in/academics"),
    ("dsu_engineering.md", "DSU School of Engineering", "https://www.dsu.edu.in/academics/schools/engineering"),
    ("dsu_commerce_management.md", "DSU School of Commerce & Management", "https://www.dsu.edu.in/academics/schools/commerce-management"),
    ("dsu_applied_sciences.md", "DSU School of Basic & Applied Sciences", "https://www.dsu.edu.in/academics/schools/basic-applied-sciences"),
    ("dsu_health_sciences.md", "DSU School of Health Sciences", "https://www.dsu.edu.in/academics/schools/health-sciences"),
    ("dsu_law.md", "DSU School of Law", "https://www.dsu.edu.in/academics/schools/law"),
    ("dsu_arts_design.md", "DSU School of Arts, Design & Humanities", "https://www.dsu.edu.in/academics/schools/arts-design-humanities"),
    ("dsu_placements.md", "DSU Placements Contact & Career Services", "https://www.dsu.edu.in/placements/placements-contact-us"),
    ("dsu_placements_overview.md", "DSU Placements Overview", "https://www.dsu.edu.in/placements"),
    ("dsu_accommodation.md", "DSU Hostels & Accommodation", "https://www.dsu.edu.in/accommodation"),
    ("dsu_library.md", "DSU Library & Learning Resources", "https://www.dsu.edu.in/library"),
    ("dsu_best_practices_naac.md", "DSU Best Practices, Innovation & AIC-DSU", "https://www.dsu.edu.in/naac/criterion-7/best-practices-naac"),
]

SEED_PDF_URLS = [
    "https://www.dsu.edu.in/images/Admission/brochures/2025/DSU_Main_Campus_Programs_2025.pdf",
]


def html_to_markdown(html: str, title: str, url: str) -> tuple[str, list[str]]:
    """Convert HTML into clean Markdown and extract any linked PDF URLs."""
    soup = BeautifulSoup(html, "html.parser")

    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if ".pdf" in href.lower():
            full_url = urljoin(url, href)
            pdf_links.append(full_url)

    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    main = soup.find("main") or soup.find(id=re.compile(r"content|main", re.I)) or soup.body or soup

    lines = [f"# {title}", f"Source URL: {url}", ""]
    seen_lines = set()

    for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if elem.name == "tr":
            cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in elem.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            line = "| " + " | ".join(cells) + " |"
        else:
            text = re.sub(r"\s+", " ", elem.get_text(" ", strip=True))
            if not text or len(text) < 3:
                continue
            if elem.name == "h1":
                line = f"# {text}"
            elif elem.name == "h2":
                line = f"## {text}"
            elif elem.name == "h3":
                line = f"### {text}"
            elif elem.name == "h4":
                line = f"#### {text}"
            elif elem.name == "li":
                line = f"- {text}"
            else:
                line = text

        if line not in seen_lines:
            seen_lines.add(line)
            lines.append(line)

    content = "\n".join(lines).strip() + "\n"
    return content, pdf_links


def download_pdf(url: str, dest_dir: str) -> str | None:
    """Download a PDF if reachable and valid."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25, stream=True)
        if resp.status_code != 200:
            print(f"  [PDF SKIP] {url} -> HTTP {resp.status_code}")
            return None
        content = resp.content
        if not content.startswith(b"%PDF") or len(content) < 1024:
            print(f"  [PDF SKIP] {url} -> not a valid PDF ({len(content)} bytes)")
            return None

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "dsu_document.pdf"
        filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        dest_path = os.path.join(dest_dir, filename)
        with open(dest_path, "wb") as f:
            f.write(content)
        print(f"  [PDF OK] {filename} ({len(content) // 1024} KB)")
        return dest_path
    except Exception as e:
        print(f"  [PDF ERR] {url} -> {e}")
        return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    discovered_pdfs = list(SEED_PDF_URLS)
    saved_pages = 0

    print("Fetching official DSU web pages...")
    for filename, title, url in TARGET_PAGES:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"  [PAGE SKIP] {url} -> HTTP {resp.status_code}")
                continue
            md_text, pdf_links = html_to_markdown(resp.text, title, url)
            discovered_pdfs.extend(pdf_links)

            if len(md_text.splitlines()) < 6:
                print(f"  [PAGE SKIP] {url} -> insufficient text")
                continue

            out_path = os.path.join(DATA_DIR, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            saved_pages += 1
            print(f"  [PAGE OK] {filename} ({len(md_text)} chars)")
        except Exception as e:
            print(f"  [PAGE ERR] {url} -> {e}")

    # Deduplicate and download up to 8 relevant DSU PDFs
    unique_pdfs = []
    seen = set()
    for p in discovered_pdfs:
        if p not in seen and "dsu.edu.in" in p.lower():
            seen.add(p)
            unique_pdfs.append(p)

    print(f"\nDownloading up to 8 discovered DSU PDFs (found {len(unique_pdfs)} candidates)...")
    saved_pdfs = 0
    for pdf_url in unique_pdfs[:12]:
        if saved_pdfs >= 8:
            break
        if download_pdf(pdf_url, DATA_DIR):
            saved_pdfs += 1

    print(f"\nDone! Saved {saved_pages} Markdown pages and {saved_pdfs} PDFs in {DATA_DIR}")


if __name__ == "__main__":
    main()
