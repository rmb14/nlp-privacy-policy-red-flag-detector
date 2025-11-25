import csv
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# just a script to scrape website privacy policies based 
# on a list of domains cus its impossible to do it manually.

# for our model, we went through ~1750 company domains
# and got 4.65k concerning sentences this way which we 
# manually checked for false positives and malformed entries and 
# reduced the dataset to 2.5k high-quality samples.

# CONFIGGGG

INPUT_CSV = "companies_sorted.csv"   # Kaggle dataset https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset/data 
OUTPUT_CSV = "output_concerning_sentences.csv"
PROGRESS_LOG = "processed_indices.log"

MAX_WORKERS = 30
DELAY_BETWEEN_DOMAINS = 0  # small delay to be a bit nicer and not get blocked

# Tried using a generic scraper user-agent but most sites
# just blocked our requests so using an android phone user agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36"
}

# Sentence length constraints
MAX_SENTENCE_CHARS = 320   # hard cutoff for output sentences
MIN_SENTENCE_CHARS = 20    # ignore very short junk

# Below list is thanks to AI and it worked better than expected.
# Privacy policy + Terms of Service common paths
COMMON_POLICY_PATHS = [
    # Privacy-ish
    "/privacy",
    "/privacy/",
    "/privacy-policy",
    "/privacy-policy/",
    "/legal/privacy-policy",
    "/legal/privacy",
    "/privacy_and_cookies",
    "/privacy-and-cookies",
    "/data-protection",
    "/data-protection-policy",
    "/en/privacy",
    "/en/privacy-policy",
    "/policies/privacy",
    "/policy/privacy",

    # Terms-ish
    "/terms",
    "/terms/",
    "/terms-of-service",
    "/terms-of-service/",
    "/terms-of-use",
    "/terms-of-use/",
    "/termsandconditions",
    "/terms-and-conditions",
    "/legal/terms",
    "/legal/terms-of-service",
    "/tos",
    "/eula",
    "/user-agreement",
    "/policies/terms",
    "/policy/terms",
]

# ============== CONCERNING PATTERNS [DONE WITH HELP OF CHATGPT AGAIN] ==============

CONCERNING_PATTERNS = {
    "Data sale or sharing for advertising": [
        # clear “sell personal data” style
        r"\bsell(?:ing)?\b.*\b(personal (information|data)|your (information|data))\b",
        r"\b(personal (information|data)|your (information|data))\b.*\bmay be (sold|sold to)\b",
        # sharing + explicit third-party / ads / marketing
        r"\bshare\b.*\b(third[- ]part(y|ies)|advertis(ers|ing)|marketing partners?)\b",
        r"\bmay (sell|share|disclose)\b.*\bfor (advertising|marketing)\b",
    ],
    "Extensive tracking / profiling": [
        # cross-site / cross-service tracking / profiling
        r"\btrack(ing)?\b.*\b(across|between|across multiple|across different)\b.*\b(sites?|services?)\b",
        r"\bprofil(e|ing)\b.*\b(user(s)?|you|customers?|profiles?)\b",
        r"\bcombine\b.*\b(browsing|usage|activity)\b.*\bdata\b",
    ],
    "Data brokers / external sources": [
        r"\bdata brokers?\b",
        r"\bfrom\b.*\b(third[- ]party|external)\b.*\b(data|information)\b",
        r"\bpublicly available sources\b.*\bdata\b",
    ],
    "Vague or long data retention": [
        r"\bretain\b.*\b(as long as|for as long as)\b.*\b(purpose|purposes|necessary)\b",
        r"\bretain\b.*\bindefinite(ly)?\b",
        r"\bretain\b.*\bfor our legitimate business purposes\b",
    ],
    "Government / law-enforcement sharing (broad)": [
        r"\bshare\b.*\b(law enforcement|government authorities?)\b",
        r"\bdisclose\b.*\b(regulators?|authorities)\b",
        r"\bcomply\b.*\b(subpoena|court order|legal obligation|investigation)\b",
    ],
    "Policy changes without strong notice": [
        r"\bwe may (change|update|modify|amend)\b.*\bat any time\b",
        r"\breserve the right\b.*\bchange\b.*\bat any time\b",
        r"\bmay be updated from time to time\b.*\bwithout (prior )?notice\b",
    ],
    "Cross-border data transfer (potentially weak protections)": [
        r"\btransfer\b.*\b(personal (information|data)|your data)\b.*\bto\b.*\b(other|third|foreign)\b.*\bcountries\b",
        r"\b(personal (information|data)|your data)\b.*\bmay be transferred\b.*\boutside\b.*\b(EEA|EU|your country|your jurisdiction)\b",
        r"\btransfer\b.*\bdata\b.*\boutside\b.*\b(EEA|EU|your country|your jurisdiction)\b",
    ],
    "General data sharing / selling language": [
        # keep only sharing with 3rd parties/affiliates, not every “we share info internally”
        r"\bmay (share|disclose)\b.*\bwith\b.*\b(third[- ]part(y|ies)|partners?|affiliates?)\b",
        r"\b(information|data)\b.*\bmay be (shared|disclosed)\b.*\bwith\b.*\b(third[- ]part(y|ies)|partners?|affiliates?)\b",
    ],
    "Third-party analytics / ad partners": [
        r"\buse\b.*\bthird[- ]party\b.*\banalytics\b",
        r"\bshare\b.*\b(information|data)\b.*\bwith\b.*\b(advertising (partners?|networks?)|analytics providers?)\b",
        r"\binterest\-based advertising\b",
        r"\btargeted advertising\b",
    ],
}

# label priority: lower number = “more specific/interesting”
LABEL_PRIORITY = {
    "Data sale or sharing for advertising": 1,
    "Third-party analytics / ad partners": 2,
    "Data brokers / external sources": 3,
    "General data sharing / selling language": 4,
    "Extensive tracking / profiling": 5,
    "Cross-border data transfer (potentially weak protections)": 6,
    "Vague or long data retention": 7,
    "Government / law-enforcement sharing (broad)": 8,
    "Policy changes without strong notice": 9,
}


file_lock = threading.Lock()

def normalize_domain(raw: str) -> str:
    # basically just clean link from URL format to bare domain
    d = (raw or "").strip()
    d = d.replace("http://", "").replace("https://", "")
    d = d.split("/")[0]
    return d


def is_missing_value(val: str) -> bool:
    if val is None:
        return True
    s = val.strip().lower()
    return s == "" or s == "null" or s == "none"


def get_html(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 200 and "text/html" in (resp.headers.get("Content-Type") or ""):
            return resp.text
    except requests.RequestException:
        return None
    return None

# this function was made by AI
def get_candidate_policy_urls(domain: str):
    """
    Returns a list of candidate policy / terms URLs for a given domain.
    Combines:
      - common known paths
      - links on homepage containing keywords
    """
    domain = normalize_domain(domain)
    base_url = "https://" + domain
    candidates = set()

    # 1) Common paths
    for path in COMMON_POLICY_PATHS:
        candidates.add(urljoin(base_url, path))

    # 2) Parse homepage and find links with keywords
    home_html = get_html(base_url)
    if home_html:
        soup = BeautifulSoup(home_html, "html.parser")
        for a in soup.find_all("a", href=True):
            text = (a.get_text() or "").strip().lower()
            href = a["href"]
            href_lower = href.lower()

            # keywords for privacy + terms
            if any(
                kw in text or kw in href_lower
                for kw in [
                    "privacy",
                    "data protection",
                    "data policy",
                    "privacy policy",
                    "terms",
                    "conditions",
                    "terms of use",
                    "terms of service",
                    "user agreement",
                    "legal",
                    "policy",
                ]
            ):
                full_url = urljoin(base_url, href)
                candidates.add(full_url)

    return list(candidates)

# Again, AI help here too.
def extract_text_from_url(url: str) -> str:
    html = get_html(url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    chunks = []
    for el in soup.find_all(["p", "li", "div", "span", "section", "article"]):
        text = el.get_text(" ", strip=True)
        if text:
            chunks.append(text)

    text = " ".join(chunks)
    # normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str):
    if not text:
        return []

    text = re.sub(r"\s+", " ", text)

    initial_sents = sent_tokenize(text)
    sentences = []

    for s in initial_sents:
        s = s.strip()
        if not s:
            continue

        # If extremely long, break into smaller chunks at word boundaries
        if len(s) > MAX_SENTENCE_CHARS * 2:
            words = s.split()
            chunk = []
            total_len = 0
            for w in words:
                if total_len + len(w) + 1 > MAX_SENTENCE_CHARS:
                    chunk_sentence = " ".join(chunk).strip()
                    if chunk_sentence:
                        sentences.append(chunk_sentence)
                    chunk = [w]
                    total_len = len(w)
                else:
                    chunk.append(w)
                    total_len += len(w) + 1
            final = " ".join(chunk).strip()
            if final:
                sentences.append(final)
        else:
            sentences.append(s)

    # Filter and dedupe
    clean = []
    seen = set()

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # enforce character limits
        if len(s) < MIN_SENTENCE_CHARS:
            continue
        if len(s) > MAX_SENTENCE_CHARS:
            continue

        # must have letters
        if not re.search(r"[a-zA-Z]", s):
            continue

        # avoid lines with extremely low word diversity (repeated spam)
        words = s.split()
        if len(set(words)) < 3:
            continue

        if s not in seen:
            seen.add(s)
            clean.append(s)

    return clean


def find_concerning_in_text(text: str):
    sentences = split_sentences(text)
    results = []
    seen = set()  # (sentence, concern_type)

    for sent in sentences:
        s_lower = sent.lower()
        matched_labels = set()

        for label, patterns in CONCERNING_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, s_lower):
                    matched_labels.add(label)
                    break

        if not matched_labels:
            continue

        # Pick the highest-priority label (lowest number)
        best_label = sorted(
            matched_labels,
            key=lambda lab: LABEL_PRIORITY.get(lab, 999)
        )[0]

        key = (sent, best_label)
        if key in seen:
            continue
        seen.add(key)

        results.append((sent, best_label))

    return results



def load_input_entries(input_csv: str):
    entries = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")

        domain_idx = None
        name_idx = None  # 'name' column (Name of Company)

        for i, fn in enumerate(reader.fieldnames):
            fl = fn.lower().strip()
            if fl == "domain":
                domain_idx = i
            if fl == "name":
                name_idx = i

        for idx, row in enumerate(reader):
            row_vals = list(row.values())

            domain_val = row_vals[domain_idx] if domain_idx < len(row_vals) else ""
            if is_missing_value(domain_val):
                # skip rows with missing domain
                continue

            name_val = ""
            if name_idx is not None and name_idx < len(row_vals):
                name_val = row_vals[name_idx]

            if is_missing_value(name_val):
                # skip rows with missing name/company
                continue

            entries.append(
                {
                    "index": idx,  # use CSV row index (0-based for data rows)
                    "company": (name_val or "").strip(),
                    "domain": (domain_val or "").strip(),
                }
            )

    return entries


def load_processed_indices(progress_log: str):
    processed = set()
    if os.path.exists(progress_log):
        with open(progress_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed.add(int(line))
                except ValueError:
                    continue
    return processed


def append_processed_index(progress_log: str, idx: int):
    with file_lock:
        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(str(idx) + "\n")


def ensure_output_header(output_csv: str):
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["company", "domain", "sentence", "concern_type"])


def append_results_to_output(output_csv: str, rows):
    """
    rows: list of (company, domain, sentence, concern_type)
    """
    if not rows:
        return
    with file_lock:
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)


# ============== WORKER ==============

def process_entry(entry):
    idx = entry["index"]
    company = entry["company"]
    domain = entry["domain"]

    rows = []

    try:
        candidates = get_candidate_policy_urls(domain)
        # cap number of pages per domain to avoid hammering & duplicates
        max_pages = 4
        visited = set()
        page_count = 0

        # per-domain de-duplication (sentence, concern_type)
        domain_seen = set()

        for url in candidates:
            if page_count >= max_pages:
                break
            if url in visited:
                continue
            visited.add(url)

            text = extract_text_from_url(url)
            if not text:
                continue

            matches = find_concerning_in_text(text)
            for sentence, concern_type in matches:
                key = (sentence, concern_type)
                if key in domain_seen:
                    continue
                domain_seen.add(key)
                rows.append((company, domain, sentence, concern_type))

            page_count += 1
            # tiny delay per page
            time.sleep(0.05)

    except Exception:
        pass

    return idx, rows


# ============== MAIN ==============
# AI helped with the threading.
def main():
    print("Loading input entries...")
    entries = load_input_entries(INPUT_CSV)
    print(f"Loaded {len(entries)} entries with non-empty name+domain from {INPUT_CSV}")

    processed_indices = load_processed_indices(PROGRESS_LOG)
    print(f"Loaded {len(processed_indices)} already processed indices from {PROGRESS_LOG}")

    pending_entries = [e for e in entries if e["index"] not in processed_indices]
    print(f"{len(pending_entries)} entries left to process")

    if not pending_entries:
        print("Everything already processed. Nothing to do.")
        return

    ensure_output_header(OUTPUT_CSV)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {}
        for entry in pending_entries:
            future = executor.submit(process_entry, entry)
            future_to_idx[future] = entry["index"]

        total = len(pending_entries)
        done_count = 0

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                entry_idx, rows = future.result()
            except Exception as e:
                entry_idx = idx
                rows = []
                print(f"Error on index {idx}: {e}")

            # Save results
            append_results_to_output(OUTPUT_CSV, rows)
            append_processed_index(PROGRESS_LOG, entry_idx)

            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"Processed {done_count}/{total} entries...")

            # small delay between domains
            time.sleep(DELAY_BETWEEN_DOMAINS)

    print("All done.")


if __name__ == "__main__":
    main()
