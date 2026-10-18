from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Sequence

from bs4 import BeautifulSoup
from pydantic import ValidationError

from config import settings
from src.news_rag.io import read_json, write_json
from src.news_rag.models import ArticleRecord, Funding, normalize_company_name
from src.news_rag.tracing import traceable
from src.news_rag.url_utils import looks_like_et_tech_article

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None


CATEGORY_DEFINITIONS = {
    "AI": (
        "Artificial intelligence, generative AI, LLMs, machine learning algorithms, "
        "neural networks, deep learning models, prompt engineering, AI chatbots, computer vision"
    ),
    "funding": (
        "Venture capital, startup investment, funding rounds, seed round, series A, series B, "
        "investors, raising capital, equity financing, valuation, angel investors"
    ),
    "product launch": (
        "New product launch, unveiling product, feature release, software debut, "
        "hardware release, rollout, announces new platform or tool, devices"
    ),
    "policy": (
        "Government regulations, legal compliance, tech policy, RBI directives, MeitY guidelines, "
        "antitrust law, data privacy laws, taxation, export restrictions, legislation"
    ),
    "M&A": (
        "Mergers and acquisitions, buyout, takeover, acquiring company, stake purchase, consolidation"
    ),
    "layoffs": (
        "Job cuts, workforce reduction, employee layoffs, severance, firing employees, downsizing"
    ),
    "earnings": (
        "Financial results, quarterly earnings, profit margin, revenue growth, net loss, fiscal report"
    ),
    "fintech": (
        "Financial technology, digital payments, UPI, banking tech, payment gateways, "
        "cryptocurrency, digital lending, wallets, credit"
    ),
    "semiconductors": (
        "Semiconductor chips, fabrication plants, foundries, microprocessors, silicon wafer, "
        "chip packaging, assembly and test, ATMP, ASICs"
    ),
    "cybersecurity": (
        "Cybersecurity, cyberattacks, data breach, malware, ransomware, hacking, "
        "security vulnerabilities, phishing, information security"
    ),
    "enterprise tech": (
        "Enterprise software, SaaS, cloud infrastructure, IT services, data centers, "
        "B2B software solutions, digital transformation"
    ),
}

# Mapping of publisher URL sections to standard categories (Option 2)
SECTION_TO_CATEGORIES = {
    "artificial-intelligence": "AI",
    "funding": "funding",
    "startups": "funding",
    "policy": "policy",
    "hardware": "semiconductors",
    "ites": "enterprise tech",
    "software": "enterprise tech",
    "information-tech": "enterprise tech",
    "catalysts": "fintech",
}

# Publisher keywords and title regex rules for Option 2
KEYWORD_DOMAIN_RULES = {
    "AI": (r"\b(ai|artificial intelligence|genai|generative ai|llms?|machine learning)\b",),
    "funding": (r"\b(funding|fundraise|fundraising|seed round|series [a-z]|venture capital|investors?|raised \$|raised rs)\b",),
    "product launch": (r"\b(launches|launched|launching|unveils|unveiled|introduces|rollout|rolls out)\b",),
    "policy": (r"\b(policy|regulation|regulatory|rbi|meity|antitrust|government mandate|guidelines)\b",),
    "M&A": (r"\b(acquisition|acquires|acquired|merger|stake purchase|buyout)\b",),
    "layoffs": (r"\b(layoff|layoffs|job cuts|fired|workforce reduction|downsizing)\b",),
    "earnings": (r"\b(earnings|quarterly profit|net profit|q[1-4] profit|q[1-4] revenue|revenue jump|fiscal results)\b",),
    "fintech": (r"\b(fintech|upi|digital payments?|payment gateway|crypto|cryptocurrency|digital lending|neobank)\b",),
    "semiconductors": (r"\b(semiconductor|semiconductors|chips?|foundry|foundries|wafer|atmp|fab)\b",),
    "cybersecurity": (r"\b(cybersecurity|cyberattack|hacker|hackers|malware|ransomware|data breach|phishing)\b",),
    "enterprise tech": (r"\b(enterprise tech|saas|cloud computing|it services|data centers?|b2b software)\b",),
}

# Out-of-taxonomy domain indicators when the article belongs to a domain not in CATEGORY_DEFINITIONS
OUT_OF_TAXONOMY_INDICATORS = {
    "quantum computing": (r"\b(quantum|qubit|qubits|quantum computer|quantum computing)\b",),
    "biotech": (r"\b(biotech|biotechnology|healthtech|genomics|crispr|bio-tech)\b",),
    "clean tech": (r"\b(clean tech|cleantech|ev|electric vehicles?|battery storage|green energy|solar tech)\b",),
    "space tech": (r"\b(space tech|spacetech|satellite|satellites|isro|spacecraft|aerospace)\b",),
    "gaming": (r"\b(gaming|esports|video games?|game studio|cloud gaming)\b",),
    "edtech": (r"\b(edtech|education tech|e-learning|learning platform)\b",),
    "agritech": (r"\b(agritech|agriculture tech|smart farming)\b",),
    "robotics": (r"\b(robotics|robots?|autonomous drones?|uav|humanoid)\b",),
    "telecom": (r"\b(telecom|telecommunications|5g|6g|spectrum|broadband|telco)\b",),
}

# Retained for backward compatibility
CATEGORY_RULES = {
    "funding": ("funding", "raised", "raises", "seed", "series a", "investor"),
    "product launch": ("launch", "launched", "unveils", "introduces"),
    "policy": ("policy", "regulation", "government", "rule", "rbi", "meity"),
    "M&A": ("acquire", "acquisition", "merger", "stake"),
    "layoffs": ("layoff", "job cuts", "fired", "workforce reduction"),
    "AI": (" ai ", "artificial intelligence", "genai", "generative ai"),
    "earnings": ("earnings", "profit", "loss", "revenue"),
}

LOCATION_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "gurugram": "Gurugram",
    "gurgaon": "Gurugram",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
    "chennai": "Chennai",
    "india": "India",
    "singapore": "Singapore",
}


def _text_or_empty(soup: BeautifulSoup, selector: str, attr: str | None = None) -> str:
    node = soup.select_one(selector)
    if not node:
        return ""
    if attr:
        return node.get(attr, "").strip()
    return node.get_text(" ", strip=True)


def _json_ld_items(payload: Any) -> list[dict[str, Any]]:
    """Return every object in a JSON-LD document, including graph members."""
    if isinstance(payload, list):
        return [item for value in payload for item in _json_ld_items(value)]
    if not isinstance(payload, dict):
        return []

    items = [payload]
    for key in ("@graph", "mainEntity", "mainEntityOfPage"):
        if key in payload:
            items.extend(_json_ld_items(payload[key]))
    return items


def parse_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "{}")
        except json.JSONDecodeError:
            continue
        items = _json_ld_items(payload)
        for item in items:
            article_types = item.get("@type", "")
            type_names = article_types if isinstance(article_types, list) else [article_types]
            if isinstance(item, dict) and any(
                str(name).lower() in {"newsarticle", "article"} for name in type_names
            ):
                return item
    return {}


def clean_body(html: str) -> str:
    extracted = None
    if trafilatura:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            output_format="txt",
        )
    if not extracted:
        soup = BeautifulSoup(html, "html.parser")
        for selector in (
            "script", "style", "nav", "footer", "aside", "iframe",
            ".breadcrumb", ".breadcrumbs", ".synopsis", ".now-playing",
            ".share-widget", ".advertisement", ".ad-container", ".newsletter-signup"
        ):
            for node in soup.select(selector):
                node.decompose()
        article = soup.find("article") or soup.find("main") or soup.body or soup
        extracted = article.get_text("\n\n", strip=True)

    # Technique 1: Preserve paragraph and sentence boundaries
    # Normalize spaces/tabs on each line while preserving newline breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in extracted.splitlines()]
    clean_text = "\n\n".join(line for line in lines if line)

    # Cut off trailing boilerplate (Synopsis repetition, catch all news, now playing, subscribe)
    boilerplate_cutoff = re.search(
        r"\n\n(?:Synopsis|\(Catch all the Technology News|Now Playing|Elevate your knowledge and leadership skills|Subscribe Now\b)",
        clean_text,
        flags=re.IGNORECASE,
    )
    if boilerplate_cutoff:
        clean_text = clean_text[:boilerplate_cutoff.start()].strip()

    return clean_text


def normalize_datetime(value: str | None) -> str:
    """Normalize a publisher-supplied timestamp, never substituting scrape time."""
    if not value:
        raise ValueError("Article has no publisher-supplied publication timestamp")
    value = value.strip()
    normalized = value.replace("Z", "+00:00")
    # Some Economic Times JSON-LD timestamps use a one-digit hour (for example T6:00:00).
    normalized = re.sub(r"T(\d):", r"T0\1:", normalized)
    # ET also emits midnight as hour 24, which is the next calendar day in ISO 8601.
    rolls_to_next_day = bool(re.search(r"T24:", normalized))
    normalized = normalized.replace("T24:", "T00:")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            for date_format in ("%A, %d %B, %Y, %I:%M:%S %p", "%d %B %Y"):
                try:
                    dt = datetime.strptime(value, date_format)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unparseable publisher publication timestamp: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if rolls_to_next_day:
        dt += timedelta(days=1)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sentence_summary(body: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", body)
    selected = [sentence.strip() for sentence in sentences if sentence.strip()][:3]
    return " ".join(selected) or body[:280]


_CAT_NAMES: list[str] = list(CATEGORY_DEFINITIONS.keys())
_CAT_VECTORS: Any = None


def get_category_vectors() -> tuple[list[str], Any]:
    global _CAT_VECTORS
    if _CAT_VECTORS is None:
        try:
            from src.news_rag.langchain_rag import get_embeddings
            import numpy as np

            emb = get_embeddings()
            _CAT_VECTORS = np.array(emb.embed_documents(list(CATEGORY_DEFINITIONS.values())))
        except Exception:
            _CAT_VECTORS = None
    return _CAT_NAMES, _CAT_VECTORS


def classify_article_categories(
    title: str,
    body: str,
    url: str = "",
    soup: BeautifulSoup | None = None,
) -> list[str]:
    """
    Hybrid classification combining:
    - Option 1: Semantic Embedding Cosine Similarity (Sentence-Transformers)
    - Option 2: Publisher URL Section & HTML <meta> Metadata
    Assigns multiple categories if article covers multiple domains.
    Detects and handles out-of-taxonomy domains not in CATEGORY_DEFINITIONS.
    """
    # 1. OPTION 2: Publisher Signals
    keywords = ""
    section = ""
    if soup:
        kw_node = soup.find("meta", attrs={"name": "keywords"}) or soup.find("meta", attrs={"name": "news_keywords"})
        if kw_node and kw_node.get("content"):
            keywords = str(kw_node["content"])
        sec_node = soup.find("meta", property="article:section") or soup.find("meta", attrs={"name": "section"})
        if sec_node and sec_node.get("content"):
            section = str(sec_node["content"])

    # URL section slug
    sec_match = re.search(r'economictimes\.indiatimes\.com/(?:tech|industry)/([^/]+)/', url)
    detected_sec = (sec_match.group(1) if sec_match else section).lower().strip()

    publisher_cats = set()
    if detected_sec in SECTION_TO_CATEGORIES:
        publisher_cats.add(SECTION_TO_CATEGORIES[detected_sec])

    domain_text = f"{title} {keywords} {body[:500]}".lower()
    for cat, patterns in KEYWORD_DOMAIN_RULES.items():
        if any(re.search(pat, domain_text) for pat in patterns):
            publisher_cats.add(cat)

    # 2. OPTION 1: Embedding Cosine Similarity
    sim_scores: dict[str, float] = {}
    cat_names, cat_vectors = get_category_vectors()
    if cat_vectors is not None:
        try:
            from src.news_rag.langchain_rag import get_embeddings
            import numpy as np

            emb = get_embeddings()
            doc_text = f"{title}. {body[:500]}"
            doc_vec = np.array(emb.embed_query(doc_text))
            sims = np.dot(cat_vectors, doc_vec) / (np.linalg.norm(cat_vectors, axis=1) * np.linalg.norm(doc_vec))
            sim_scores = {name: float(s) for name, s in zip(cat_names, sims)}
        except Exception:
            sim_scores = {}

    # 3. OUT-OF-TAXONOMY HANDLING:
    # When article belongs to a domain not present in CATEGORY_DEFINITIONS
    out_of_taxonomy_matches: list[str] = []
    for domain, patterns in OUT_OF_TAXONOMY_INDICATORS.items():
        if any(re.search(pat, domain_text) for pat in patterns):
            out_of_taxonomy_matches.append(domain)

    generic_sections = {"newsletters", "technology", "tech", "articleshow", "internet", "news"}
    if detected_sec and detected_sec not in generic_sections and detected_sec not in SECTION_TO_CATEGORIES:
        clean_sec = detected_sec.replace("-", " ")
        if clean_sec not in out_of_taxonomy_matches and len(clean_sec) > 3:
            out_of_taxonomy_matches.append(clean_sec)

    # 4. HYBRID FUSION:
    candidates: dict[str, float] = {}
    HIGH_THRESHOLD = 0.30
    MEDIUM_THRESHOLD = 0.25

    if sim_scores:
        for cat, s in sim_scores.items():
            score = float(s)
            if cat in publisher_cats:
                score += 0.15
            if score >= HIGH_THRESHOLD:
                candidates[cat] = score

        for cat in publisher_cats:
            if cat not in candidates:
                if sim_scores.get(cat, 0.0) >= MEDIUM_THRESHOLD:
                    candidates[cat] = float(sim_scores.get(cat, 0.0)) + 0.15
                elif any(re.search(pat, domain_text) for pat in KEYWORD_DOMAIN_RULES.get(cat, ())):
                    candidates[cat] = 0.40
    else:
        # Fallback if embeddings not available (e.g. lightweight environment)
        for cat in publisher_cats:
            candidates[cat] = 0.50

    for domain in out_of_taxonomy_matches:
        max_predefined = max(sim_scores.values()) if sim_scores else 0.0
        if max_predefined < 0.40 or len(candidates) == 0:
            candidates[domain] = 0.50
        else:
            candidates[domain] = 0.38

    if not candidates:
        return ["general"]

    sorted_cats = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_cats[0][1]
    selected: list[str] = [sorted_cats[0][0]]

    # Assign multiple categories if article covers multiple domains
    for cat, score in sorted_cats[1:]:
        if score >= 0.30 and score >= (top_score * 0.65):
            if cat not in selected:
                selected.append(cat)

    return selected


def category_for(text: str) -> str:
    """Backward compatibility wrapper returning the primary category."""
    cats = classify_article_categories(title=text[:200], body=text)
    return cats[0] if cats else "general"


_GLINER_MODEL: Any = None


def get_gliner_model() -> Any:
    """Lazily load or initialize the GLiNER model for zero-shot Named Entity Recognition."""
    global _GLINER_MODEL
    if _GLINER_MODEL is not None:
        return _GLINER_MODEL
    try:
        from gliner import GLiNER

        model_name = getattr(settings, "gliner_model", "urchade/gliner_small-v2.1")
        _GLINER_MODEL = GLiNER.from_pretrained(model_name)
    except Exception as exc:
        _GLINER_MODEL = None
        raise RuntimeError("GLiNER not working") from exc
    return _GLINER_MODEL


RELATIONSHIP_MODIFIERS = re.compile(
    r"\s*-(?:backed|funded|led|owned|driven|focused|centric|based)\b",
    re.IGNORECASE,
)

NOISE_SUFFIXES = re.compile(
    r"\s+(?:Synopsis|Now Playing|Latest News|News News|News Updates|Updates|IPO|Fundraise|Funding Round|Funding)\b",
    re.IGNORECASE,
)

NON_COMPANY_TERMS = {
    "ai", "genai", "ipo", "fintech", "cxo", "deeptech startups", "technology news",
    "news news", "news", "updates", "latest news updates", "the economic times",
    "economic times", "synopsis", "the end", "subscribe now", "the technology news news",
    "et spotlight", "spotlight", "etspotlight", "enterprise", "enterprises",
}


def clean_company_name(name: str) -> str | None:
    """Remove common article noise, legal suffixes, and non-entity candidates."""
    cleaned = RELATIONSHIP_MODIFIERS.sub("", name).strip()
    cleaned = NOISE_SUFFIXES.sub("", cleaned).strip()
    normalized = normalize_company_name(cleaned)
    if not normalized or normalized.casefold() in NON_COMPANY_TERMS or "spotlight" in normalized.casefold():
        return None
    return normalized


def pack_paragraphs_greedy(text: str, max_chars: int = 1000) -> list[str]:
    """Greedily accumulate paragraphs together until adding the next paragraph would exceed max_chars."""
    raw_paras = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs: list[str] = []
    for p in raw_paras:
        while len(p) > max_chars:
            split_idx = p.rfind(". ", 0, max_chars)
            if split_idx == -1:
                split_idx = max_chars
            else:
                split_idx += 1
            paragraphs.append(p[:split_idx].strip())
            p = p[split_idx:].strip()
        if p:
            paragraphs.append(p)

    chunks: list[str] = []
    current_chunk = ""
    for para in paragraphs:
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + 1 + len(para) <= max_chars:
            current_chunk = f"{current_chunk}\n{para}"
        else:
            chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def extract_named_items(
    text: str,
    threshold: float = 0.20,
    model: Any = None,
) -> tuple[list[str], list[str], list[str]]:
    if model is None:
        model = get_gliner_model()
    if model is None:
        raise RuntimeError("GLiNER not working")

    locations = []
    lowered = text.lower()
    for needle, normalized in LOCATION_ALIASES.items():
        if re.search(rf"\b{re.escape(needle)}\b", lowered) and normalized not in locations:
            locations.append(normalized)

    stop_titles = {
        "mr.", "mr", "ms.", "ms", "mrs.", "mrs", "dr.", "dr",
        "prof.", "prof", "ceo", "cfo", "cto", "md", "chairman", "president",
        "co-founder", "founder", "director", "minister"
    }
    stop_phrases = {
        "the end", "subscribe now", "latest news updates", "the economic times",
        "economic times", "ettech", "et tech", "synopsis", "synopsis india",
        "read more", "click here", "read full"
    }
    stop_noise_substrings = (
        "economic times", "subscribe now", "latest news", "the end", "synopsis"
    )

    stop_org_tokens = {
        "labs", "lab", "group", "sons", "technologies", "technology", "ventures",
        "capital", "partners", "inc", "ltd", "limited", "corp", "corporation",
        "fund", "holdings", "systems", "solutions", "network", "networks", "media",
        "bank", "council", "embassy", "bureau", "association", "institute",
        "ministry", "department", "foundation", "models", "model"
    }

    # Strip hashtags (e.g. #TechLeadership #ADA) so hashtags are not misidentified as entities
    cleaned_text = re.sub(r"#\w+", "", text[:25000])

    # Greedy Packing: accumulate paragraphs together up to 1,000 characters per chunk
    chunks = pack_paragraphs_greedy(cleaned_text, max_chars=1000)

    companies: list[str] = []
    people: list[str] = []
    labels = ["company", "person", "location"]

    if chunks:
        try:
            batch_ents = model.batch_predict_entities(chunks, labels, threshold=threshold)
        except Exception as exc:
            raise RuntimeError("GLiNER not working") from exc
        ents = [ent for chunk_ents in batch_ents for ent in chunk_ents]
    else:
        ents = []

    for ent in ents:
            cleaned = re.sub(r"\s+", " ", ent.get("text", "").strip())
            label = ent.get("label", "").lower()
            clow = cleaned.lower()

            if any(noise in clow for noise in stop_noise_substrings) or len(cleaned) < 2:
                continue

            if label == "person":
                tokens = cleaned.split()
                while tokens and tokens[0].lower() in stop_titles:
                    tokens = tokens[1:]
                cleaned_person = " ".join(tokens).strip(" ,.-")
                person_tokens = cleaned_person.lower().split()

                stop_person_tokens = {
                    "ai", "bot", "bots", "agent", "agents", "model", "models",
                    "system", "systems", "team", "author", "spotlight", "user",
                    "users", "enterprise", "enterprises", "interview", "podcast",
                }
                if any(t in stop_person_tokens for t in person_tokens):
                    continue

                if any(t in stop_org_tokens for t in person_tokens):
                    cleaned_org = clean_company_name(cleaned_person)
                    if cleaned_org and cleaned_org not in companies and cleaned_org not in locations:
                        companies.append(cleaned_org)
                    continue

                if (
                    len(person_tokens) >= 2
                    and "." not in cleaned_person
                    and cleaned_person not in people
                    and cleaned_person not in locations
                    and cleaned_person.lower() not in stop_phrases
                ):
                    people.append(cleaned_person)

            elif label in ("company", "organization"):
                cleaned_org = clean_company_name(cleaned)
                if not cleaned_org:
                    continue
                clow_org = cleaned_org.lower()
                if (
                    cleaned_org not in companies
                    and cleaned_org not in locations
                    and clow_org not in stop_phrases
                    and not any(cleaned_org == p or (len(cleaned_org.split()) == 1 and cleaned_org in p.split()) for p in people)
                ):
                    companies.append(cleaned_org)

            elif label == "location":
                if clow in LOCATION_ALIASES:
                    norm = LOCATION_ALIASES[clow]
                    if norm not in locations:
                        locations.append(norm)

    return companies[:15], people[:15], locations


FUNDING_KEYWORD_PATTERN = re.compile(
    r"\b("
    r"fund(?:ing|ed|s|raise|raising|raiser|raised)?|"
    r"rais(?:e|ed|es|ing)|"
    r"invest(?:ment|ments|or|ors|ed|ing|s)?|"
    r"series\s+[a-z]|"
    r"pre-seed|seed|"
    r"venture(?:\s+capital)?|"
    r"angel(?:\s+investor|\s+investment)?|"
    r"valuation|"
    r"infus(?:ion|e|ed)|"
    r"capital-raise|"
    r"equity\s+round"
    r")\b|"
    r"(?:\$|Rs\.?|INR|USD|EUR)\s*[\d,.]+|"
    r"\b[\d,.]+\s*(?:crore|crores|million|billion|mn|bn)\b",
    re.IGNORECASE,
)

GLINER_FUNDING_LABELS = [
    "startup receiving funding",
    "venture capital or angel investor",
    "funding round type",
    "money amount raised",
]


def isolate_funding_paragraphs(text: str) -> list[str]:
    """Stage 1: Use a regex to isolate only the paragraphs within the webpage that mention funding keywords."""
    if not text or not text.strip():
        return []

    # If raw HTML with paragraph/article tags is provided
    if "<p" in text.lower() or "<article" in text.lower():
        soup = BeautifulSoup(text, "html.parser")
        p_tags = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        raw_paras = [p for p in p_tags if p and len(p.split()) >= 3]
        if not raw_paras:
            raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    else:
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if len(raw_paras) <= 1:
            raw_paras = [p.strip() for p in text.splitlines() if p.strip()]

    return [p for p in raw_paras if FUNDING_KEYWORD_PATTERN.search(p)]


def extract_funding(
    text: str,
    model: Any = None,
    threshold: float = 0.3,
) -> Funding | None:
    """Extract structured funding details using a two-stage approach:
    Stage 1: Use a regex to isolate only the paragraphs within the webpage that mention funding keywords.
    Stage 2 (Model Inference): Pass only the relevant 1-3 paragraphs into GLiNER with semantic labels:
        - 'startup receiving funding'
        - 'venture capital or angel investor'
        - 'funding round type'
        - 'money amount raised'
    """
    # Stage 1: Isolate relevant paragraphs using funding regex
    relevant_paras = isolate_funding_paragraphs(text)[:3]
    if not relevant_paras:
        return None

    # Stage 2: Model Inference with GLiNER on the relevant 1-3 paragraphs
    if model is None:
        model = get_gliner_model()
    if model is None:
        raise RuntimeError("GLiNER not working")

    chunks = relevant_paras
    try:
        batch_ents = model.batch_predict_entities(chunks, GLINER_FUNDING_LABELS, threshold=threshold)
    except Exception as exc:
        raise RuntimeError("GLiNER not working") from exc

    ents: list[dict[str, Any]] = [ent for chunk_ents in batch_ents for ent in chunk_ents]

    startup: str | None = None
    investors: list[str] = []
    round_type: str | None = None
    amount: str | None = None
    currency: str | None = None

    stop_investor_words = {
        "investor", "investors", "investment", "investments", "venture capital",
        "venture", "angel", "angels", "vc", "vcs", "fund", "funds", "firm",
        "companies", "company", "board", "regulatory filing",
    }

    for ent in ents:
        label = ent.get("label", "").lower()
        raw_val = ent.get("text", "").strip()
        if not raw_val:
            continue

        if label == "startup receiving funding":
            cleaned = clean_company_name(raw_val)
            if cleaned and cleaned.lower() not in NON_COMPANY_TERMS:
                if not startup:
                    startup = cleaned

        elif label == "venture capital or angel investor":
            cleaned = re.sub(
                r"^(?:new\s+investors?|investors?|existing\s+investors?)\s+",
                "",
                raw_val,
                flags=re.I,
            ).strip(" ,.-")
            cleaned_org = clean_company_name(cleaned) or cleaned
            clow = cleaned_org.lower()
            if (
                cleaned_org
                and len(cleaned_org) >= 2
                and clow not in NON_COMPANY_TERMS
                and clow not in stop_investor_words
                and (not startup or clow != startup.lower())
                and cleaned_org not in investors
            ):
                investors.append(cleaned_org)

        elif label == "funding round type":
            if not round_type:
                series_m = re.search(r"series\s+([a-z])", raw_val, re.I)
                if series_m:
                    round_type = f"Series {series_m.group(1).upper()}"
                elif re.search(r"pre-seed", raw_val, re.I):
                    round_type = "Pre-Seed"
                elif re.search(r"seed", raw_val, re.I):
                    round_type = "Seed"
                elif re.search(r"debt", raw_val, re.I):
                    round_type = "Debt"
                elif re.search(r"growth", raw_val, re.I):
                    round_type = "Growth"
                else:
                    norm = re.sub(r"\b(?:round|funding)\b", "", raw_val, flags=re.I).strip().title()
                    if norm:
                        round_type = norm

        elif label == "money amount raised":
            if not amount:
                m = re.search(
                    r"(?P<currency>\$|Rs\.?|INR|USD|EUR|£|€)?\s*(?P<amount>[\d,.]+)\s*(?P<unit>crore|crores|million|billion|mn|bn|m|b)?",
                    raw_val,
                    re.IGNORECASE,
                )
                if m and m.group("amount"):
                    parts = [m.group("amount")]
                    unit = m.group("unit")
                    if unit:
                        u_low = unit.lower()
                        if u_low in ("crore", "crores"):
                            parts.append("crore")
                        elif u_low in ("million", "mn", "m"):
                            parts.append("million")
                        elif u_low in ("billion", "bn", "b"):
                            parts.append("billion")
                        else:
                            parts.append(unit)
                    amount = " ".join(parts)
                    if m.group("currency"):
                        currency = m.group("currency")

    # Fallback checks on the 1-3 isolated paragraphs for amount / round if GLiNER entity span was partial
    combined_paras = " ".join(relevant_paras)
    if not amount or not currency:
        amount_match = re.search(
            r"(?P<currency>\$|Rs\.?|INR|USD|EUR|£|€)\s?(?P<amount>[\d,.]+)\s?(?P<unit>crore|crores|million|billion|mn|bn|m|b)?",
            combined_paras,
            flags=re.IGNORECASE,
        )
        if amount_match:
            if not amount:
                amt = amount_match.group("amount")
                unit = amount_match.group("unit")
                if unit:
                    u_low = unit.lower()
                    if u_low in ("crore", "crores"):
                        unit_str = "crore"
                    elif u_low in ("million", "mn", "m"):
                        unit_str = "million"
                    elif u_low in ("billion", "bn", "b"):
                        unit_str = "billion"
                    else:
                        unit_str = unit
                    amount = f"{amt} {unit_str}"
                else:
                    amount = amt
            if not currency:
                currency = amount_match.group("currency")

    if not round_type:
        round_match = re.search(r"\b(seed|pre-seed|series [a-z]|debt)\b", combined_paras, flags=re.IGNORECASE)
        if round_match:
            r_str = round_match.group(1).lower()
            if "series" in r_str:
                round_type = r_str.title()
            elif "pre-seed" in r_str:
                round_type = "Pre-Seed"
            else:
                round_type = r_str.title()

    if not amount and not round_type and not investors and not startup:
        return None

    return Funding(
        amount=amount,
        currency=currency,
        round=round_type,
        investors=investors,
        startup=startup,
    )


def funding_for(
    text: str,
    category: str | Sequence[str] | None = None,
) -> Funding | None:
    if category is not None:
        is_funding = (
            "funding" in category
            if isinstance(category, (list, tuple, set))
            else category == "funding"
        )
        if not is_funding:
            return None
    return extract_funding(text)


@traceable(name="extract_article", run_type="parser", tags=["extracting"])
def extract_article(
    path: Path,
    manifest_row: dict[str, Any],
) -> ArticleRecord:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    ld = parse_json_ld(soup)
    body = clean_body(html)
    title = (
        ld.get("headline")
        or _text_or_empty(soup, 'meta[property="og:title"]', "content")
        or _text_or_empty(soup, "title")
        or "Untitled"
    )
    author = ld.get("author") or _text_or_empty(soup, 'meta[name="author"]', "content")
    if isinstance(author, list):
        author = [item.get("name", str(item)) if isinstance(item, dict) else str(item) for item in author]
    elif isinstance(author, dict):
        author = author.get("name", "")
    published = (
        ld.get("datePublished")
        or _text_or_empty(soup, 'meta[property="article:published_time"]', "content")
        or _text_or_empty(soup, 'meta[name="datePublished"]', "content")
        or _text_or_empty(soup, 'meta[itemprop="datePublished"]', "content")
    )
    companies, people, locations = extract_named_items(f"{title}. {body}")
    url_str = str(manifest_row.get("url", ""))
    categories = classify_article_categories(title, body, url=url_str, soup=soup)
    category = categories[0] if categories else "general"
    funding_info = funding_for(body, categories)
    if funding_info and funding_info.startup:
        if funding_info.startup not in companies:
            companies.insert(0, funding_info.startup)
    if funding_info and funding_info.investors:
        for inv in funding_info.investors:
            if inv not in companies and inv not in people and inv not in locations:
                companies.append(inv)

    return ArticleRecord(
        id=str(manifest_row["id"]),
        url=str(manifest_row["url"]),
        title=str(title).strip(),
        author=author or "",
        published_at=normalize_datetime(published),
        body=body,
        summary=sentence_summary(body),
        companies=companies,
        people=people,
        locations=locations,
        category=category,
        categories=categories,
        funding=funding_info.model_dump() if funding_info else None,
    )


@traceable(name="extract_pipeline", run_type="chain", tags=["extracting"])
def run_extract(force: bool = False) -> None:
    settings.ensure_dirs()
    manifest_path = settings.raw_dir / "manifest.json"
    manifest = {row["id"]: row for row in read_json(manifest_path)} if manifest_path.exists() else {}
    error_path = settings.records_dir / "extraction_errors.log"
    if force:
        error_path.write_text("", encoding="utf-8")

    html_files = sorted(settings.raw_dir.glob("*.html"))
    total = len(html_files)
    extracted = 0
    skipped = 0

    print(f"Starting extraction for {total} raw articles...")

    for idx, html_path in enumerate(html_files, start=1):
        article_id = html_path.stem
        output_path = settings.records_dir / f"{article_id}.json"
        row = manifest.get(article_id)
        if not row:
            continue
        if not str(row.get("url", "")) or not looks_like_et_tech_article(str(row["url"])):
            if output_path.exists():
                output_path.unlink()
            continue
        if output_path.exists() and not force:
            skipped += 1
            continue
        try:
            record = extract_article(html_path, row)
            write_json(output_path, record.model_dump(mode="json"))
            extracted += 1
            if extracted % 10 == 0 or extracted == 1:
                print(f"  [{idx}/{total}] Extracted: {record.title[:50]}...", flush=True)
        except (ValidationError, ValueError, OSError) as exc:
            if force and output_path.exists():
                output_path.unlink()
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{article_id}: {exc}\n")

    print(
        f"Extraction complete: {extracted} records extracted, "
        f"{skipped} existing records preserved in {settings.records_dir}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract validated article records.")
    parser.add_argument("--force", action="store_true", help="Re-extract existing records.")
    args = parser.parse_args()
    run_extract(force=args.force)


if __name__ == "__main__":
    main()

