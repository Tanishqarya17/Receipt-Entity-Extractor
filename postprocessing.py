"""
Field-specific post-processing rules for SROIE-style entity extraction.

Generated on Day 19. Each rule is justified by an observation from the
test-set error analysis (Days 13-18).
"""

import re


DATE_CORE_PATTERN = re.compile(
    r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2})'
)
DAYS_OF_WEEK = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
                'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
                'SATURDAY', 'SUNDAY']
CURRENCY_PATTERNS = re.compile(
    r'^\s*(RM|MYR|USD|\$|S\$|SGD|HK\$|HKD|EUR|€|£|GBP|INR|₹|RP|IDR)\s*',
    re.IGNORECASE,
)
TRAILING_GARBAGE = re.compile(r'[^\d\.,]+$')


def clean_date(s):
    if not s:
        return ''
    match = DATE_CORE_PATTERN.search(s)
    if match:
        return match.group(0)
    cleaned = s.upper()
    for dow in DAYS_OF_WEEK:
        cleaned = re.sub(rf'\b{dow}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{1,2}[:\.]\d{2}(?:[:\.]\d{2})?\b', '', cleaned)
    return cleaned.strip()


def clean_total(s):
    if not s:
        return ''
    s = s.strip()
    s = CURRENCY_PATTERNS.sub('', s)
    s = TRAILING_GARBAGE.sub('', s)
    match = re.search(r'\d+[.,]\d{2}', s)
    if match:
        return match.group(0).replace(',', '.')
    match = re.search(r'\d+(?:[.,]\d+)?', s)
    return match.group(0).replace(',', '.') if match else ''


def clean_company(s):
    if not s:
        return ''
    s = ' '.join(s.split()).strip()
    s = s.strip('.,;:!?"\'()-')
    return s


def clean_address(s):
    if not s:
        return ''
    s = ' '.join(s.split())
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r'\s*\.\s+', '. ', s)
    return s.strip(' ,.')


FIELD_CLEANERS = {
    'company': clean_company,
    'date':    clean_date,
    'address': clean_address,
    'total':   clean_total,
}


def postprocess(entities):
    """Apply field-specific cleaners to an entities dict."""
    return {f: FIELD_CLEANERS[f](v) for f, v in entities.items()}


# ---- Date recall fallback (added after per-field audit) ----
_DATE_NUMERIC = re.compile(
    r'\b(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2})\b'
)
_DATE_TEXT = re.compile(
    r'\b(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}'
    r'|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{1,2},?\s+\d{2,4})\b',
    re.IGNORECASE,
)

def recover_date(words):
    if not words:
        return ''
    text = ' '.join(words)
    m = _DATE_NUMERIC.search(text)
    if m:
        return m.group(1)
    m = _DATE_TEXT.search(text)
    if m:
        return m.group(1)
    return ''
