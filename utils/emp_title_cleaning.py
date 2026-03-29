import re
import pandas as pd


# Sequential regex replacements to collapse similar job titles
EMP_TITLE_LUMP_RULES = [
    # software
    (r"\bsenior\s+software\s+engineer\b", "senior software engineer"),
    (r"\bsenior\s+software\s+developer\b", "senior software developer"),
    (r"\bsoftware\s+developer\b", "software developer"),
    (r"\bsoftware\s+engineer\b", "software engineer"),
    # nursing & clinical
    (r"\brn\b|\bregistered\s+nurse\b", "registered nurse"),
    (r"\b(lpn|licensed\s+practical\s+nurse|lvn)\b", "licensed practical nurse"),
    (r"\bnurse\s+practitioner\b", "nurse practitioner"),
    (r"\bstaff\s*rn\b", "registered nurse"),
    (r"\bnurse\b", "nurse"),
    # law enforcement / public safety
    (r"\bpolice\s+officer\b", "police officer"),
    (r"\bdeputy\s+sheriff\b", "deputy sheriff"),
    (r"\bcorrection(s|al)\s+officer\b", "corrections officer"),
    (r"\bfirefighter/paramedic\b", "firefighter/paramedic"),
    (r"\bpolice\s+sergeant\b", "police sergeant"),
    # teaching / education
    (r"\bteacher/coach\b", "teacher"),
    (r"\bassistant\s+professor\b", "assistant professor"),
    (r"\bassociate\s+professor\b", "associate professor"),
    (r"\bprofessor\b|\binstructor\b|\beducator\b", "professor"),
    (r"\bspecial\s+education\s+teacher\b", "teacher"),
    (r"\b(substitute\s+)?teacher\b", "teacher"),
    # drivers & carriers
    (r"\btruck\s+driver\b", "truck driver"),
    (r"\bdelivery\s+driver\b", "delivery driver"),
    (r"\bbus\s+driver\b", "bus driver"),
    (r"\broute\s+driver\b", "route driver"),
    (r"\b(driver|conductor)\b", "driver"),
    # assistants / admin
    (r"\badmin(istrative)?\s+(assistant|asst)\b", "administrative assistant"),
    (r"\bexecutive\s+assistant\b", "executive assistant"),
    (r"\boffice\s+manager\b", "office manager"),
    (r"\boffice\s+assistant\b", "office assistant"),
    # sales / account
    (r"\bsales\s+manager\b", "sales manager"),
    (r"\baccount\s+manager\b", "account manager"),
    (r"\binside\s+sales\b", "inside sales"),
    (r"\bsales\s+rep(resentative)?\b", "sales representative"),
    (r"\bsales\s+associate\b", "sales associate"),
    (r"\bsales(?!\w)", "sales"),
    # managers (specific → general)
    (r"\bgeneral\s+manager\b|\bgm\b", "general manager"),
    (r"\bproject\s+manager\b", "project manager"),
    (r"\bprogram\s+manager\b", "program manager"),
    (r"\boperations\s+manager\b", "operations manager"),
    (r"\bregional\s+manager\b", "regional manager"),
    (r"\bdistrict\s+manager\b", "district manager"),
    (r"\bplant\s+manager\b", "plant manager"),
    (r"\bwarehouse\s+manager\b", "warehouse manager"),
    (r"\bbranch\s+manager\b", "branch manager"),
    (r"\bstore\s+manager\b", "store manager"),
    (r"\bproperty\s+manager\b", "property manager"),
    (r"\bservice\s+manager\b", "service manager"),
    (r"\bquality\s+manager\b", "quality manager"),
    (r"\b(it|hr|finance|marketing)\s+manager\b", r"\1 manager"),
    (r"\bmanager\b|\bmgr\b", "manager"),
    # executives & leadership
    (r"\bchief\s+executive\s+officer\b|\bceo\b", "ceo"),
    (r"\bchief\s+financial\s+officer\b|\bcfo\b", "cfo"),
    (r"\bchief\s+operating\s+officer\b|\bcoo\b", "coo"),
    (r"\bchief\s+technology\s+officer\b|\bcto\b", "cto"),
    (r"\bsvp\b|\bsenior\s+vice\s+president\b", "senior vice president"),
    (r"\bavp\b|\bassistant\s+vice\s+president\b", "assistant vice president"),
    (r"\bvp\b|\bvice\s+president\b", "vice president"),
    (r"\bpresident/ceo\b", "ceo"),
    (r"\bpresident\b", "president"),
    (r"\bmanaging\s+director\b", "managing director"),
    (r"\bdirector\b", "director"),
    # engineering (not software)
    (r"\bsenior\s+engineer\b", "senior engineer"),
    (r"\bsystems?\s+engineer\b", "systems engineer"),
    (r"\bmechanical\s+engineer\b", "mechanical engineer"),
    (r"\belectrical\s+engineer\b", "electrical engineer"),
    (r"\bnetwork\s+engineer\b", "network engineer"),
    (r"\bengineer(?!ing)\b|\bengineering\b", "engineer"),
    # finance / accounting / analysis
    (r"\baccounting\s+manager\b", "accounting manager"),
    (r"\b(staff|senior)\s+accountant\b", "accountant"),
    (r"\baccountant\b|\bcpa\b", "accountant"),
    (r"\bfinancial\s+analyst\b", "financial analyst"),
    (r"\b(senior\s+)?business\s+analyst\b", "business analyst"),
    (r"\b(qa|quality\s+assurance)\s+analyst\b", "qa analyst"),
    (r"\banalyst\b", "analyst"),
    # legal
    (r"\battorney\b|\blawyer\b", "attorney"),
    (r"\blegal\s+assistant\b", "legal assistant"),
    (r"\bparalegal\b", "paralegal"),
    # tech support / admins
    (r"\bsystems?\s+administrator\b", "system administrator"),
    (r"\bsystems?\s+analyst\b", "systems analyst"),
    (r"\bsystem\s+engineer\b", "systems engineer"),
    (r"\bit\s+director\b", "it director"),
    (r"\bit\s+consultant\b", "it consultant"),
    (r"\b(it|tech(nician)?)\b", "technician"),
    # trades & ops
    (r"\belectrician\b", "electrician"),
    (r"\bmechanic\b|\bmaintenance\s+mechanic\b", "mechanic"),
    (r"\bwelder\b", "welder"),
    (r"\bplumber\b", "plumber"),
    (r"\bcarpenter\b", "carpenter"),
    (r"\b(machine\s+operator|equipment\s+operator|forklift\s+operator)\b", "machine operator"),
    (r"\bproduction\s+supervisor\b", "production supervisor"),
    # banking / insurance
    (r"\bpersonal\s+banker\b", "personal banker"),
    (r"\bmortgage\s+loan\s+officer\b|\bloan\s+officer\b", "loan officer"),
    (r"\bund(er)?writer\b|\bunderwriter\b", "underwriter"),
    # retail / service
    (r"\bcashier\b", "cashier"),
    (r"\bbartender\b", "bartender"),
    (r"\bserver\b|\bwaiter\b|\bwaitress\b", "server"),
    (r"\bpharmacist\b", "pharmacist"),
    (r"\bpharmacy\s+technician\b|\bpharmacy\s+tech\b", "pharmacy technician"),
    # generic role collapses
    (r"\bsupervisor\b", "supervisor"),
    (r"\boperator\b", "operator"),
    (r"\bassociate\b", "associate"),
    (r"\bcoordinator\b", "coordinator"),
    (r"\badministrator\b", "administrator"),
    (r"\bconsultant\b", "consultant"),
    (r"\bcontroller\b", "controller"),
    (r"\breceptionist\b", "receptionist"),
    (r"\bsecretary\b", "secretary"),
]


def clean_emp_title(text: str) -> str:
    """Basic cleanup for emp_title."""
    if pd.isna(text):
        return "unknown"
    cleaned = str(text).lower()
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9/ ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "unknown"


def lump_emp_title(cleaned_text: str) -> str:
    """Apply regex lumping rules to the cleaned title."""
    t = cleaned_text
    for pattern, replacement in EMP_TITLE_LUMP_RULES:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else "unknown"


def collapse_emp_title(cleaned_text: str) -> str:
    """High-level collapse categories."""
    if cleaned_text in {"unknown", ""}:
        return "unknown"

    if re.search(r"manager|supervisor|director|chief|lead|head", cleaned_text):
        return "management"
    if re.search(r"nurse|medical|health|therap", cleaned_text):
        return "healthcare"
    if re.search(r"teacher|professor|educat|instructor", cleaned_text):
        return "education"
    if re.search(r"account|finance|cpa|bank|financial", cleaned_text):
        return "finance"
    if re.search(r"engineer|engineering|technician|tech|mechanic", cleaned_text):
        return "engineering"
    if re.search(r"driver|truck|transport|delivery", cleaned_text):
        return "transportation"
    if re.search(r"sales|associate|representative|rep", cleaned_text):
        return "sales"
    if re.search(r"army|navy|air force|marines|sergeant|soldier", cleaned_text):
        return "military"
    if re.search(r"construction|electrician|plumber|carpenter|laborer", cleaned_text):
        return "construction"
    if re.search(r"cook|chef|server|bartender|food|restaurant", cleaned_text):
        return "food_service"
    if re.search(r"city|county|government|federal|state", cleaned_text):
        return "government"
    if re.search(r"retail|cashier|store", cleaned_text):
        return "retail"
    if re.search(r"customer service|csr|support", cleaned_text):
        return "customer_service"
    if re.search(r"software|developer|it|information technology", cleaned_text):
        return "it_software"
    if re.search(r"maintenance|janitor|custodian|grounds", cleaned_text):
        return "maintenance"
    if re.search(r"lawyer|attorney|legal|paralegal", cleaned_text):
        return "legal"
    if re.search(r"manufacturing|factory|assembler|operator", cleaned_text):
        return "manufacturing"
    if re.search(r"real estate|realtor|broker", cleaned_text):
        return "real_estate"
    if re.search(r"self|owner|ceo|founder", cleaned_text):
        return "self_employed"

    return "other"


def process_emp_title_features(df: pd.DataFrame, source_col: str = "emp_title") -> pd.DataFrame:
    """Add cleaned / lumped / collapsed variants of emp_title."""
    df = df.copy()
    df[source_col] = df[source_col].fillna("")
    df["emp_title_clean"] = df[source_col].apply(clean_emp_title)
    df["emp_title_lumped"] = df["emp_title_clean"].apply(lump_emp_title)
    df["emp_title_collaposed_high"] = df["emp_title_clean"].apply(collapse_emp_title)
    return df
