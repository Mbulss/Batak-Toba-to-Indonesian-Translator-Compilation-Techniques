# batak_compiler.py
"""
Batak -> Indonesian Compiler (single-file)

Features:
 - Lexer (tokenizes text into WORD, NUMBER, PUNCT, EOF)
 - Efficient POS tagger (class-based + morphology heuristics)
 - Phrase lookup (longest-match for multi-word lexicon entries)
 - Recursive-descent parser (S -> [INTJ] CLAUSE)
 - Grammar validation and structured error reporting
 - Translation engine combining phrase_lookup + parse-based reordering
 - CLI demo at the bottom
"""

import re
import json
import os
from typing import List, Tuple, Optional, Dict, Any


# -----------------------
# Token & TokenType
# -----------------------
class TokenType:
    WORD = "WORD"
    NUMBER = "NUMBER"
    PUNCT = "PUNCT"
    EOF = "EOF"


class Token:
    def __init__(self, type_: str, value: str, position: int):
        self.type = type_
        self.value = value
        self.position = position
        self.norm = self.normalize(value)
        self.pos_tag: Optional[str] = None  # filled by POS tagger

    def __repr__(self):
        return f"Token({self.type}, '{self.value}', pos={self.position}, tag={self.pos_tag})"

    @staticmethod
    def normalize(s: str) -> str:
        return re.sub(r"[^\w]", "", s).lower()


# -----------------------
# Lexer
# ----------------------
class Lexer:
    """
    Master-regex lexer: yields Token objects in order.
    """

    def __init__(self, text: str):
        self.text = text
        self.length = len(text)
        # token specs (ordered)
        self.token_specs: List[Tuple[str, str]] = [
            (TokenType.WORD, r"[A-Za-zÀ-ÿ]+"),
            (TokenType.NUMBER, r"\d+"),
            (TokenType.PUNCT, r"[.,!?;:\(\)\-\[\]\"']"),
            # whitespace is skipped
        ]
        self.master_regex = re.compile(
            "|".join(f"(?P<{t}>{p})" for t, p in self.token_specs), re.UNICODE
        )

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        for m in self.master_regex.finditer(self.text):
            typ = m.lastgroup
            val = m.group(typ)
            pos = m.start()
            tokens.append(Token(typ, val, pos))
        tokens.append(Token(TokenType.EOF, "", self.length))
        return tokens


# -----------------------
# Lexicon / phrase lookup (multi-word)
# -----------------------
def load_lexicon(json_path: str = "batak_to_indo_lexicon.json") -> Dict[str, str]:
    """
    Load lexicon from JSON file.
    JSON format: {"batak_word": ["indonesian1", "indonesian2", ...]}
    Returns: {"batak_word": "indonesian1"} (takes first translation)
    """
    lexicon: Dict[str, str] = {}
    
    # Fallback minimal lexicon
    fallback_lexicon = {
        "horas": "halo",
        "horas ma di hamu": "halo semuanya",
        "ma": "",  # often untranslated particle
        "di": "di",
        "tu": "ke",
        "au": "saya",
        "ho": "kamu",
        "ia": "dia",
        "hita": "kita",
        "hamu": "kalian",
        "dongan": "teman",
        "manangi": "pergi",
        "manangih": "menangis",
        "manangihon": "memberi",
        "mangalului": "mencari",
        "marhobas": "membaca",
        "mangan": "makan",
        "marniat": "bernyanyi",
        "mangulahon": "melakukan",
        "mambahen": "melakukan",
        "ulaon": "pekerjaan",  # or "karya"
        "salah": "salah",
        "jabu": "rumah",
        "pasar": "pasar",
        "buku": "buku",
        "rohang": "rumah",
        "on": "",  # suffix/ignored
        "do": "adalah",
        # Conjunctions
        "alai": "tetapi",
        "tapi": "tetapi",
        "dohot": "dan",
        "jala": "dan",
        "alana": "karena",
        "jadi": "jadi",
        "asa": "agar",
        "sai": "agar",
        "molo": "jika",
        # Negation
        "ndang": "tidak",
        "dang": "tidak",
    }
    
    # Try to load from JSON file
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_lexicon = json.load(f)
                # Convert list values to string (take first translation)
                for key, value_list in json_lexicon.items():
                    if isinstance(value_list, list) and len(value_list) > 0:
                        lexicon[key] = value_list[0]  # Use first translation
                    elif isinstance(value_list, str):
                        lexicon[key] = value_list
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load lexicon from {json_path}: {e}")
            print("Using fallback lexicon.")
            return fallback_lexicon
    else:
        print(f"Warning: Lexicon file '{json_path}' not found. Using fallback lexicon.")
        return fallback_lexicon
    
    # Merge with fallback (fallback takes precedence for duplicates)
    lexicon.update(fallback_lexicon)
    
    return lexicon


# Load the lexicon (will use JSON file if available)
LEXICON: Dict[str, str] = load_lexicon()


def phrase_lookup(tokens: List[str], lex: Dict[str, str], max_lookahead: int = 5) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """
    Longest-match phrase lookup.
    tokens: list of original token strings (preserve punctuation tokens)
    returns:
      - translated_tokens: list aligned to tokens length (some items can be multi-token phrases)
      - spans: list of (start_idx, end_idx, key) for matched lexicon entries
    Unknown word fallback: keep bracketed original [word]
    """
    n = len(tokens)
    i = 0
    out = []
    spans: List[Tuple[int, int, str]] = []
    while i < n:
        t = tokens[i]
        # punctuation passthrough (no lookup)
        if re.fullmatch(r"[^\w\s]", t):
            out.append(t)
            i += 1
            continue

        matched = False
        max_L = min(max_lookahead, n - i)
        for L in range(max_L, 0, -1):
            phrase = " ".join(tokens[i : i + L])
            key = " ".join(re.findall(r"\w+", phrase)).lower()
            if key and key in lex:
                val = lex[key].strip()
                out.append(val if val != "" else "")
                spans.append((i, i + L, key))
                i += L
                matched = True
                break
        if matched:
            continue

        # fallback single token
        key = re.sub(r"[^\w]", "", t).lower()
        if key and key in lex:
            out.append(lex[key])
        else:
            out.append(f"[{t}]")
        i += 1
    return out, spans


# -----------------------
# POS Lexicon and efficient class sets
# -----------------------
# Keep small sets of classes; morphological heuristics for verbs.
POS_CLASSES = {
    "PRON": {"au", "ho", "ia", "hita", "hamu", "kami", "kita"},
    "PART": {"ma", "on", "na", "do"},
    "PREP": {"tu", "di"},
    "INTJ": {"horas"},
    "CONJ": {"dohot", "alana", "jala", "jadi", "asa", "sai", "molo", "alai", "tapi"},
    "NOUN_BASE": {"pasar", "buku", "rohang", "dongan", "anak"},
}

VERB_PREFIXES = ("ma", "mar", "man", "mang", "mam")
VERB_SUFFIXES = ("on", "i", "as", "an")


def tag_pos(tokens: List[Token]) -> List[Token]:
    """
    Assign POS tags in-place to tokens using:
      - class membership
      - morphological heuristics (prefix/suffix)
      - default fallback = NOUN
    """
    pronouns = POS_CLASSES["PRON"]
    parts = POS_CLASSES["PART"]
    preps = POS_CLASSES["PREP"]
    intjs = POS_CLASSES["INTJ"]
    conjunctions = POS_CLASSES["CONJ"]
    nouns_base = POS_CLASSES["NOUN_BASE"]

    for tok in tokens:
        if tok.type == TokenType.EOF:
            tok.pos_tag = "EOF"
            continue
        if tok.type == TokenType.PUNCT:
            tok.pos_tag = "PUNCT"
            continue
        if tok.type == TokenType.NUMBER:
            tok.pos_tag = "NUM"
            continue

        key = tok.norm
        if not key:
            tok.pos_tag = "UNKNOWN"
            continue

        if key in pronouns:
            tok.pos_tag = "PRON"
            continue
        if key in parts:
            tok.pos_tag = "PART"
            continue
        if key in preps:
            tok.pos_tag = "PREP"
            continue
        if key in intjs:
            tok.pos_tag = "INTJ"
            continue
        if key in conjunctions:
            tok.pos_tag = "CONJ"
            continue
        if key in nouns_base:
            tok.pos_tag = "NOUN"
            continue

        # morphological heuristics: verb prefixes/suffixes
        if any(key.startswith(pref) for pref in VERB_PREFIXES) or any(key.endswith(sfx) for sfx in VERB_SUFFIXES):
            tok.pos_tag = "VERB"
            continue

        # fallback to NOUN
        tok.pos_tag = "NOUN"
    return tokens


# -----------------------
# Grammar error types
# -----------------------
class GrammarError:
    def __init__(self, code: str, desc: str, details: Optional[str] = None, position: Optional[int] = None):
        self.code = code
        self.desc = desc
        self.details = details
        self.position = position

    def __repr__(self):
        pos = f" at pos {self.position}" if self.position is not None else ""
        det = f" ({self.details})" if self.details else ""
        return f"{self.code}: {self.desc}{det}{pos}"


# -----------------------
# Parser (recursive-descent)
# -----------------------
class ParseError(Exception):
    pass


class Node:
    def __init__(self, node_type: str, children: Optional[List[Any]] = None, token: Optional[Token] = None):
        self.type = node_type
        self.children = children if children is not None else []
        self.token = token

    def __repr__(self):
        if self.token:
            return f"{self.type}({self.token.value}/{self.token.pos_tag})"
        return f"{self.type}({self.children})"


class Parser:
    """
    Parser for a small Batak grammar:
      Sentence -> [INTJ] Clause EOF
      Clause   -> NP VP
      NP       -> PRON | (NOUN | PART)+
      VP       -> (PART)? VERB (NP)? (PP)? (PART)?
      PP       -> PREP NP
    """
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0
        self.errors: List[GrammarError] = []

    def peek(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        if tok.type != TokenType.EOF:
            self.i += 1
        return tok

    def parse(self) -> Node:
        root = Node("S", [])
        # optionally INTJ
        if self.peek().pos_tag == "INTJ":
            root.children.append(Node("INTJ", token=self.advance()))
        
        # Parse first clause
        try:
            clause = self.parse_clause()
            root.children.append(clause)
        except ParseError as e:
            # record error and attempt recovery
            self.errors.append(GrammarError("PARSE_ERROR", str(e), position=self.peek().position))
            # Try to continue parsing remaining clauses
            pass
        
        # Parse additional clauses (compound/complex sentences)
        # Grammar: Clause (CONJ Clause | PUNCT Clause)*
        while self.peek().type != TokenType.EOF:
            # Skip punctuation between clauses
            while self.peek().type == TokenType.PUNCT:
                root.children.append(Node("PUNCT", token=self.advance()))
            
            if self.peek().type == TokenType.EOF:
                break
            
            # Check for conjunction
            if self.peek().pos_tag == "CONJ":
                conj_node = Node("CONJ", token=self.advance())
                root.children.append(conj_node)
                # Try to parse next clause
                try:
                    next_clause = self.parse_clause()
                    root.children.append(next_clause)
                except ParseError as e:
                    self.errors.append(GrammarError("PARSE_ERROR", str(e), position=self.peek().position))
                    # Consume remaining as EXTRA
                    while self.peek().type != TokenType.EOF:
                        root.children.append(Node("EXTRA", token=self.advance()))
                    break
            # If no conjunction but there are more tokens, try parsing as next clause
            # (handles cases where clauses are separated by commas only)
            elif self.peek().pos_tag in ("PRON", "NOUN", "PART"):
                try:
                    next_clause = self.parse_clause()
                    root.children.append(next_clause)
                except ParseError:
                    # If parsing fails, consume remaining as EXTRA
                    while self.peek().type != TokenType.EOF:
                        root.children.append(Node("EXTRA", token=self.advance()))
                    break
            else:
                # Unknown token, consume as EXTRA
                while self.peek().type != TokenType.EOF:
                    root.children.append(Node("EXTRA", token=self.advance()))
                break
        
        # consume trailing punctuation
        while self.peek().type == TokenType.PUNCT:
            root.children.append(Node("PUNCT", token=self.advance()))
        # append EOF
        root.children.append(Node("EOF", token=self.peek()))
        return root

    def parse_clause(self) -> Node:
        # Check if this is a verb-initial clause (starts with VERB)
        if self.peek().pos_tag == "VERB":
            return self.parse_verb_initial_clause()
        
        # Standard clause: NP VP
        np = self.parse_np(expect_subject=True)
        vp = self.parse_vp()
        return Node("CLAUSE", [np, vp])
    
    def parse_verb_initial_clause(self) -> Node:
        """Parse verb-initial clauses, including focus constructions with 'do'."""
        verb_node = Node("VERB", token=self.advance())
        children = [verb_node]
        
        # Parse first NP (could be agent or object)
        if self.peek().pos_tag in ("PRON", "NOUN", "PART"):
            first_np = self.parse_np(expect_subject=False)
            children.append(first_np)
            
            # Check for focus marker "do"
            if self.peek().type == TokenType.WORD and self.peek().norm == "do":
                focus_marker = Node("FOCUS", token=self.advance())
                children.append(focus_marker)
                
                # Parse focused NP (patient)
                if self.peek().pos_tag in ("PRON", "NOUN", "PART"):
                    focused_np = self.parse_np(expect_subject=False)
                    # Mark first NP as AGENT and focused NP as PATIENT
                    return Node("CLAUSE", [
                        verb_node,
                        Node("AGENT", [first_np]),
                        focus_marker,
                        Node("PATIENT", [focused_np])
                    ])
                else:
                    # "do" without following NP - still mark as focus construction
                    return Node("CLAUSE", [
                        verb_node,
                        Node("AGENT", [first_np]),
                        focus_marker
                    ])
            else:
                # Verb-initial without focus: VERB NP (object)
                return Node("CLAUSE", [verb_node, Node("OBJECT", [first_np])])
        else:
            # Just verb, no arguments
            return Node("CLAUSE", [verb_node])

    def parse_np(self, expect_subject: bool = False) -> Node:
        tok = self.peek()
        if tok.pos_tag == "PRON":
            return Node("NP", [Node("PRON", token=self.advance())])
        # allow sequences of NOUN/PART as NP
        if tok.pos_tag in ("NOUN", "PART"):
            children = []
            while self.peek().pos_tag in ("NOUN", "PART"):
                child_tok = self.advance()
                # Preserve actual POS tag in node type
                children.append(Node(child_tok.pos_tag, token=child_tok))
            return Node("NP", children)
        # if subject expected but not found, raise
        if expect_subject:
            raise ParseError(f"Expected subject NP but found '{tok.value}/{tok.pos_tag}'")
        # fallback empty NP
        return Node("NP", [])

    def parse_vp(self) -> Node:
        # optional leading particle
        if self.peek().pos_tag == "PART":
            lead = Node("PART", token=self.advance())
        else:
            lead = None

        tok = self.peek()
        if tok.pos_tag != "VERB":
            if lead:
                # we consumed a leading particle; check next
                raise ParseError(f"Expected VERB after particle but found '{tok.value}/{tok.pos_tag}' at pos {tok.position}")
            else:
                raise ParseError(f"Expected VERB in VP but found '{tok.value}/{tok.pos_tag}' at pos {tok.position}")

        verb_node = Node("VERB", token=self.advance())
        children = []
        if lead:
            children.append(lead)
        children.append(verb_node)

        # optional object NP (PRON, NOUN, or PART)
        if self.peek().pos_tag in ("PRON", "NOUN", "PART"):
            obj_np = self.parse_np(expect_subject=False)
            children.append(obj_np)

        # optional PP
        if self.peek().pos_tag == "PREP":
            children.append(self.parse_pp())

        # optional trailing particle
        if self.peek().pos_tag == "PART":
            children.append(Node("PART", token=self.advance()))

        return Node("VP", children)

    def parse_pp(self) -> Node:
        prep_tok = self.advance()  # must be PREP
        try:
            np = self.parse_np(expect_subject=False)
        except ParseError:
            np = Node("NP", [])
            self.errors.append(GrammarError("PP_ERROR", "Invalid NP in PP", position=prep_tok.position))
        return Node("PP", [Node("PREP", token=prep_tok), np])


# -----------------------
# Utilities: collect leaves, extract SVO
# -----------------------
def collect_leaf_tokens(node: Node) -> List[Token]:
    leaves: List[Token] = []
    if node.token:
        leaves.append(node.token)
    for c in node.children:
        leaves.extend(collect_leaf_tokens(c))
    return leaves


def find_first_leaf_by_type(node: Node, node_type: str) -> Optional[Token]:
    if node.type == node_type and node.token:
        return node.token
    for c in node.children:
        res = find_first_leaf_by_type(c, node_type)
        if res:
            return res
    return None


def extract_svo(tree: Node) -> Dict[str, Optional[str]]:
    subj = None
    verb = None
    obj = None
    agent = None
    patient = None
    has_focus = False
    clause = None
    for c in tree.children:
        if c.type == "CLAUSE":
            clause = c
            break
    if not clause:
        return {"subject": None, "verb": None, "object": None, "has_focus": False}

    # Check for focus construction (verb-initial with FOCUS marker)
    has_focus_marker = any(ch.type == "FOCUS" for ch in clause.children)
    has_agent = any(ch.type == "AGENT" for ch in clause.children)
    has_patient = any(ch.type == "PATIENT" for ch in clause.children)
    
    if has_focus_marker and has_agent and has_patient:
        # Focus construction: VERB AGENT FOCUS PATIENT
        has_focus = True
        vtok = find_first_leaf_by_type(clause, "VERB")
        verb = vtok.value if vtok else None
        
        # Extract agent (NP before "do")
        agent_node = None
        for ch in clause.children:
            if ch.type == "AGENT":
                agent_node = ch
                break
        if agent_node and agent_node.children:
            agent_tokens = collect_leaf_tokens(agent_node.children[0])
            agent = " ".join(t.value for t in agent_tokens) if agent_tokens else None
        
        # Extract patient (NP after "do", focused)
        patient_node = None
        for ch in clause.children:
            if ch.type == "PATIENT":
                patient_node = ch
                break
        if patient_node and patient_node.children:
            patient_tokens = collect_leaf_tokens(patient_node.children[0])
            patient = " ".join(t.value for t in patient_tokens) if patient_tokens else None
        
        return {
            "subject": None,  # No subject in focus construction
            "verb": verb,
            "object": None,  # Use agent/patient instead
            "agent": agent,
            "patient": patient,
            "has_focus": True
        }
    
    # Standard NP VP structure
    # subject
    if len(clause.children) >= 1:
        first_child = clause.children[0]
        if first_child.type == "NP":
            subj_tokens = collect_leaf_tokens(first_child)
            subj = " ".join(t.value for t in subj_tokens) if subj_tokens else None
        elif first_child.type == "VERB":
            # Verb-initial without focus
            vtok = find_first_leaf_by_type(first_child, "VERB")
            verb = vtok.value if vtok else None
            if len(clause.children) >= 2:
                obj_node = clause.children[1]
                if obj_node.type == "OBJECT":
                    obj_tokens = collect_leaf_tokens(obj_node.children[0]) if obj_node.children else []
                    obj = " ".join(t.value for t in obj_tokens) if obj_tokens else None
            return {"subject": None, "verb": verb, "object": obj, "has_focus": False}
    
    # verb and object
    if len(clause.children) >= 2:
        vp = clause.children[1]
        vtok = find_first_leaf_by_type(vp, "VERB")
        verb = vtok.value if vtok else None
        # first NP inside VP is object
        obj_np = None
        for ch in vp.children:
            if ch.type == "NP":
                obj_np = ch
                break
        if obj_np:
            obj_toks = collect_leaf_tokens(obj_np)
            obj = " ".join(t.value for t in obj_toks) if obj_toks else None

    return {"subject": subj, "verb": verb, "object": obj, "has_focus": False}


# -----------------------
# Grammar validation (additional checks)
# -----------------------
def validate_grammar(tokens: List[Token], tree: Node, parser_errors: List[GrammarError]) -> List[GrammarError]:
    """
    Produce additional grammar checks:
      - unknown words (not in lexicon)
      - missing subject/verb
      - 'tu' misplaced (before verb)
      - 'do' misuse
      - double verbs
    """
    errors: List[GrammarError] = list(parser_errors)  # start with parser reported errors

    # token strings
    token_norms = [t.norm for t in tokens if t.type != TokenType.EOF]
    token_values = [t.value for t in tokens if t.type != TokenType.EOF]
    
    # Check which tokens are part of matched phrases (to avoid false UNKNOWN_WORDS)
    # Run phrase_lookup to see which tokens are in phrases
    matched_token_indices = set()
    try:
        _, spans = phrase_lookup(token_values, LEXICON)
        for span_start, span_end, key in spans:
            for idx in range(span_start, span_end):
                matched_token_indices.add(idx)
    except:
        pass  # If phrase_lookup fails, continue without phrase info
    
    # unknown tokens (exclude tokens that are part of matched phrases)
    unk = []
    for idx, t in enumerate(tokens):
        if t.type == TokenType.WORD and t.norm:
            # Skip if token is part of a matched phrase
            if idx in matched_token_indices:
                continue
            # Check if word is in lexicon or is a known POS class
            if t.norm not in LEXICON and t.pos_tag not in ("PRON", "PREP", "PART", "INTJ", "NUM", "PUNCT"):
                unk.append(t.value)
    if unk:
        errors.append(GrammarError("UNKNOWN_WORDS", "Kata tidak dikenal", details=", ".join(unk)))

    # missing subject/verb checks
    svo = extract_svo(tree)
    # Focus constructions don't have a traditional subject, they have agent/patient
    if not svo.get("has_focus"):
        if not svo.get("subject"):
            errors.append(GrammarError("MISSING_SUBJECT", "Subjek tidak ditemukan"))
    if not svo.get("verb"):
        errors.append(GrammarError("MISSING_VERB", "Kata kerja tidak ditemukan"))

    # 'tu' misuse: if 'tu' appears before the first verb token
    first_verb_idx = None
    first_tu_idx = None
    for idx, t in enumerate(tokens):
        if t.type == TokenType.WORD and t.norm == "tu" and first_tu_idx is None:
            first_tu_idx = idx
        if t.pos_tag == "VERB" and first_verb_idx is None:
            first_verb_idx = idx
    if first_tu_idx is not None:
        if first_verb_idx is None:
            errors.append(GrammarError("TU_WITHOUT_VERB", "'tu' muncul tetapi tidak ada kata kerja", position=tokens[first_tu_idx].position))
        elif first_tu_idx < first_verb_idx:
            errors.append(GrammarError("TU_MISPLACED", "'tu' ditempatkan sebelum kata kerja", position=tokens[first_tu_idx].position))

    # 'do' misuse: should be followed by NP (unless it's a focus marker in verb-initial clause)
    # Check if this is a focus construction: VERB ... do NP
    for idx, t in enumerate(tokens):
        if t.type == TokenType.WORD and t.norm == "do":
            # Check if there's a verb before "do" (focus construction pattern: VERB ... do NP)
            is_focus_construction = False
            for prev_idx in range(max(0, idx - 3), idx):
                if tokens[prev_idx].pos_tag == "VERB":
                    is_focus_construction = True
                    break
            
            # Also check if sentence starts with verb (verb-initial focus construction)
            if idx > 0 and tokens[0].pos_tag == "VERB":
                is_focus_construction = True
            
            if not is_focus_construction:
                # next non-punct token
                j = idx + 1
                while j < len(tokens) and tokens[j].type == TokenType.PUNCT:
                    j += 1
                if j >= len(tokens) or tokens[j].pos_tag not in ("NOUN", "PRON"):
                    errors.append(GrammarError("DO_NO_NP", "'do' tidak diikuti NP", position=t.position))

    # double verbs: adjacent verbs (but exclude if they're part of a matched phrase)
    verb_indices = [i for i, t in enumerate(tokens) if t.pos_tag == "VERB"]
    for a in range(len(verb_indices) - 1):
        if verb_indices[a + 1] - verb_indices[a] <= 1:
            # Check if these adjacent verbs are part of a matched phrase
            verb1_idx = verb_indices[a]
            verb2_idx = verb_indices[a + 1]
            # If both verbs are in the same phrase span, it's a valid phrase (not double verb)
            both_in_phrase = False
            try:
                _, spans = phrase_lookup(token_values, LEXICON)
                for span_start, span_end, key in spans:
                    if span_start <= verb1_idx < span_end and span_start <= verb2_idx < span_end:
                        both_in_phrase = True
                        break
            except:
                pass
            
            if not both_in_phrase:
                errors.append(GrammarError("DOUBLE_VERB", "Dua kata kerja berdampingan", position=tokens[verb_indices[a]].position))
                break

    # deduplicate by code
    dedup = {}
    for e in errors:
        dedup[e.code] = e
    return list(dedup.values())


# -----------------------
# Translation: combine parse + phrase_lookup
# -----------------------
def translate_by_parse(tree: Node, tokens: List[Token], lexicon: Dict[str, str]) -> str:
    """
    Unified translation approach:
      - Always uses phrase_lookup first (longest-match for phrases like "mangulahon ulaon")
      - For simple sentences: extracts SVO components and reorders
      - For complex sentences: uses phrase_lookup results as-is
    """
    # rebuild token list strings for phrase lookup
    toks_values = [t.value for t in tokens if t.type != TokenType.EOF]

    # Helper: translate a phrase string (word-by-word fallback)
    def translate_phrase(phrase: Optional[str]) -> str:
        if not phrase:
            return ""
        parts = []
        for w in re.findall(r"[A-Za-zÀ-ÿ]+", phrase):
            parts.append(lexicon.get(w.lower(), w))
        return " ".join([p for p in parts if p])
    
    # Helper: build mapping from original token indices to phrase_lookup results
    def build_token_mapping(spans: List[Tuple[int, int, str]], num_tokens: int) -> Dict[int, int]:
        """Map original token index to phrase_lookup result index."""
        orig_to_phrase_idx = {}
        phrase_idx = 0
        
        # Map tokens covered by spans
        for span_start, span_end, key in spans:
            for orig_idx in range(span_start, span_end):
                orig_to_phrase_idx[orig_idx] = phrase_idx
            phrase_idx += 1
        
        # Map tokens not in spans (single-word translations)
        for orig_idx in range(num_tokens):
            if orig_idx not in orig_to_phrase_idx:
                spans_before = sum(1 for s, e, _ in spans if e <= orig_idx)
                single_before = sum(1 for i in range(orig_idx) if i not in orig_to_phrase_idx)
                orig_to_phrase_idx[orig_idx] = spans_before + single_before
        
        return orig_to_phrase_idx
    
    # Helper: extract translation from phrase_lookup results by finding token in original
    def extract_from_phrase_lookup(text: str, orig_to_phrase_idx: Dict[int, int], 
                                   trans_tokens: List[str], toks_values: List[str]) -> str:
        """Find text in original tokens and return corresponding phrase_lookup translation."""
        if not text:
            return ""
        words = text.lower().split()
        if not words:
            return ""
        
        # Find first word in original tokens
        for i, tok_val in enumerate(toks_values):
            if tok_val.lower() == words[0]:
                phrase_idx = orig_to_phrase_idx.get(i, -1)
                if phrase_idx >= 0 and phrase_idx < len(trans_tokens):
                    result = trans_tokens[phrase_idx]
                    # Check if text spans multiple tokens
                    if len(words) > 1:
                        # Try to get additional tokens if needed
                        for j in range(1, min(len(words), len(toks_values) - i)):
                            if i + j < len(toks_values) and toks_values[i + j].lower() == words[j]:
                                next_phrase_idx = orig_to_phrase_idx.get(i + j, -1)
                                if next_phrase_idx >= 0 and next_phrase_idx < len(trans_tokens) and next_phrase_idx != phrase_idx:
                                    result += " " + trans_tokens[next_phrase_idx]
                    return result
                break
        
        # Fallback to word-by-word
        return translate_phrase(text)
    
    # STEP 1: Always use phrase_lookup first to catch multi-word phrases (longest-match)
    trans_tokens, spans = phrase_lookup(toks_values, lexicon)
    
    # STEP 2: Check if we can safely apply SVO reordering (only for simple, well-parsed sentences)
    clause_count = sum(1 for c in tree.children if c.type == "CLAUSE")
    has_conjunctions = any(c.type == "CONJ" for c in tree.children)
    has_extra = any(c.type == "EXTRA" for c in tree.children)
    can_reorder = (clause_count == 1 and not has_conjunctions and not has_extra)
    
    # STEP 3: If we can't reorder, just return phrase_lookup results
    if not can_reorder:
        out = " ".join([t for t in trans_tokens if t])
        out = re.sub(r"\s+([.,!?;:])", r"\1", out)
        out = re.sub(r"\s+", " ", out).strip()
        if out and not re.search(r"[.!?]$", out):
            out += "."
        if out:
            out = out[0].upper() + out[1:]
        return out
    
    # STEP 4: For simple sentences, extract SVO and reorder using phrase_lookup results
    svo = extract_svo(tree)
    orig_to_phrase_idx = build_token_mapping(spans, len(toks_values))
    
    # Handle focus construction: "VERB AGENT do PATIENT" → "AGENT-lah yang VERB PATIENT-object"
    # Example: "Mangalului ho do au" → "Kamulah yang mencariku"
    if svo.get("has_focus") and svo.get("agent") and svo.get("patient") and svo.get("verb"):
        # Extract from phrase_lookup results
        agent_tr = extract_from_phrase_lookup(svo["agent"], orig_to_phrase_idx, trans_tokens, toks_values)
        verb_tr = extract_from_phrase_lookup(svo["verb"], orig_to_phrase_idx, trans_tokens, toks_values)
        patient_tr = extract_from_phrase_lookup(svo["patient"], orig_to_phrase_idx, trans_tokens, toks_values)
        
        # Convert patient to object form if it's a pronoun
        # "saya" → "ku", "kamu" → "mu", "dia" → "nya"
        patient_obj = patient_tr
        if patient_tr.lower() == "saya":
            patient_obj = "ku"
        elif patient_tr.lower() == "kamu":
            patient_obj = "mu"
        elif patient_tr.lower() == "dia":
            patient_obj = "nya"
        # Add space if patient_obj is not a suffix
        if patient_obj in ("ku", "mu", "nya"):
            # Object pronoun attaches to verb
            out = f"{agent_tr}-lah yang {verb_tr}{patient_obj}"
        else:
            # Regular noun, keep separate
            out = f"{agent_tr}-lah yang {verb_tr} {patient_obj}"
        
        out = re.sub(r"\s+([.,!?;:])", r"\1", out)
        if out and not re.search(r"[.!?]$", out):
            out += "."
        if out:
            out = out[0].upper() + out[1:]
        return out
    
    # Standard SVO translation
    # Extract components from phrase_lookup results, then reorder based on SVO structure
    if svo.get("subject") and svo.get("verb"):
        # Extract subject from phrase_lookup results
        subj_tr = extract_from_phrase_lookup(svo["subject"], orig_to_phrase_idx, trans_tokens, toks_values)
        
        # Extract verb+object (may be a phrase like "mangulahon ulaon")
        # Find verb position first (needed for both object and PP checks)
        verb_start_idx = None
        if svo.get("verb"):
            verb_words = svo["verb"].lower().split()
            for i, tok_val in enumerate(toks_values):
                if tok_val.lower() == verb_words[0]:
                    verb_start_idx = i
                    break
        
        verb_obj_tr = ""
        if svo.get("verb"):
            verb_tr = extract_from_phrase_lookup(svo["verb"], orig_to_phrase_idx, trans_tokens, toks_values)
            if svo.get("object"):
                # Check if verb and object are in the same phrase (like "mangulahon ulaon" → "bekerja")
                
                if verb_start_idx is not None:
                    verb_phrase_idx = orig_to_phrase_idx.get(verb_start_idx, -1)
                    obj_words = svo["object"].lower().split()
                    obj_start_idx = verb_start_idx + len(verb_words)
                    
                    if obj_start_idx < len(toks_values):
                        obj_phrase_idx = orig_to_phrase_idx.get(obj_start_idx, -1)
                        # If verb and object map to same phrase_idx, they're a single phrase
                        if verb_phrase_idx == obj_phrase_idx and verb_phrase_idx >= 0 and verb_phrase_idx < len(trans_tokens):
                            verb_obj_tr = trans_tokens[verb_phrase_idx]  # Single phrase like "bekerja"
                        else:
                            # Separate: verb + object
                            obj_tr = extract_from_phrase_lookup(svo["object"], orig_to_phrase_idx, trans_tokens, toks_values)
                            verb_obj_tr = " ".join([p for p in (verb_tr, obj_tr) if p])
                    else:
                        verb_obj_tr = verb_tr
                else:
                    verb_obj_tr = verb_tr
            else:
                verb_obj_tr = verb_tr
        
        # Extract PP translation
        # But first check if verb and prep are part of the same phrase (like "mangido tu")
        verb_prep_in_phrase = False
        if svo.get("verb") and verb_start_idx is not None:
            verb_phrase_idx = orig_to_phrase_idx.get(verb_start_idx, -1)
            # Check if the token after verb is a PREP and if they're in the same phrase
            if verb_start_idx + 1 < len(toks_values):
                next_tok_val = toks_values[verb_start_idx + 1]
                # Check if next token is a PREP
                next_is_prep = False
                for t in tokens:
                    if t.value == next_tok_val and t.pos_tag == "PREP":
                        next_is_prep = True
                        break
                if next_is_prep:
                    next_phrase_idx = orig_to_phrase_idx.get(verb_start_idx + 1, -1)
                    if verb_phrase_idx == next_phrase_idx and verb_phrase_idx >= 0:
                        verb_prep_in_phrase = True
        
        pp_tr_parts = []
        if not verb_prep_in_phrase:  # Only extract PP if verb+prep are not a phrase
            clause_node = None
            for c in tree.children:
                if c.type == "CLAUSE":
                    clause_node = c
                    break
            if clause_node and len(clause_node.children) >= 2:
                vp_node = clause_node.children[1]
                for child in vp_node.children:
                    if child.type == "PP":
                        prep_tok = None
                        np_node = None
                        if child.children:
                            if child.children[0].type == "PREP" and child.children[0].token:
                                prep_tok = child.children[0].token
                            if len(child.children) > 1 and child.children[1].type == "NP":
                                np_node = child.children[1]
                        
                        prep_tr = ""
                        if prep_tok:
                            prep_norm = prep_tok.norm
                            prep_tr = extract_from_phrase_lookup(prep_tok.value, orig_to_phrase_idx, trans_tokens, toks_values)
                            if not prep_tr:
                                prep_tr = lexicon.get(prep_norm, prep_tok.value)
                        
                        if np_node:
                            np_leaves = collect_leaf_tokens(np_node)
                            np_text = " ".join(t.value for t in np_leaves)
                            np_tr = extract_from_phrase_lookup(np_text, orig_to_phrase_idx, trans_tokens, toks_values)
                            if prep_tr and np_tr:
                                pp_tr_parts.append(f"{prep_tr} {np_tr}")
                            elif np_tr:
                                pp_tr_parts.append(np_tr)
                        elif prep_tr:
                            pp_tr_parts.append(prep_tr)
        else:
            # Verb+prep is a phrase, but we still need to extract the NP after the prep
            # Find the NP that comes after the prep in the PP
            clause_node = None
            for c in tree.children:
                if c.type == "CLAUSE":
                    clause_node = c
                    break
            if clause_node and len(clause_node.children) >= 2:
                vp_node = clause_node.children[1]
                for child in vp_node.children:
                    if child.type == "PP":
                        np_node = None
                        if child.children and len(child.children) > 1 and child.children[1].type == "NP":
                            np_node = child.children[1]
                        if np_node:
                            np_leaves = collect_leaf_tokens(np_node)
                            np_text = " ".join(t.value for t in np_leaves)
                            np_tr = extract_from_phrase_lookup(np_text, orig_to_phrase_idx, trans_tokens, toks_values)
                            if np_tr:
                                pp_tr_parts.append(np_tr)

        # Combine: Subject + Verb+Object + PP
        parts = [p for p in (subj_tr, verb_obj_tr) if p]
        parts.extend(pp_tr_parts)
        out = " ".join(parts).strip()
        out = re.sub(r"\s+([.,!?;:])", r"\1", out)
        if out and not re.search(r"[.!?]$", out):
            out += "."
        if out:
            out = out[0].upper() + out[1:]
        return out
    
    # Verb-initial without focus
    if svo.get("verb") and not svo.get("subject"):
        verb_tr = extract_from_phrase_lookup(svo["verb"], orig_to_phrase_idx, trans_tokens, toks_values)
        
        # Check if verb and object are part of the same phrase (like "mambahen salah" → "berbuat salah")
        obj_tr = ""
        if svo.get("object"):
            # Find verb position in original tokens
            verb_words = svo["verb"].lower().split()
            verb_start_idx = None
            for i, tok_val in enumerate(toks_values):
                if tok_val.lower() == verb_words[0]:
                    verb_start_idx = i
                    break
            
            if verb_start_idx is not None:
                verb_phrase_idx = orig_to_phrase_idx.get(verb_start_idx, -1)
                obj_words = svo["object"].lower().split()
                obj_start_idx = verb_start_idx + len(verb_words)
                
                if obj_start_idx < len(toks_values):
                    obj_phrase_idx = orig_to_phrase_idx.get(obj_start_idx, -1)
                    # If verb and object map to same phrase_idx, they're a single phrase
                    if verb_phrase_idx == obj_phrase_idx and verb_phrase_idx >= 0:
                        # Verb+object is already in verb_tr, don't add object separately
                        obj_tr = ""
                    else:
                        # Separate: verb + object
                        obj_tr = extract_from_phrase_lookup(svo["object"], orig_to_phrase_idx, trans_tokens, toks_values)
                else:
                    obj_tr = ""
            else:
                obj_tr = extract_from_phrase_lookup(svo["object"], orig_to_phrase_idx, trans_tokens, toks_values) if svo.get("object") else ""
        
        parts = [p for p in (verb_tr, obj_tr) if p]
        out = " ".join(parts).strip()
        out = re.sub(r"\s+([.,!?;:])", r"\1", out)
        if out and not re.search(r"[.!?]$", out):
            out += "."
        if out:
            out = out[0].upper() + out[1:]
        return out

    # Fallback: use phrase_lookup results (already computed)
    out = " ".join([t for t in trans_tokens if t])
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    if out and not re.search(r"[.!?]$", out):
        out += "."
    if out:
        out = out[0].upper() + out[1:]
    return out



# -----------------------
# Main compile pipeline
# -----------------------
def compile_text(text: str, lexicon: Dict[str, str] = LEXICON) -> Dict[str, Any]:
    """
    Run full pipeline:
      - lexing
      - pos tagging
      - parsing (with basic recovery)
      - grammar validation & error collection
      - translation
    Returns a dict with keys: tokens, parse_tree, grammar_errors, translation
    """
    # 1. lexing
    lexer = Lexer(text)
    tokens = lexer.tokenize()

    # 2. pos tagging
    tag_pos(tokens)

    # 3. parsing
    parser = Parser(tokens)
    tree = parser.parse()
    parser_errors = parser.errors

    # 4. grammar validation
    grammar_errors = validate_grammar(tokens, tree, parser_errors)

    # 5. translation (always translate)
    translation = translate_by_parse(tree, tokens, lexicon)

    return {
        "text": text,
        "tokens": tokens,
        "parse_tree": tree,
        "grammar_errors": grammar_errors,
        "translation": translation,
    }


# -----------------------
# Pretty-print helpers
# -----------------------
def pretty_print_result(res: Dict[str, Any]):
    print("INPUT:", res["text"])
    print("\nTOKENS:")
    for t in res["tokens"]:
        print(" ", t)
    print("\nPARSE TREE:")
    print(" ", res["parse_tree"])
    print("\nGRAMMAR ERRORS:")
    if res["grammar_errors"]:
        for e in res["grammar_errors"]:
            print("  -", e)
    else:
        print("  (none)")

    print("\nTRANSLATION:")
    if res["translation"]:
        print(" ", res["translation"])
    else:
        print(" (no translation produced)")

    # SVO
    svo = extract_svo(res["parse_tree"])
    print("\nEXTRACTED S/V/O:")
    print("  Subject:", svo["subject"])
    print("  Verb   :", svo["verb"])
    print("  Object :", svo["object"])
    print("\n" + "-" * 60)


# -----------------------
# GUI Application
# -----------------------
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class BatakCompilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Batak to Indonesian Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Examples
        self.examples = [
            "Horas ma di hamu!",
            "Au manangi tu pasar.",
            "Hita mangan buku.",
            "Ho marniat di rohang.",
            "Dongan manangihon au on buku tu hamu.",
            "Tu pasar au manangi.",
            "Do",
            "Manangi manangi pasar",
        ]
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Batak to Indonesian Translator",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Input
        left_panel = tk.Frame(main_frame, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Input section
        input_label = tk.Label(
            left_panel,
            text="Input (Batak Text):",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        input_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(
            left_panel,
            height=8,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white",
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        control_frame = tk.Frame(left_panel, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Action buttons
        button_frame = tk.Frame(control_frame, bg="#f0f0f0")
        button_frame.pack(side=tk.RIGHT)
        
        compile_btn = tk.Button(
            button_frame,
            text="Compile & Translate",
            command=self.compile_text,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        compile_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 10),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Example buttons
        example_label = tk.Label(
            left_panel,
            text="Examples:",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0"
        )
        example_label.pack(anchor=tk.W, pady=(10, 5))
        
        example_frame = tk.Frame(left_panel, bg="#f0f0f0")
        example_frame.pack(fill=tk.X)
        
        for i, example in enumerate(self.examples[:4]):  # Show first 4 examples
            btn = tk.Button(
                example_frame,
                text=example[:30] + "..." if len(example) > 30 else example,
                command=lambda e=example: self.load_example(e),
                bg="#ecf0f1",
                font=("Arial", 9),
                padx=5,
                pady=2,
                relief=tk.RAISED,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2, fill=tk.X, expand=True)
        
        # Right panel - Output
        right_panel = tk.Frame(main_frame, bg="#f0f0f0")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Output section with tabs
        output_label = tk.Label(
            right_panel,
            text="Output:",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        output_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Translation tab
        trans_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(trans_frame, text="Translation")
        
        self.translation_text = scrolledtext.ScrolledText(
            trans_frame,
            height=6,
            font=("Arial", 12),
            wrap=tk.WORD,
            bg="white",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.translation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tokens tab
        tokens_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tokens_frame, text="Tokens")
        
        self.tokens_text = scrolledtext.ScrolledText(
            tokens_frame,
            font=("Courier", 10),
            wrap=tk.WORD,
            bg="white",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.tokens_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Parse Tree tab
        tree_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tree_frame, text="Parse Tree")
        
        self.tree_text = scrolledtext.ScrolledText(
            tree_frame,
            font=("Courier", 10),
            wrap=tk.WORD,
            bg="white",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.tree_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Grammar Errors tab
        errors_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(errors_frame, text="Grammar Errors")
        
        self.errors_text = scrolledtext.ScrolledText(
            errors_frame,
            font=("Arial", 10),
            wrap=tk.WORD,
            bg="white",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.errors_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # SVO tab
        svo_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(svo_frame, text="S/V/O")
        
        self.svo_text = scrolledtext.ScrolledText(
            svo_frame,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="white",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.svo_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_example(self, example_text):
        """Load an example into the input field"""
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(1.0, example_text)
        
    def clear_all(self):
        """Clear all input and output"""
        self.input_text.delete(1.0, tk.END)
        self.translation_text.config(state=tk.NORMAL)
        self.translation_text.delete(1.0, tk.END)
        self.translation_text.config(state=tk.DISABLED)
        self.tokens_text.config(state=tk.NORMAL)
        self.tokens_text.delete(1.0, tk.END)
        self.tokens_text.config(state=tk.DISABLED)
        self.tree_text.config(state=tk.NORMAL)
        self.tree_text.delete(1.0, tk.END)
        self.tree_text.config(state=tk.DISABLED)
        self.errors_text.config(state=tk.NORMAL)
        self.errors_text.delete(1.0, tk.END)
        self.errors_text.config(state=tk.DISABLED)
        self.svo_text.config(state=tk.NORMAL)
        self.svo_text.delete(1.0, tk.END)
        self.svo_text.config(state=tk.DISABLED)
        
    def compile_text(self):
        """Compile and translate the input text"""
        input_text = self.input_text.get(1.0, tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some Batak text to translate.")
            return
        
        try:
            result = compile_text(input_text)
            
            # Update translation tab
            self.translation_text.config(state=tk.NORMAL)
            self.translation_text.delete(1.0, tk.END)
            if result["translation"]:
                self.translation_text.insert(1.0, result["translation"])
            else:
                self.translation_text.insert(1.0, "(No translation produced)")
            self.translation_text.config(state=tk.DISABLED)
            
            # Update tokens tab
            self.tokens_text.config(state=tk.NORMAL)
            self.tokens_text.delete(1.0, tk.END)
            for token in result["tokens"]:
                if token.type != "EOF":
                    self.tokens_text.insert(tk.END, f"{token}\n")
            self.tokens_text.config(state=tk.DISABLED)
            
            # Update parse tree tab
            self.tree_text.config(state=tk.NORMAL)
            self.tree_text.delete(1.0, tk.END)
            self.tree_text.insert(1.0, str(result["parse_tree"]))
            self.tree_text.config(state=tk.DISABLED)
            
            # Update errors tab
            self.errors_text.config(state=tk.NORMAL)
            self.errors_text.delete(1.0, tk.END)
            if result["grammar_errors"]:
                for error in result["grammar_errors"]:
                    self.errors_text.insert(tk.END, f"• {error}\n")
            else:
                self.errors_text.insert(1.0, "No grammar errors found. ✓")
            self.errors_text.config(state=tk.DISABLED)
            
            # Update SVO tab
            self.svo_text.config(state=tk.NORMAL)
            self.svo_text.delete(1.0, tk.END)
            svo = extract_svo(result["parse_tree"])
            self.svo_text.insert(1.0, f"Subject: {svo['subject'] or '(none)'}\n")
            self.svo_text.insert(tk.END, f"Verb   : {svo['verb'] or '(none)'}\n")
            self.svo_text.insert(tk.END, f"Object : {svo['object'] or '(none)'}\n")
            self.svo_text.config(state=tk.DISABLED)
            
            # Switch to translation tab
            self.notebook.select(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


def run_gui():
    """Run the GUI application"""
    if not GUI_AVAILABLE:
        print("GUI not available. tkinter is required.")
        return
    
    root = tk.Tk()
    app = BatakCompilerGUI(root)
    root.mainloop()


# -----------------------
# CLI demo / examples
# -----------------------
if __name__ == "__main__":
    import sys
    
    # Check if GUI mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        run_gui()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode
        examples = [
            "Horas ma di hamu!",
            "Au manangi tu pasar.",
            "Hita mangan buku.",
            "Ho marniat di rohang.",
            "Dongan manangihon au on buku tu hamu.",
            "Tu pasar au manangi.",  # tu before verb (misplaced)
            "Do",  # 'do' with no NP
            "Manangi manangi pasar",  # double verb
        ]

        for s in examples:
            res = compile_text(s)
            pretty_print_result(res)
    else:
        # Default: try GUI, fallback to CLI
        if GUI_AVAILABLE:
            run_gui()
        else:
            print("GUI not available. Running CLI mode...")
            examples = [
                "Horas ma di hamu!",
                "Au manangi tu pasar.",
                "Hita mangan buku.",
                "Ho marniat di rohang.",
                "Dongan manangihon au on buku tu hamu.",
                "Tu pasar au manangi.",
                "Do",
                "Manangi manangi pasar",
            ]

            for s in examples:
                res = compile_text(s)
                pretty_print_result(res)
