"""
EasyRead Pipeline Service Module
Provides graph construction, GNN-derived structural analysis, SpaCy Knowledge Graph ranking,
and graph-guided LLM prompt generation.
"""

from __future__ import annotations
import os
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx

# Lazy imports for heavy NLP packages
try:
    import spacy
except ImportError:
    spacy = None

try:
    import amrlib
except ImportError:
    amrlib = None

try:
    import penman
except ImportError:
    penman = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def build_doc_graph(amr_strings: List[str]) -> nx.DiGraph:
    """
    Merge multiple sentence-level AMR graphs into a single document graph.
    Node ID convention: "{sentence_index}::{variable}".
    Edges added:
      1. Semantic-role edges (:ARG0, :ARG1, :mod, etc.)
      2. Discourse/flow edges (NEXT_SENT between sentence roots)
      3. Coreference edges (COREF between matching base concepts across sentences)
    """
    G = nx.DiGraph()
    if not penman:
        return G

    sentence_roots = []
    concept_to_nodes = defaultdict(list)

    for sent_idx, amr_str in enumerate(amr_strings):
        if not amr_str:
            continue
        try:
            g = penman.decode(amr_str)
        except Exception:
            continue

        instances = g.instances()
        if not instances:
            continue

        sent_root = None
        for inst in instances:
            node_id = f"{sent_idx}::{inst.source}"
            base_concept = inst.target.split("-")[0]
            G.add_node(
                node_id,
                concept=inst.target,
                base_concept=base_concept,
                sent_idx=sent_idx,
            )
            concept_to_nodes[base_concept].append(node_id)
            if sent_root is None:
                sent_root = node_id

        if sent_root:
            sentence_roots.append(sent_root)

        for edge in g.edges():
            src_id = f"{sent_idx}::{edge.source}"
            tgt_id = f"{sent_idx}::{edge.target}"
            if G.has_node(src_id) and G.has_node(tgt_id):
                G.add_edge(src_id, tgt_id, etype="semantic", role=edge.role)
                G.add_edge(tgt_id, src_id, etype="semantic", role=edge.role + "_inv")

    for i in range(len(sentence_roots) - 1):
        G.add_edge(
            sentence_roots[i],
            sentence_roots[i + 1],
            etype="discourse",
            role="NEXT_SENT",
        )

    for base_concept, node_ids in concept_to_nodes.items():
        if len(node_ids) > 1:
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    ni, nj = node_ids[i], node_ids[j]
                    si = G.nodes[ni].get("sent_idx", -1)
                    sj = G.nodes[nj].get("sent_idx", -1)
                    if si != sj:
                        G.add_edge(ni, nj, etype="coref", role="COREF")
                        G.add_edge(nj, ni, etype="coref", role="COREF")

    return G


def embedding_to_structural_stats(nx_doc_graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Derive structural statistics from the document NetworkX graph.
    """
    G = nx_doc_graph
    if G.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_sentences": 1,
            "avg_nodes_per_sent": 0.0,
            "graph_density": 0.0,
            "max_path_length": 0,
            "num_coref_edges": 0,
            "top_concepts": [],
        }

    num_nodes = G.number_of_nodes()
    sent_indices = {
        data["sent_idx"] for _, data in G.nodes(data=True) if "sent_idx" in data
    }

    if sent_indices:
        num_sentences = len(sent_indices)
    else:
        discourse_edges = sum(
            1 for _, _, d in G.edges(data=True) if d.get("role") == "NEXT_SENT"
        )
        num_sentences = max(discourse_edges + 1, 1)

    avg_nodes_per_sent = round(num_nodes / num_sentences, 1)
    density = round(nx.density(G), 3)

    G_und = G.to_undirected()
    try:
        largest_cc = max(nx.connected_components(G_und), key=len)
        sub = G_und.subgraph(largest_cc)
        max_path = nx.diameter(sub)
    except Exception:
        max_path = 0

    num_coref = sum(1 for _, _, d in G.edges(data=True) if d.get("etype") == "coref")

    degree_map = {
        G.nodes[n].get("base_concept", n): G.degree(n) for n in G.nodes()
    }
    top_concepts = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "num_nodes": num_nodes,
        "num_sentences": num_sentences,
        "avg_nodes_per_sent": avg_nodes_per_sent,
        "graph_density": density,
        "max_path_length": max_path,
        "num_coref_edges": num_coref,
        "top_concepts": top_concepts,
    }


def derive_structural_guidance(stats: Dict[str, Any], easy_read_baseline: Optional[Dict[str, float]] = None) -> str:
    """
    Convert structural graph stats into natural language LLM prompt constraints.
    """
    if easy_read_baseline is None:
        easy_read_baseline = {
            "avg_nodes_per_sent": 5.5,
            "max_path_length": 4,
            "graph_density": 0.35,
        }

    lines = ["STRUCTURAL GUIDANCE (from Easy Read GNN & Graph Analysis):"]

    current = stats["avg_nodes_per_sent"]
    target = easy_read_baseline["avg_nodes_per_sent"]
    if current > target * 1.3:
        lines.append(
            f"- This document averages {current} concepts per sentence "
            f"(Easy Read target: ~{target}). "
            f"Split long sentences so each expresses only one idea."
        )
    else:
        lines.append(
            f"- Sentence complexity is close to Easy Read level "
            f"({current} concepts/sentence). Maintain this simplicity."
        )

    depth = stats["max_path_length"]
    target_depth = easy_read_baseline["max_path_length"]
    if depth > target_depth:
        lines.append(
            f"- Concept chain depth is {depth} steps "
            f"(Easy Read target: ≤{target_depth}). "
            f"Avoid nested clauses; flatten each idea to a direct statement."
        )
    else:
        lines.append(
            f"- Concept chain depth ({depth}) is within Easy Read range. "
            f"Keep sentences direct and avoid sub-clauses."
        )

    density = stats["graph_density"]
    if density < 0.1:
        lines.append(
            "- Graph is sparse. Ensure logical connectives "
            "('then', 'because', 'so') are explicit rather than implied."
        )
    elif density > 0.5:
        lines.append(
            "- High concept overlap detected. Consolidate repeated ideas."
        )

    coref = stats["num_coref_edges"]
    if coref == 0:
        lines.append(
            "- Introduce clear topic references and pronouns to improve flow."
        )
    else:
        lines.append(
            f"- {coref} cross-sentence coreference links found. "
            f"Preserve these references to maintain topic coherence."
        )

    target_sent_count = max(stats["num_sentences"], round(stats["num_nodes"] / 5.0) if stats["num_nodes"] > 0 else 1)
    lines.append(
        f"- Target output length: approximately {target_sent_count} "
        f"short sentences (one idea each, 8–12 words)."
    )

    if stats["top_concepts"]:
        vocab = ", ".join(c for c, _ in stats["top_concepts"])
        lines.append(
            f"- Central concepts to preserve: {vocab}."
        )

    return "\n".join(lines)


def extract_knowledge_graph_spacy(text: str, nlp_model=None) -> Tuple[nx.DiGraph, Dict[str, float], List[Tuple[str, float]]]:
    """
    Extract knowledge graph and rank key entity/concept nodes using SpaCy.
    """
    G = nx.DiGraph()
    if not spacy or nlp_model is None:
        # Simple fallback based on word frequencies
        words = [w.strip(".,!?;:\"'()").lower() for w in text.split() if len(w) > 3]
        counts = defaultdict(int)
        for w in words:
            counts[w] += 1
        total = max(sum(counts.values()), 1)
        ranked = sorted([(w, c / total) for w, c in counts.items()], key=lambda x: x[1], reverse=True)
        return G, dict(ranked), ranked

    doc = nlp_model(text)
    token_to_ent = {}

    for ent in doc.ents:
        G.add_node(ent.text, type=ent.label_, is_ent=True, is_stop=False)
        for token in ent:
            token_to_ent[token.i] = ent.text

    CONTENT_POS = {"NOUN", "PROPN", "VERB", "NUM", "ADJ"}
    for token in doc:
        if token.i in token_to_ent:
            continue
        node_id = token.text
        if node_id not in G:
            is_content = (
                token.pos_ in CONTENT_POS
                and not token.is_stop
                and not token.is_punct
            )
            G.add_node(
                node_id,
                type=token.pos_,
                is_ent=False,
                is_stop=(not is_content),
            )

    for token in doc:
        src_id = token_to_ent.get(token.i, token.text)
        tgt_id = token_to_ent.get(token.head.i, token.head.text)
        if src_id != tgt_id and G.has_node(src_id) and G.has_node(tgt_id):
            G.add_edge(src_id, tgt_id, relation=token.dep_)

    full_centrality = nx.degree_centrality(G)
    centrality = {
        node: score
        for node, score in full_centrality.items()
        if not G.nodes[node].get("is_stop", True)
        and G.nodes[node].get("type", "") not in {"PUNCT", "SPACE"}
    }
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    return G, centrality, ranked


def build_llm_prompt(
    original_text: str,
    nx_doc_graph: nx.DiGraph,
    knowledge_graph_ranked: List[Tuple[str, float]],
) -> str:
    """
    Build 3-part graph-guided LLM prompt.
    """
    stats = embedding_to_structural_stats(nx_doc_graph)
    structural_guidance = derive_structural_guidance(stats)

    top_concepts = knowledge_graph_ranked[:10]
    if top_concepts:
        concept_lines = "\n".join(
            f"  {i+1}. {concept} (importance: {score:.2f})"
            for i, (concept, score) in enumerate(top_concepts)
        )
    else:
        concept_lines = "  (no key concepts extracted)"

    return f"""You are simplifying a document into Easy Read format for people with intellectual disabilities.

RULES:
- Use short sentences (8–12 words each).
- Use active voice.
- Express only one idea per sentence.
- Use simple, everyday words.
- Do not use jargon or technical terms without explaining them.

KEY CONCEPTS TO PRESERVE (ranked by importance — do not omit these):
{concept_lines}

{structural_guidance}

ORIGINAL TEXT:
\"\"\"{original_text}\"\"\"

SIMPLIFIED OUTPUT:"""


def build_baseline_prompt(original_text: str) -> str:
    """
    Build standard baseline LLM prompt.
    """
    return f"""You are simplifying a document into Easy Read format for people with intellectual disabilities.

RULES:
- Use short sentences, about 8-12 words each.
- Use active voice.
- Express only one idea per sentence.
- Use simple, everyday words.
- Do not use jargon or technical terms without explaining them.
- Keep the important meaning of the original text.

ORIGINAL TEXT:
\"\"\"{original_text}\"\"\"

SIMPLIFIED OUTPUT:"""


class EasyReadPipeline:
    """
    Wrapper for loading NLP models and running simplification inference.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.stog_model = None
        self.spacy_nlp = None
        self.model_dir = model_dir

    def load_models(self) -> Dict[str, bool]:
        """Lazy load heavy models."""
        status = {"amr": False, "spacy": False}
        if amrlib and self.stog_model is None:
            try:
                if self.model_dir and os.path.exists(self.model_dir):
                    self.stog_model = amrlib.load_stog_model(model_dir=self.model_dir)
                else:
                    self.stog_model = amrlib.load_stog_model()
                status["amr"] = True
            except Exception as e:
                print(f"AMR Model load notice: {e}")

        if spacy and self.spacy_nlp is None:
            for model_name in ["en_core_web_trf", "en_core_web_sm"]:
                try:
                    self.spacy_nlp = spacy.load(model_name)
                    status["spacy"] = True
                    break
                except Exception:
                    continue

        return status

    def process_text(self, original_text: str) -> Tuple[nx.DiGraph, Dict[str, Any], List[Tuple[str, float]]]:
        """
        Run AMR parsing + GNN graph analysis + SpaCy KG ranking.
        """
        self.load_models()

        # 1. AMR Parsing & Document Graph
        amrs = []
        if self.stog_model:
            if self.spacy_nlp:
                sents = [s.text for s in self.spacy_nlp(original_text).sents]
            else:
                sents = [s.strip() for s in original_text.split(".") if s.strip()]
            try:
                amrs = self.stog_model.parse_sents(sents)
            except Exception as e:
                print(f"AMR parse error: {e}")

        doc_G = build_doc_graph(amrs)
        stats = embedding_to_structural_stats(doc_G)

        # 2. Knowledge Graph Extraction
        _, _, ranked = extract_knowledge_graph_spacy(original_text, self.spacy_nlp)

        return doc_G, stats, ranked

    def generate_prompts(self, original_text: str) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, float]]]:
        """
        Return (baseline_prompt, graph_guided_prompt, structural_stats, ranked_concepts)
        """
        doc_G, stats, ranked = self.process_text(original_text)
        baseline_prompt = build_baseline_prompt(original_text)
        graph_prompt = build_llm_prompt(original_text, doc_G, ranked)
        return baseline_prompt, graph_prompt, stats, ranked

    def run_llm(self, prompt: str, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> Optional[str]:
        """
        Invoke OpenAI API.
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key or not OpenAI:
            return None

        try:
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM execution error: {e}")
            return f"Error invoking LLM: {str(e)}"
