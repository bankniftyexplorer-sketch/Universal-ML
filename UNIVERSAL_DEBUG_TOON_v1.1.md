# UNIVERSAL DEBUG PROTOCOL · TOON/v1.1
> Token-Oriented Object Notation · Compact Execution Spec for AI Coding Agents
> Domain: ANY · Reusable · Drop-in · Read fully before first file open.
> v1.1 change: §P1.5 ACQUIRE_LOGS added — runtime evidence cross-reference before discovery.

---

## §META

```
PROTOCOL {
  purpose   : "Root-cause bug discovery + permanent architectural fixes"
  scope     : ANY_PROJECT [ lang:*, domain:*, size:small→medium_complex ]
  phases    : [ ACQUIRE → ACQUIRE_LOGS → DISCOVER → REPORT → PLAN → IMPLEMENT ]
  sequence  : STRICT_SEQUENTIAL | NEVER_PARALLEL | NEVER_REORDERED
  output    : BugReport → FixPlan → VerifiedFixes
}

AXIOMS {
  A1 : comprehension BEFORE action, always
  A2 : symptom ≠ bug  →  find_root_cause, never suppress_symptom
  A3 : fix.scope = architectural | REJECT if fix.scope = patch
  A4 : fix ∈ dependency_graph  →  map_all_dependents BEFORE edit
  A5 : correctness DEFINED_BY domain_spec, NOT BY code
  A6 : token_discipline = analytical_discipline
       → NO speculative_output
       → NO re-read of comprehended files
       → NO code_execution to answer questions solvable analytically
  A7 : runtime_evidence SUPERSEDES static_analysis
       → log-confirmed bug = CONFIRMED severity (never downgrade)
       → log-absent bug over long window = LATENT (may downgrade)
       → log event with no code explanation = new_bug_candidate
}
```

---

## §P1 · PHASE: ACQUIRE

```
ENTRY  : task assigned
EXIT   : full_system_model ∈ working_memory (no output required)
OUTPUT : ∅  ← zero written deliverables this phase

STEPS [
  S1.1: MAP_STRUCTURE {
    before  : open ANY file content
    build   : inventory[ file → { role, lang, loc≈, position∈dep_graph } ]
    role_∈  : [ ENTRY | MODEL | DOMAIN_LOGIC | UTIL | TEST | CONFIG | ADAPTER ]
    position: leaf (no deps) → root (all deps)  ← read order = leaf→root
  }

  S1.2: READ_DOMAIN_FIRST {
    targets : [ README, spec_docs, ADRs, API_docs, external_standards,
                business_rules, inline_domain_comments ]
    WHY     : domain_spec = correctness_oracle
              without_it → false_pos (flag correct code) +
                           false_neg (accept wrong code)
    ACTION  : if external_standard referenced → FETCH now, extract ALL
               relevant rules in ONE pass → compact rule_set
  }

  S1.3: READ_FILES [ order: leaf→root ] {
    per_file, build+hold internally {
      INVARIANTS  : conditions that must always be true
      ASSUMPTIONS : caller/env dependencies not verified by code
      CONTRACTS   : fn(valid_input)→? | fn(invalid_input)→?
      DATA_FLOW   : raw_input → transform_chain → output | mutations
      SEAMS       : all external touches [ file|net|clock|rng|api|db ]
    }
  }

  S1.4: VERIFY_EXTERNAL_DEPS {
    ∀ external_dep    → { assumed_behavior vs actual_behavior → gap? }
    ∀ embedded_static → { source_authority, last_verified, current_accurate? }
    static_data ⚡HIGH_YIELD ← changes in world, code stays frozen
  }
]
```

---

## §P1.5 · PHASE: ACQUIRE_LOGS

```
ENTRY  : P1.EXIT confirmed — full code model in working memory
EXIT   : runtime_evidence_map built | severity_adjustments queued
OUTPUT : ∅  ← zero written deliverables this phase
         internal bug candidates from log signals → feed P2

NOTE   : this phase is CONDITIONAL.
         if no log files are attached → skip to P2, note absence in report header.
         if log files are attached → MANDATORY before P2.
         partial logs (some channels missing) → process what exists,
         note missing channels in report header.

STEPS [

  S1.5.1: IDENTIFY_LOG_FILES {
    standard_targets [
      logs/engine.txt      ← ENGINE channel
      logs/timeline.txt    ← TIMELINE channel
      logs/ingestion.txt   ← INGESTION channel
      logs/state.txt       ← STATE channel
      logs/alerts.txt      ← ALERT channel
      SPECTRE_ENGINE_AUDIT_REPORT_*.md  ← latest file only if multiple exist
      runtime_audit/*.log  ← any supplementary module-level log files
    ]
    non_standard : accept ANY structured log file attached by human
    per_file : record { channel, line_count, time_window_start, time_window_end }
  }

  S1.5.2: EXTRACT_SIGNALS {
    per_log_file, extract and hold internally:

    ERROR_LINES {
      predicate : line contains [ ERROR | FAIL | CRITICAL | exception |
                                  traceback | AssertionError | RuntimeError ]
      capture   : full line + timestamp + surrounding 2 lines (context)
    }

    WARN_LINES {
      predicate : line contains [ WARN | STALE | DROP | TIMEOUT | MISS |
                                  PARTIAL | CORRUPT | OVERFLOW | SHORTFALL ]
      capture   : full line + timestamp
    }

    FREQ_MAP {
      key   : event_name (token between first and second pipe in structured lines)
      value : { count, first_seen, last_seen, rate_per_minute }
      build : for ALL event_names, not just errors
    }

    ANOMALY_SIGNALS {
      events firing at > 10× their expected_rate_per_minute
      gaps in timeline > 2× median inter-event interval
      event sequences that violate known ordering invariants
        (e.g.: HISTORY_RESET without preceding SYMBOL_SWITCH)
      numeric field values outside plausible domain
        (e.g.: latency > 5000ms | iv > 3.0 | spot = 0 | lot_size = 0)
    }
  }

  S1.5.3: BUILD_EVIDENCE_MAP {
    ∀ event_name in FREQ_MAP → {
      source_fn     : which fn in which file emits this event (from P1 knowledge)
      classification: NOMINAL | DEGRADED | ERROR | UNKNOWN
      NOMINAL       : expected in healthy run at observed rate
      DEGRADED      : expected event but rate or value indicates problem
      ERROR         : should never appear in healthy run
      UNKNOWN       : event has no matching emit site in codebase → new_bug_candidate
    }
  }

  S1.5.4: CROSS_REFERENCE_PREP {
    output (internal only): evidence_queue = [
      {
        event_name    : string
        classification: NOMINAL | DEGRADED | ERROR | UNKNOWN
        log_channel   : string
        sample_line   : exact log line (most recent occurrence)
        count         : int
        rate          : float (per minute)
        first_seen    : timestamp
        last_seen     : timestamp
        linked_bug_id : null  ← filled during P3 when bug is written
      }
    ]

    new_bug_signals = [
      items from evidence_queue where classification = UNKNOWN
      + items where classification = ERROR with no obvious P2 code match
      + anomaly_signals from S1.5.2
    ]
    ← these are investigated as additional candidates in P2
  }
]
```

---

## §P2 · PHASE: DISCOVER

```
ENTRY  : P1 + P1.5 both EXIT confirmed
EXIT   : all 6 layers applied to ALL files (not just suspicious ones)
         + all new_bug_signals from P1.5 investigated
OUTPUT : internal bug_list → feeds P3

LAYERS [

  L1: STRUCTURAL {
    CHECK [
      duplicate_symbols    → silent override risk
      misplaced_imports    → in loops | hot paths | conditional namespaces
      hardcoded_constants  → encoding domain rules invisibly
      silent_fallbacks     → degraded behavior without log/raise
    ]
    PER_BUG: note { file, location, active|latent }
  }

  L2: CONTRACT_INTERFACES {
    ∀ cross-module call → verify match on ALL dims {
      param: [ units, scale, sign_convention, type, valid_range ]
      return: [ semantics, units, error_signals ]
      errors: [ exception|sentinel|empty ] handled at callsite?
    }
    ⚡WATCH: ambiguous param names
      t       → elapsed_time | remaining_time?  ← opposite signs
      size    → bytes | elements?
      price   → bid | ask | mid | last?
      value   → absolute | normalized?
    mismatch_on_ANY_dim = contract_violation_bug
  }

  L3: LOGIC_ALGORITHM {
    METHOD {
      1. derive expected_output FROM domain_spec (NOT from code)
      2. read code → determine actual_output
      3. compare term-by-term → identify divergence
    }
    CHECK_ALL {
      branches   : covers ALL cases? | correct case→branch mapping?
                   boundary_of_condition → correct_branch?
      iteration  : termination guaranteed? | base_case correct?
                   step N→N+1 preserves correctness?
      numerics   : stable across full production input range?
                   division → denom=0 guard?
                   log/sqrt → domain guard?
                   accumulation → float_error magnitude?
    }
    ⊗ NEVER judge correctness by "looks plausible"
  }

  L4: DATA_INTEGRITY {
    ∀ static_data_structure → verify {
      COMPLETE   : all required keys present? → no missing_key silent_wrong_default
      ACCURATE   : values match authority_source?
      CURRENT    : accurate today (not just at write-time)?
      LOCAL      : related data co-located? → scattered = sync_risk
    }
    ∀ dynamic_data FROM untrusted_source → validated BEFORE use in computation?
  }

  L5: DEFENSIVE_GUARDRAILS {
    ∀ fn → ask per_param { zero | negative | empty | null | boundary | wrong_type }
      → detected+informative_error? OR silent_wrong_propagation?

    FIND [
      loops    : exit guaranteed without relying on external state?
      recursion: depth_limit enforced by CODE (not runtime stack)?
      except   : broad_catch discarding error_info?
      platform : environment-specific ops checked before use?
    ]

    ERROR_SPEC = { fn_name, param_name, value_received, constraint_violated }
    ← minimum required content for any raised error
  }

  L6: ARCHITECTURAL {
    ASK {
      shared_structure serving 2 semantically_distinct domains?
      constant encoding domain_rule with NO source documentation?
      silent_degradation to weaker_mode without logging?
      missing_abstraction: concept recurring across files, not encapsulated?
      coupling: modules forced to change together when only one should?
    }
    TYPE: bugs that pass today, FAIL after next routine system change
  }

  L7: LOG_SIGNAL_INVESTIGATION {
    ← this layer exists ONLY when P1.5 ran
    ← investigates new_bug_signals from P1.5.4

    ∀ item in new_bug_signals → apply full L1-L6 analysis
    ∀ UNKNOWN event → find missing emit_site or missing handler
    ∀ DEGRADED at high rate → find rate-cause in code
    ← findings feed into normal bug_list alongside L1-L6 findings
  }
]
```

---

## §P3 · PHASE: REPORT

```
ENTRY  : P2.EXIT confirmed (all layers × all files)
EXIT   : bug_report complete → reviewed+approved by human
OUTPUT : BugReport {
           header_block
           + log_coverage_summary  ← NEW in v1.1
           + summary_table[severity_sort]
           + entries[dep_order]
         }

─────────────────────────────────────────────────────────
HEADER_BLOCK {
  files_audited    : list of all code files read in P1
  log_files_read   : list of log files read in P1.5 | "NONE — P1.5 skipped"
  log_window       : { start_timestamp, end_timestamp, duration }
  prior_fixes      : list of prior-session fixes verified intact | none
}
─────────────────────────────────────────────────────────

LOG_COVERAGE_SUMMARY {  ← NEW in v1.1
  ← appears immediately after header, before summary table
  ← omitted only if P1.5 was skipped

  FREQ_TABLE {
    columns : [ event_name | channel | count | rate/min | classification ]
    rows    : ALL events observed (sorted by count desc)
    ← human can immediately see what the engine was actually doing
  }

  TOP_ERRORS    : up to 10 most frequent ERROR/WARN events with sample lines
  ANOMALIES     : all items from P1.5 ANOMALY_SIGNALS
  NEW_SIGNALS   : events with classification=UNKNOWN (no code match found)
}

─────────────────────────────────────────────────────────
ENTRY_SCHEMA {
  BUG_ID    : "{FILE_ABBR}-{N}"  e.g. AUTH-2 | DB-5 | UTIL-1
  SEVERITY  : CRITICAL | HIGH | MEDIUM | LOW   ← see rules below
  TYPE      : T1..T7  ← see §TAXONOMY
  LOCATION  : file + fn_name + line_range  ← NOT just filename
  ROOT_CAUSE: 1-3 sentences: WHY does this exist (gap|missing_case|wrong_rule)
              ← if unclear → you don't understand the bug yet → keep analyzing
  SYMPTOM   : observable wrong behavior when triggered
  DERIVATION: FOR bugs involving formula|algorithm|domain_rule ONLY:
              { domain_correct_result, code_actual_result, divergence }
              ← MOST IMPORTANT FIELD · proof bug exists · spec for correct behavior

  RUNTIME_EVIDENCE {   ← NEW in v1.1
    ← MANDATORY field when P1.5 ran
    ← format varies by result:

    CONFIRMED {
      observed    : YES
      log_channel : string
      sample_line : exact log line proving the bug fired
      count       : int  ← occurrences in log window
      rate        : float (per minute)
      first_seen  : timestamp
      last_seen   : timestamp
      severity_note: "confirmed firing → severity UPGRADED from X to Y" | none
    }

    NOT_OBSERVED {
      observed    : NO
      log_window  : duration of log data examined
      note        : "no log evidence in {duration} window — latent defect"
      severity_note: "no observed firing → severity REMAINS X | downgraded from X to Y"
    }

    LOG_ONLY {
      ← used for bugs discovered FROM logs (new_bug_signals) not from code review
      observed    : LOG_TRIGGERED
      log_channel : string
      sample_line : exact triggering log line
      count       : int
      pattern     : description of anomalous pattern that surfaced the bug
    }
  }

  DEPS      : bugs requiring fix BEFORE this | downstream files affected
  SOLUTION  : logical spec of correct behavior {
                main_case + all_edge_cases + all_boundary_cases + all_error_cases
              } ← precise enough for independent implementation
  VERIFY    : deterministic test { input → known_correct_output }
              ← mathematical_identity | conservation_law | analytical_value
              ← "should work now" = ⊗ NOT ACCEPTABLE
}

SEVERITY_RULES {
  CRITICAL : wrong result on NORMAL execution path | fires every run
  HIGH     : wrong result on COMMON condition | at least monthly in prod
  MEDIUM   : wrong result on RARE but DEFINED condition | latent defect
  LOW      : no wrong result ever | maintenance_risk | future_dev confusion

  SEVERITY_ADJUSTMENT_RULES {  ← NEW in v1.1
    log_confirmed + was_MEDIUM → upgrade to HIGH
    log_confirmed + was_LOW    → upgrade to MEDIUM
    not_observed in window > 1hr + was_HIGH → may downgrade to MEDIUM (note only)
    not_observed in window > 1hr + was_CRITICAL → DO NOT downgrade
      WHY: CRITICAL bugs may not have fired yet but will on next trigger
  }
}

ORDER_RULE : entries ordered by fix_execution_dependency (foundation→top)
             summary_table ordered by severity (critical→low)
```

---

## §P4 · PHASE: PLAN

```
ENTRY  : BugReport human-approved
EXIT   : all fixes specified | dep_graph built | sequence determined
OUTPUT : ImplementationPlan { dep_graph + fix_specs[topo_order] }

STEPS [
  S4.1: DEP_GRAPH {
    nodes : each fix
    edge A→B EXISTS when ANY of {
      A modifies fn that B also modifies
      A changes data_structure B depends on
      A changes fn_signature that B must call correctly
      A behavior change affects B's verify_test known_output
    }
    → topo_sort → implementation_sequence
    same_topo_level → any order | all must complete before next level
  }

  S4.2: FIX_SPECS {
    ∀ fix → answer ALL without ambiguity {
      new_logical_behavior: complete description
      param_changes       : added | removed | renamed | type_changed
      case_coverage       : ALL branches + boundary conditions + error conditions
      return_values       : per valid_case
      raises              : exception_class + message_content per invalid_input
      formula_fix ⊕       : correct formula in full notation, all terms named
    }
    ← "fix the formula" = ⊗ NOT A SPEC
  }

  S4.3: PROPAGATION_MAP {
    ∀ fix changing { signature | return_type | data_shape | contract }
    → enumerate EVERY call_site { file_name + fn_name }
    ← "update all callers" = ⊗ NOT ACCEPTABLE → enumerate explicitly
  }

  S4.4: VERIFY_FIRST {
    write verify_test BEFORE writing fix_spec
    WHY: forces concrete definition of "correct" before defining "how"
         unspecifiable_test → incomplete understanding of correctness
  }
]
```

---

## §P5 · PHASE: IMPLEMENT

```
ENTRY  : P4 complete | sequence locked
EXIT   : all fixes done | all tests pass | checklist §P8 fully confirmed

RULES [
  R5.1: ONE_FIX_PER_UNIT {
    atomic unit = one bug fix
    after each fix → run its verify_test BEFORE next fix
    fix passes own test BUT breaks prior test → missed_dep → STOP
    → re-examine dep_graph → resolve → continue
  }

  R5.2: TEST_BEFORE_FIX {
    sequence: write_test → confirm_fails_on_buggy_code → implement_fix
              → confirm_passes
    WHY: proves test is sensitive to bug AND fix is complete
    SKIP = fixes whose correctness rests on assertion, not evidence
  }

  R5.3: FIX_COMPLETENESS {
    fix.complete = implementation_changed AND
                  all_propagation_map_items_updated AND
                  all_downstream_tests_pass
    partial_fix (fn correct, callers wrong) = bug moved, not fixed
  }

  R5.4: NO_OPPORTUNISTIC_REFACTOR {
    fix.scope = plan_spec EXACTLY | nothing more
    rename|restructure|reorganize = separate_task, separate_commit
    ⚡bundling obscures signal + introduces unintended_behavior_risk
  }

  R5.5: DOC_UPDATE_MANDATORY {
    ∀ fixed fn → update { docstring | inline_comments | docs }
    old_behavior docstring AFTER fix = bug (will cause re-introduction)
    no_docstring → ADD one containing {
      purpose, param_semantics+valid_range, return_value, raises
    }
  }
]
```

---

## §QUOTA · TOKEN EFFICIENCY RULES

```
READ_ONCE    : each file read ONCE in P1 | need_to_reread = P1_was_insufficient
               → slow down, read more carefully | not a normal workflow step

BATCH_READS  : ALL code files read BEFORE any analysis output
               ALL log files read in P1.5 BEFORE P2 begins
               interleaving = fragmented_comprehension + redundant_reconstruction

NO_SPEC_OUTPUT : reason internally | output only complete verified thoughts
                 partial_analysis | exploratory_hypothesis = ⊗ FORBIDDEN

NO_EXEC_TO_REASON : code_execution = verify_implemented_fix_matches_analytic_result
                    code_execution ≠ substitute for reasoning about behavior
                    need_to_run_to_understand = analytical_comprehension incomplete
                    → build comprehension through reasoning first

EXTERNAL_FETCH : fetch once | extract ALL relevant rules in ONE pass
                 output = compact_precise_ruleset (not verbatim reproduction)
                 re-fetch = ⊗ FORBIDDEN | plan extraction to be complete

NO_RECHECK : once dep fully characterized { behavior | returns | assumes }
             → done | do not revisit without specific concrete new evidence

LOG_READ_ONCE : log files read once in P1.5 | internal FREQ_MAP built in one pass
                do NOT re-read log files during P2, P3, or later phases
                all log knowledge = what was extracted in P1.5
```

---

## §TAXONOMY · BUG TYPES

```
T1 · LOGICAL_INCORRECTNESS {
  def : algorithm|formula|decision_logic → result ≠ domain_authoritative_result
  inc : missing_term | wrong_sign | wrong_operator | wrong_branch_case
        wrong_loop_termination | wrong_base_case
  key : domain_authoritative_correct_result EXISTS and code DIVERGES from it
}

T2 · SEMANTIC_CONTRACT_VIOLATION {
  def : caller↔callee mismatch on param/return SEMANTICS
        each side internally consistent | collectively incompatible
  inc : unit_mismatch | scale_mismatch | sign_convention_mismatch
        return_semantic_misinterpretation
  key : both sides correct in isolation | interface between them is wrong
}

T3 · STRUCTURAL_DEFECT {
  def : syntactically valid | correct under typical conditions
        structure creates invisible_deps | override_risk | maintenance_trap
  inc : duplicate_symbol_defs | misplaced_imports | undocumented_fallback
        missing_separation_of_concerns
  key : bug lives in CODE STRUCTURE, not current behavioral output
}

T4 · DATA_STALENESS_OR_INCOMPLETENESS {
  def : embedded static_data incorrect | authority_source changed
        or never complete over full input domain
  inc : expired_lookup_tables | missing_entries | stale_constants
  key : bug lives in DATA VALUES, not logic
}

T5 · MISSING_DEFENSIVE_GUARDRAILS {
  def : correct for valid clean inputs
        fails silent|confusing|catastrophic for invalid inputs
  inc : no_input_validation | unbounded_loops | broad_except_discard
        missing_zero|null|empty|boundary handling
  key : manifests ONLY under conditions code wasn't written to handle
}

T6 · ARCHITECTURAL_ANTIPATTERN {
  def : locally defensible | structurally wrong at system level
        guarantees incorrect behavior after routine future changes
  inc : shared_state across distinct domains | hardcoded_undocumented_rules
        silent_degradation | missing_abstraction | wrong_coupling
  key : passes today | FAILS after next routine system evolution
}

T7 · DEAD_OR_UNREACHABLE_CODE {
  def : syntactically present | structurally correct | never_executed
  risk: misleads devs → believe feature implemented | masks missing integration
  key : NO execution path reaches this code under any reachable input
}
```

---

## §ATTACH · WHAT TO PROVIDE EACH SESSION

```
MANDATORY [
  all source files (leaf→root order if known)
  last_session_bug_fix_report.txt  ← so prior fixes are verified, not re-found
]

STRONGLY_RECOMMENDED [
  spectre/logs/engine.txt      ← or equivalent ENGINE channel log
  spectre/logs/timeline.txt
  spectre/logs/ingestion.txt
  spectre/logs/state.txt
  spectre/logs/alerts.txt
  SPECTRE_ENGINE_AUDIT_REPORT_*.md  ← latest stability audit report
]

OPTIONAL [
  runtime_audit/*.log          ← any module-level supplementary logs
  test output (pytest -v)      ← shows which tests pass/fail before audit
]

WITHOUT_LOGS {
  audit proceeds via P1 → P2 → P3 (P1.5 skipped)
  report header will state: "P1.5 skipped — no log files provided"
  all RUNTIME_EVIDENCE fields will read: "not_observed — no logs provided"
  severity = code-analysis-only (no confirmation possible)
}
```

---

## §P8 · COMPLETION CHECKLIST

```
TASK.complete = TRUE  iff  ALL of {

  [ ] all project files read ≥ once during P1
  [ ] P1.5 executed if log files were provided | skipped and noted if not
  [ ] LOG_COVERAGE_SUMMARY present in report if P1.5 ran
  [ ] all 6 layers (+ L7 if P1.5 ran) applied to ALL files
  [ ] every bug has complete entry (all fields present, none omitted)
  [ ] every bug has RUNTIME_EVIDENCE field populated
  [ ] severity adjustments applied per §P3 SEVERITY_ADJUSTMENT_RULES
  [ ] bug_report human-reviewed BEFORE implementation began
  [ ] dep_graph explicitly constructed | implementation follows topo_order
  [ ] every fix has verify_test written BEFORE fix | passes AFTER fix
  [ ] every call_site in propagation_map updated for every fix
  [ ] every docstring|comment of fixed code updated to new behavior
  [ ] no fix = symptom_suppression | every fix addresses root_cause
  [ ] all embedded static_data verified vs authority_source AS OF TODAY
  [ ] full test suite passes | no previously-passing test now fails

}

ANY unconfirmed item → task NOT complete → return to relevant phase
```

---

```
