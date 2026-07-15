"""Meeting orchestrator.

Two modes:
  - "parallel_qa": fully scripted (the caller decides up front who speaks, in what
    order, for how many rounds). Deterministic Python control flow, not an LLM-
    moderator-driven tool loop -- this is what makes it suitable for reproducible
    experimentation.
  - "moderator": an LLM moderator dynamically decides who joins the meeting and who
    speaks next (research_agent.tools.meeting-style), but built on this project's own
    trajectory schema/logging/role-system/resume machinery rather than that module's
    bare participant-dict system.

Both build the same `steps[]` shape (see docs/schema notes for how `decided_by`
expresses scripted vs. moderator vs. hybrid without changing the output format), and
both funnel every actual participant turn through the same low-level `_execute_turn()`
so every turn -- however it was triggered -- gets identical full-fidelity trajectory
capture and per-turn resume caching.

Multi-round design (config.rounds > 1, parallel_qa only): each round after the first
carries forward, per participant, ONLY (a) the previous round's aggregated plan, (b)
that participant's OWN previous-round action log + stated changes + answer -- never raw
tool-call trajectory replayed as GeneralAgent `history=`, and never other participants'
raw answers (the aggregation step already distills those). Moderator mode uses the same
principle (agenda + moderator's rolling notes + the specific question + the
participant's own prior turn if they've spoken before) -- see trajectory.py's
summarize_turn_actions() (deterministic, from recorder.events) and round_tools.py's
submit_round_answer (schema-required changes_from_prior_round, parallel_qa only) for how
"what happened last time" is produced without raw replay.

Resume (run_meeting(config, resume=meeting_id)): parallel_qa resumes at the round
boundary (already-completed rounds skipped, per-participant turn_cache reused within
the round that was in progress). Moderator mode resumes the moderator's own raw LLM
conversation via its session_path + agent.run(history=...) -- conceptually the same
mechanism research_agent's own CLI `--resume` flag uses (state.load_session()), except
we read the raw messages straight off our own session_path file rather than calling
state.load_session(): that helper reads from research_agent's own hardcoded
SESSIONS_DIR, which our session_path= override deliberately bypasses (see agent.py's
patched save behavior), so it would always come back empty for us. Combined with our
own explicitly-checkpointed notes/agenda/roster/steps state (not derived by replaying
the moderator's tool-call history).
"""
from __future__ import annotations

import concurrent.futures
import json
from datetime import datetime
from typing import Any, Callable

from research_agent.agent import GeneralAgent

from . import roles as roles_api
from .aggregate import aggregate_responses
from .config import MeetingConfig, ParticipantConfig
from .storage import (
    load_meeting,
    load_turn_cache,
    meeting_exists,
    new_meeting_id,
    participant_workspace_dir,
    save_meeting,
    save_turn_cache,
    sessions_dir,
    shared_dir,
)
from .tools_setup import build_participant_registry
from .trajectory import LoggingLLMClient, TrajectoryUI, TurnRecorder, iso, log, now, summarize_turn_actions


def _assemble_meeting_dict(
    meeting_id: str,
    config: MeetingConfig,
    created_at,
    steps: list[dict[str, Any]],
    *,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra or {}
    final_response = steps[-1]["turns"][-1]["output"] if steps and steps[-1]["turns"] else ""

    if config.mode == "moderator":
        orchestration = {
            "type": "moderator",
            "notes": "moderator dynamically adds participants and decides who speaks next",
        }
        roster: dict[str, Any] = extra.get("roster") or {}
        participants = list(roster.values()) if roster else [
            {"name": p.name, "role": p.role, "model": p.model, "provider": p.provider}
            for p in config.participants
        ]
    else:
        orchestration = {
            "type": "scripted",
            "notes": f"{config.rounds} concurrent round(s) + LLM aggregation per round",
        }
        roster = {}
        participants = [
            {"name": p.name, "role": p.role, "model": p.model, "provider": p.provider}
            for p in config.participants
        ]

    data: dict[str, Any] = {
        "meeting_id": meeting_id,
        "mode": config.mode,
        "status": status,  # "in_progress" | "completed"
        "question": config.question,
        "created_at": iso(created_at),
        "closed_at": iso(now()) if status != "in_progress" else None,
        "orchestration": orchestration,
        "participants": participants,
        "steps": steps,
        "final_response": final_response,
    }
    if config.mode == "moderator":
        data["notes"] = extra.get("notes", "")
        data["agenda"] = extra.get("agenda", "")
        data["roster"] = roster
        data["moderator_session_id"] = extra.get("moderator_session_id")
    return data


def run_meeting(config: MeetingConfig, resume: str | None = None) -> dict[str, Any]:
    """resume: pass a previous run's meeting_id (from a "FAILED" log line / a
    runs/<id>.json with status="in_progress") to continue it instead of starting over.

    parallel_qa: already-completed rounds are skipped entirely; within the round that
    was in progress when it failed, participants that had already finished are reused
    from their per-turn cache and only the ones that hadn't finished yet are re-run.

    moderator: the moderator's own conversation is resumed via its saved session
    (history=), continuing from exactly where it left off; notes/agenda/roster/already-
    completed participant steps are restored from the checkpoint rather than replayed.
    """
    handlers: dict[str, Callable[..., list[dict[str, Any]]]] = {
        "parallel_qa": _run_parallel_qa,
        "moderator": _run_moderator,
    }
    handler = handlers.get(config.mode)
    if handler is None:
        raise ValueError(f"Unknown meeting mode: {config.mode!r}")
    if config.mode == "moderator" and config.moderator is None:
        raise ValueError("mode='moderator' requires config.moderator to be set")

    mode_kwargs: dict[str, Any] = {}

    if resume:
        if not meeting_exists(resume):
            raise FileNotFoundError(f"No checkpoint found for meeting_id={resume!r} under runs/")
        checkpoint_data = load_meeting(resume)
        if checkpoint_data.get("status") == "completed":
            raise ValueError(f"Meeting {resume} already completed -- nothing to resume")
        meeting_id = resume
        created_at = datetime.fromisoformat(checkpoint_data["created_at"])
        existing_steps = checkpoint_data.get("steps", [])

        if config.mode == "parallel_qa":
            # Steps come in (participant_step, aggregation_step) pairs per fully-
            # completed round -- a round only ever lands in the checkpoint once
            # _run_round returns both, so len(existing_steps) // 2 is exactly the
            # count of finished rounds.
            completed_rounds = len(existing_steps) // 2
            start_round = completed_rounds + 1
            aggregated_plan: str | None = None
            prior_turns: dict[str, dict[str, Any]] = {}
            if completed_rounds > 0:
                aggregated_plan = existing_steps[completed_rounds * 2 - 1]["turns"][0]["output"]
                prior_turns = {t["agent"]: t for t in existing_steps[completed_rounds * 2 - 2]["turns"]}
            mode_kwargs = dict(
                start_round=start_round,
                initial_steps=existing_steps,
                initial_aggregated_plan=aggregated_plan,
                initial_prior_turns=prior_turns,
            )
            if config.verbose:
                log("meeting", f"{meeting_id} resuming from round {start_round} ({completed_rounds} round(s) already complete)")
        else:
            resume_state = {
                "notes": checkpoint_data.get("notes", ""),
                "agenda": checkpoint_data.get("agenda", ""),
                "roster": checkpoint_data.get("roster") or {},
                "steps": existing_steps,
                "moderator_session_id": checkpoint_data.get("moderator_session_id"),
            }
            mode_kwargs = dict(resume_state=resume_state)
            if config.verbose:
                log(
                    "meeting",
                    f"{meeting_id} resuming moderator session "
                    f"{resume_state['moderator_session_id']} ({len(existing_steps)} step(s) already recorded)",
                )
    else:
        meeting_id = new_meeting_id()
        created_at = now()
        if config.verbose:
            names = ", ".join(p.name for p in config.participants) or "(none pre-seeded)"
            log("meeting", f"{meeting_id} starting ({config.mode}) -- participants: {names}")
        # Write the checkpoint file immediately, before any real work starts --
        # otherwise a failure before the first round/call ever completes leaves no
        # runs/<meeting_id>.json at all, so there'd be nothing to resume= from even
        # though per-participant turn_cache entries for whoever already finished exist.
        save_meeting(_assemble_meeting_dict(meeting_id, config, created_at, [], status="in_progress"))

    def checkpoint(steps_so_far: list[dict[str, Any]], **extra: Any) -> None:
        save_meeting(_assemble_meeting_dict(meeting_id, config, created_at, steps_so_far, status="in_progress", extra=extra))

    try:
        steps = handler(config, meeting_id, checkpoint, **mode_kwargs)
    except Exception:
        if config.verbose:
            log("meeting", f"{meeting_id} FAILED -- resume with run_meeting(config, resume={meeting_id!r})")
        raise

    if config.verbose:
        log("meeting", f"{meeting_id} done, saving to runs/{meeting_id}.json")

    final_extra: dict[str, Any] = {}
    if config.mode == "moderator":
        # The handler only returns `steps` -- notes/agenda/roster live in whatever the
        # last checkpoint() call inside it wrote, so pull them back out for the final record.
        last = load_meeting(meeting_id)
        final_extra = {
            "notes": last.get("notes", ""),
            "agenda": last.get("agenda", ""),
            "roster": last.get("roster") or {},
        }

    data = _assemble_meeting_dict(meeting_id, config, created_at, steps, status="completed", extra=final_extra)
    save_meeting(data)
    return data


def _build_round_message(
    question: str,
    round_num: int,
    aggregated_plan: str | None,
    own_prior_turn: dict[str, Any] | None,
) -> str:
    if round_num == 1 or own_prior_turn is None:
        return question

    action_log = summarize_turn_actions(own_prior_turn)
    prior_changes = own_prior_turn.get("changes_from_prior_round") or "(not recorded)"
    prior_answer = own_prior_turn.get("output") or ""

    return (
        f"=== Original meeting question ===\n{question}\n\n"
        f"=== Aggregated plan after round {round_num - 1} ===\n{aggregated_plan}\n\n"
        f"=== Your own round {round_num - 1}: actions taken ===\n{action_log}\n\n"
        f"=== Your own round {round_num - 1}: what you changed and why ===\n{prior_changes}\n\n"
        f"=== Your own round {round_num - 1}: your answer ===\n{prior_answer}\n\n"
        f"=== This round ({round_num}) ===\n"
        "Given the aggregated plan above, revise your position where needed. You MUST "
        "call submit_round_answer to finish this round -- state what you changed and "
        "why in changes_from_prior_round (write 'No changes' if you kept your prior "
        "position), and give your full current position in answer."
    )


def _execute_turn(
    participant: ParticipantConfig,
    user_message: str,
    meeting_id: str,
    round_num: int,
    verbose: bool,
    *,
    decided_by: str = "script",
    round_aware: bool = False,
) -> dict[str, Any]:
    """Shared low-level turn executor: role resolution -> GeneralAgent construction ->
    run -> record -> cache. `round_num` here is only a cache-key/log-label -- for
    moderator-triggered turns it's a monotonic call counter, not a literal round."""
    cached = load_turn_cache(meeting_id, round_num, participant.name)
    if cached is not None:
        if verbose:
            log(participant.name, f"turn {round_num}: reusing cached turn (resume)")
        return cached

    recorder = TurnRecorder(agent=participant.name, round_num=round_num, decided_by=decided_by)
    ui = TrajectoryUI(recorder, verbose=verbose)

    role: roles_api.RoleDefinition | None = None
    extra_runtime: dict[str, Any] = {}
    if participant.role_ref:
        role = roles_api.load_role(participant.role_ref)
        extra_runtime["role_memory_path"] = str(role.memory_path)
        extra_runtime["role_skills"] = role.skill_names
        system_prompt = roles_api.role_system_prompt(role)
        model = participant.model or role.model
        provider = participant.provider or role.provider
        max_iterations = participant.max_iterations if participant.max_iterations != 8 else role.max_iterations
    else:
        system_prompt = participant.build_system_prompt()
        model = participant.model
        provider = participant.provider
        max_iterations = participant.max_iterations

    meeting_shared_dir = shared_dir(meeting_id)
    system_prompt += (
        f"\n\nShared meeting files: {meeting_shared_dir} is a shared directory for this "
        "meeting. You may read and write files there to intentionally share material "
        "with other participants (e.g. a source document, or a result you want others "
        "to build on). Files in your own workspace stay private -- only put something "
        "there if you want other participants to see it."
    )

    registry = build_participant_registry(role_backed=role is not None, round_aware=round_aware)
    recorder.available_tools = sorted(registry.names)

    session_path = sessions_dir(meeting_id) / f"{participant.name}_r{round_num}.json"
    # Role-backed participants get a private, persistent workspace (roles/<name>/
    # workspace/, alongside memory.md -- carries across meetings). Ad-hoc participants
    # get a private but ephemeral one, scoped to this meeting only. Either way, no two
    # participants ever share a workspace_root -- workspace_root() is thread-local in
    # research_agent.paths, so concurrent participants setting different overrides here
    # don't race each other.
    workspace_root = role.workspace_path if role else participant_workspace_dir(meeting_id, participant.name)

    agent = GeneralAgent(
        model=model,
        provider=provider,
        max_iterations=max_iterations,
        self_review=False,
        registry=registry,
        ui=ui,
        session_path=session_path,
        sub_agent=True,
        agent_role="participant",
        extra_runtime=extra_runtime,
        workspace_root=workspace_root,
        shared_roots=[meeting_shared_dir],
    )
    agent.llm = LoggingLLMClient(agent.llm, recorder, ui)

    recorder.start_time = now()
    result = agent.run(user_message, system_prompt=system_prompt)
    recorder.end_time = now()

    recorder.output = result.get("final") or ""
    recorder.session_id = result.get("session_id")
    recorder.session_path = result.get("session_path")

    turn = recorder.to_turn_dict()
    turn["role"] = role.name if role else participant.role
    turn["role_ref"] = participant.role_ref
    # submit_round_answer (round >= 2) stashes this into the agent's own runtime dict;
    # agent.run() doesn't return runtime, but the GeneralAgent instance is still in
    # scope here so we read it straight off the instance (same coupling agent.llm
    # swapping already relies on).
    turn["changes_from_prior_round"] = getattr(agent, "_runtime", {}).get("round_changes")
    save_turn_cache(meeting_id, round_num, participant.name, turn)
    return turn


def _run_participant_turn(
    participant: ParticipantConfig,
    question: str,
    meeting_id: str,
    round_num: int,
    verbose: bool,
    aggregated_plan: str | None = None,
    own_prior_turn: dict[str, Any] | None = None,
) -> dict[str, Any]:
    round_message = _build_round_message(question, round_num, aggregated_plan, own_prior_turn)
    return _execute_turn(
        participant, round_message, meeting_id, round_num, verbose,
        decided_by="script", round_aware=round_num > 1,
    )


def _run_round(
    config: MeetingConfig,
    meeting_id: str,
    round_num: int,
    aggregated_plan: str | None,
    prior_turns: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    step_start = now()

    # Each participant is independent (own registry instance, own browser/session state
    # if it uses tools), so plain ThreadPoolExecutor threads are used — no shared
    # contextvars.Context, since a single copied Context cannot be entered by two
    # threads concurrently (research_agent.tools.meeting shares one across futures for
    # a different reason: propagating a moderator's browser session to participants,
    # which doesn't apply here).
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(config.participants))) as ex:
        futures = {
            ex.submit(
                _run_participant_turn,
                p,
                config.question,
                meeting_id,
                round_num,
                config.verbose,
                aggregated_plan,
                prior_turns.get(p.name),
            ): p.name
            for p in config.participants
        }
        turns_unordered = [f.result() for f in concurrent.futures.as_completed(futures)]

    order = {p.name: i for i, p in enumerate(config.participants)}
    turns = sorted(turns_unordered, key=lambda t: order.get(t["agent"], 999))

    step_end = now()
    participant_step = {
        "step_index": None,  # filled in by the caller once the global step index is known
        "step_start": iso(step_start),
        "step_end": iso(step_end),
        "decided_by": "script",
        "trigger_reason": f"parallel_qa round {round_num}: all participants answer/revise",
        "turns": turns,
    }

    if config.verbose:
        log("meeting", f"round {round_num}: all {len(turns)} participant(s) done, aggregating ({config.aggregation_strategy})...")

    agg_start = now()
    aggregation = aggregate_responses(
        config.question,
        turns,
        strategy=config.aggregation_strategy,
        model=config.aggregation_model,
        provider=config.aggregation_provider,
    )
    agg_end = now()

    if config.verbose:
        log("meeting", f"round {round_num}: aggregation done in {int((agg_end - agg_start).total_seconds())}s")

    aggregation_step = {
        "step_index": None,
        "step_start": iso(agg_start),
        "step_end": iso(agg_end),
        "decided_by": "config",
        "trigger_reason": f"aggregation after round {round_num}",
        "turns": [
            {
                "turn_id": f"trn_agg_{meeting_id}_r{round_num}",
                "agent": "__aggregator__",
                "decided_by": "config",
                "round": round_num,
                "strategy": aggregation["strategy"],
                "input": aggregation["prompt"],
                "output": aggregation["output"],
                "start_time": iso(agg_start),
                "end_time": iso(agg_end),
                "duration_ms": int((agg_end - agg_start).total_seconds() * 1000),
            }
        ],
    }

    return {
        "turns_by_participant": {t["agent"]: t for t in turns},
        "aggregated_plan": aggregation["output"],
        "steps": [participant_step, aggregation_step],
    }


def _run_parallel_qa(
    config: MeetingConfig,
    meeting_id: str,
    checkpoint: Callable[..., None] | None = None,
    *,
    start_round: int = 1,
    initial_steps: list[dict[str, Any]] | None = None,
    initial_aggregated_plan: str | None = None,
    initial_prior_turns: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    all_steps: list[dict[str, Any]] = list(initial_steps or [])
    aggregated_plan: str | None = initial_aggregated_plan
    prior_turns: dict[str, dict[str, Any]] = dict(initial_prior_turns or {})

    for round_num in range(start_round, max(1, config.rounds) + 1):
        round_result = _run_round(config, meeting_id, round_num, aggregated_plan, prior_turns)
        all_steps.extend(round_result["steps"])
        aggregated_plan = round_result["aggregated_plan"]
        prior_turns = round_result["turns_by_participant"]

        for i, step in enumerate(all_steps):
            step["step_index"] = i
        if checkpoint is not None:
            checkpoint(all_steps)

    if config.final_audit and aggregated_plan is not None:
        audit_step = _run_final_audit(config, meeting_id, aggregated_plan)
        all_steps.append(audit_step)
        for i, step in enumerate(all_steps):
            step["step_index"] = i
        if checkpoint is not None:
            checkpoint(all_steps)

    return all_steps


def _run_final_audit(
    config: MeetingConfig,
    meeting_id: str,
    draft_plan: str,
) -> dict[str, Any]:
    if config.verbose:
        log("meeting", "final audit: checking last aggregated plan against constraints...")

    audit_start = now()
    audit = aggregate_responses(
        config.question,
        [{"agent": "__draft_plan__", "role": "last aggregation", "output": draft_plan}],
        strategy="final_audit_llm",
        model=config.aggregation_model,
        provider=config.aggregation_provider,
    )
    audit_end = now()

    if config.verbose:
        log("meeting", f"final audit: done in {int((audit_end - audit_start).total_seconds())}s")

    return {
        "step_index": None,
        "step_start": iso(audit_start),
        "step_end": iso(audit_end),
        "decided_by": "config",
        "trigger_reason": "final constraint/evidence audit",
        "turns": [
            {
                "turn_id": f"trn_final_audit_{meeting_id}",
                "agent": "__final_auditor__",
                "decided_by": "config",
                "round": "final_audit",
                "strategy": audit["strategy"],
                "input": audit["prompt"],
                "output": audit["output"],
                "start_time": iso(audit_start),
                "end_time": iso(audit_end),
                "duration_ms": int((audit_end - audit_start).total_seconds() * 1000),
            }
        ],
    }


def _moderator_system_prompt(config: MeetingConfig) -> str:
    return (
        f"You are {config.moderator.name}, the moderator of a multi-agent meeting.\n\n"
        f"Your job is to reach a good outcome for this question:\n{config.question}\n\n"
        "You do not answer the question yourself -- you orchestrate participants who do. "
        "Use meeting_add_participant to bring people into the meeting: either an existing "
        "role via role_ref (use role_list/role_load to see what's available in the role "
        "library), or an ad-hoc participant defined inline with a persona/purpose. Use "
        "meeting_set_agenda and meeting_add_notes to maintain shared context -- every "
        "participant you call on will see the current agenda and notes, so keep notes "
        "updated with whatever the next speaker needs to know. Use meeting_call_on to ask "
        "ONE participant ONE question and read their answer; you decide who speaks and in "
        "what order, one at a time, and you may call on the same participant more than "
        "once. When you are satisfied, call meeting_conclude with a synthesized final "
        "answer -- there is no separate aggregation step in this mode, your conclusion "
        "IS the final answer."
    )


def _run_moderator(
    config: MeetingConfig,
    meeting_id: str,
    checkpoint: Callable[..., None] | None = None,
    *,
    resume_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from .moderator_tools import build_moderator_tools  # deferred: avoids a circular import

    def _checkpoint(steps_so_far: list[dict[str, Any]], **extra: Any) -> None:
        if checkpoint is not None:
            checkpoint(steps_so_far, **extra)

    registry, state = build_moderator_tools(meeting_id, config, _checkpoint)

    if resume_state:
        state["notes"] = resume_state.get("notes", "")
        state["agenda"] = resume_state.get("agenda", "")
        state["roster"] = dict(resume_state.get("roster") or {})
        state["steps"] = list(resume_state.get("steps") or [])
        state["prior_turns"] = {
            t["agent"]: t
            for step in state["steps"]
            for t in step["turns"]
            if step.get("decided_by") == "moderator"
        }
        state["call_counter"] = len(state["steps"])
    else:
        for p in config.participants:
            state["roster"][p.name] = {
                "name": p.name,
                "role_ref": p.role_ref,
                "role": p.role,
                "skills": p.skills,
                "system_prompt": p.system_prompt,
                "model": p.model,
                "provider": p.provider,
                "max_iterations": p.max_iterations,
            }

    session_path = sessions_dir(meeting_id) / "moderator.json"
    recorder = TurnRecorder(agent=config.moderator.name, round_num=0, decided_by="moderator")
    ui = TrajectoryUI(recorder, verbose=config.verbose)

    role: roles_api.RoleDefinition | None = None
    extra_runtime: dict[str, Any] = {}
    if config.moderator.role_ref:
        role = roles_api.load_role(config.moderator.role_ref)
        extra_runtime["role_memory_path"] = str(role.memory_path)
        extra_runtime["role_skills"] = role.skill_names
        system_prompt = roles_api.role_system_prompt(role)
        model = config.moderator.model or role.model
        provider = config.moderator.provider or role.provider
    else:
        system_prompt = config.moderator.system_prompt or _moderator_system_prompt(config)
        model = config.moderator.model
        provider = config.moderator.provider

    history = None
    session_id = resume_state.get("moderator_session_id") if resume_state else None
    prompt = config.question
    if session_id:
        # Read straight off our own session_path rather than research_agent.state.
        # load_session(): that helper reads from the library's own hardcoded
        # SESSIONS_DIR, which our session_path= override deliberately bypasses (agent.py
        # skips its default SESSIONS_DIR write whenever a caller supplies session_path),
        # so it would always come back empty here.
        if session_path.exists():
            history = json.loads(session_path.read_text(encoding="utf-8"))
        prompt = "Continue the meeting from where you left off."

    agent = GeneralAgent(
        model=model,
        provider=provider,
        max_iterations=config.moderator.max_iterations,
        self_review=False,
        registry=registry,
        ui=ui,
        session_path=session_path,
        session_id=session_id,
        sub_agent=True,
        agent_role="moderator",
        extra_runtime=extra_runtime,
    )
    agent.llm = LoggingLLMClient(agent.llm, recorder, ui)

    # session_path persists progressively on every iteration regardless of whether
    # agent.run() below ever returns, but the session_id itself must land in `state`
    # (read by every checkpoint call, including ones triggered by moderator_tools.py's
    # closures during agent.run()) and be checkpointed NOW, before the call, so a crash
    # on the very first iteration still leaves a session_id a later resume can find and
    # load_session() from.
    state["moderator_session_id"] = agent.session_id
    _checkpoint(state["steps"], notes=state["notes"], agenda=state["agenda"],
                roster=state["roster"], moderator_session_id=agent.session_id)

    recorder.start_time = now()
    result = agent.run(prompt, history=history, system_prompt=system_prompt)
    recorder.end_time = now()

    recorder.output = result.get("final") or state.get("conclusion") or ""
    recorder.session_id = result.get("session_id")
    recorder.session_path = result.get("session_path")

    moderator_turn = recorder.to_turn_dict()
    moderator_turn["role"] = role.name if role else config.moderator.name
    moderator_turn["role_ref"] = config.moderator.role_ref

    all_steps = state["steps"] + [
        {
            "step_index": None,
            "step_start": iso(recorder.start_time),
            "step_end": iso(recorder.end_time),
            "decided_by": "moderator",
            "trigger_reason": "moderator session (full decision trajectory)",
            "turns": [moderator_turn],
        }
    ]
    for i, s in enumerate(all_steps):
        s["step_index"] = i

    _checkpoint(all_steps, notes=state["notes"], agenda=state["agenda"], roster=state["roster"],
                moderator_session_id=agent.session_id)
    return all_steps
