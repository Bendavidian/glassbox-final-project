"""GB-67 step 2: the Co-Pilot approval dialog.

**Run in Streamlit's own runtime through ``AppTest``, not against a stand-in.** ``AppTest``
executes the dialog's body and records it as a ``dialog`` block under the ``event`` root,
so a test can ask whether a dialog was opened, what is in it, and which buttons exist -
the questions these guards are about. The tree is read through ``AppTest._tree``, which
is private; the version is pinned (``streamlit==1.61.1``), and a change there fails these
tests rather than passing them.

Page-level rendering - fills, radius, focus - is in ``test_approval_fills_on_the_page.py``,
which drives a real browser.
"""

from __future__ import annotations

import ast
import html
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from glassbox import records
from glassbox.dashboard import app

#: Every value deliberately unlike what a formatter or a recomputation would produce:
#: nine decimals where the panel prints four, and a notional that is **not** shares x
#: price, so a card that multiplied instead of reading would show a different number.
ENTRY = {
    "decision_id": "20260930-MODL",
    "as_of": "2026-09-30T00:00:00+00:00",
    "symbol": "MODL",
    "shares": 0.123456789,
    "price": 987.654321,
    "notional": 121.93,
    "stop_loss": 950.5,
    "take_profit": 1040.25,
    "narrative": "MODL on 2026-09-29: the model predicts a 1.23% rise.  Kept verbatim.",
    "provenance": records.LIVE,
    "record": {"signal": {"action": "enter_long"}},
}

#: AppTest's default of 3 s covers a script, not the first import of the dashboard.
TIMEOUT = 60


def _modal_script(root: str, loop_present: bool) -> None:
    from pathlib import Path

    import streamlit as st

    from glassbox.config.loader import load_config
    from glassbox.dashboard import app

    cfg = load_config()
    verdict = app.liveness(
        lock=app.LOCK_ALIVE if loop_present else app.LOCK_NONE,
        band_fires=True,
        in_session=True,
        loop_age=0.0 if loop_present else float("inf"),
        broker_age=0.0,
        heartbeat=cfg.live.heartbeat_seconds,
    )
    app.approval_modal(Path(root), cfg, verdict, st)


def _panel_script(root: str) -> None:
    from pathlib import Path

    import streamlit as st

    from glassbox.config.loader import load_config
    from glassbox.dashboard import app

    cfg = load_config()
    verdict = app.liveness(
        lock=app.LOCK_NONE,
        band_fires=True,
        in_session=True,
        loop_age=float("inf"),
        broker_age=0.0,
        heartbeat=cfg.live.heartbeat_seconds,
    )
    app._copilot_panel(Path(root), cfg, verdict, st)


def _run(script, *args) -> AppTest:
    return AppTest.from_function(script, args=args, default_timeout=TIMEOUT).run()


def _dialogs(at: AppTest) -> list:
    """Every dialog block the run produced, found by walking the element tree."""
    found = []

    def walk(node) -> None:
        if getattr(node, "type", None) == "dialog":
            found.append(node)
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at._tree)
    return found


def _queue(root: Path, *entries: dict) -> None:
    for entry in entries:
        records.save_pending(root, entry)


# ── guard 5: a record that is not an entry can be neither approved nor rejected ─────


NOT_ENTRIES = {
    "an exit": {**ENTRY, "decision_id": "20260930-EXIT", "kind": "exit"},
    "an action other than enter_long": {
        **ENTRY,
        "decision_id": "20260930-ACTN",
        "record": {"signal": {"action": "exit"}},
    },
    "a record that is not a mapping": {
        **ENTRY,
        "decision_id": "20260930-BLOB",
        "record": "not decoded here",
    },
    "a record with no signal": {**ENTRY, "decision_id": "20260930-NSIG", "record": {}},
    **{
        f"no {key}": {k: v for k, v in ENTRY.items() if k != key}
        for key in app.ENTRY_KEYS
        if key != "decision_id"
    },
}


def test_only_a_positively_identified_entry_is_approvable() -> None:
    """**Guard 5, the predicate.** Every required key, no ``kind`` other than
    ``"entry"``, and ``record.signal.action == "enter_long"`` read as a key - the record is
    never decoded. Anything short of that is refused, including a record missing its id.
    """
    assert app.is_approvable_entry(ENTRY)
    assert app.is_approvable_entry({**ENTRY, "kind": "entry"})
    assert not app.is_approvable_entry(
        {k: v for k, v in ENTRY.items() if k != "decision_id"}
    )
    wrongly_approvable = [
        name for name, entry in NOT_ENTRIES.items() if app.is_approvable_entry(entry)
    ]
    assert not wrongly_approvable, f"approvable but not an entry: {wrongly_approvable}"


def test_a_record_that_is_not_an_entry_renders_no_answer_anywhere(tmp_path) -> None:
    """**Guard 5, rendered: what stands between a future exit record and a buy.**

    ``executor._submit`` hard-codes a BUY, so an Approve on an exit would add to the
    position it was meant to close. Neither answer may be rendered for a non-entry - not
    disabled, absent - in the dialog or in the panel, and the panel must not crash on one
    the way ``pending_summary`` would.
    """
    exit_like = NOT_ENTRIES["an exit"]
    _queue(tmp_path, ENTRY, exit_like)

    modal = _run(_modal_script, str(tmp_path), True)
    assert not modal.exception, modal.exception
    (dialog,) = _dialogs(modal)
    keys = {button.key for button in dialog.button}
    assert keys == {
        f"modal-a-{ENTRY['decision_id']}",
        f"modal-r-{ENTRY['decision_id']}",
    }, f"answer controls in the dialog: {sorted(keys)}"
    assert app.NOT_ANSWERABLE in "".join(m.value for m in dialog.markdown)

    panel_root = tmp_path / "panel"
    _queue(panel_root, exit_like)
    panel = _run(_panel_script, str(panel_root))
    assert not panel.exception, panel.exception
    assert not list(panel.button), [b.key for b in panel.button]
    assert app.NOT_ANSWERABLE in "".join(m.value for m in panel.markdown)


# ── the source, read as structure ────────────────────────────────────────────────


def _tree() -> ast.Module:
    return ast.parse(Path(app.__file__).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (found,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def _calls(node: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attribute
    ]


def _calls_by_name(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def _session_state_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "session_state"
        and isinstance(node.slice, ast.Name)
        and node.slice.id == "APPROVAL_OPEN"
    )


# ── guard 1: open while something is pending, and only then ──────────────────────


def test_the_modal_opens_when_something_is_pending_and_not_otherwise(tmp_path) -> None:
    """**Guard 1.** A queued entry opens one dialog; an empty or absent queue opens none.
    And `main` calls it - a dialog nothing calls is a library, not a product - as a
    statement of its own body, never under a condition, before the first region draws.
    Since step 3 that statement assigns the queue it returns to the status strip.
    """
    empty = _run(_modal_script, str(tmp_path), True)
    assert not empty.exception, empty.exception
    assert not _dialogs(empty), "a dialog opened with nothing pending"
    assert empty.session_state[app.APPROVAL_OPEN] is False

    _queue(tmp_path, ENTRY)
    pending = _run(_modal_script, str(tmp_path), True)
    assert not pending.exception, pending.exception
    (dialog,) = _dialogs(pending)
    assert dialog.proto.dialog.title == app.APPROVAL_TITLE
    assert pending.session_state[app.APPROVAL_OPEN] is True

    main = _function(_tree(), "main")
    statements = [
        statement.value.func.id
        for statement in main.body
        if isinstance(statement, ast.Expr | ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    assert "approval_modal" in statements, "main does not call approval_modal"
    first_strip = min(call.lineno for call in _calls_by_name(main, "session_strip"))
    (modal_call,) = _calls_by_name(main, "approval_modal")
    assert modal_call.lineno < first_strip, "the modal is called after the page draws"


# ── guard 2: dismissing is not an answer ─────────────────────────────────────────


def test_dismissing_the_modal_answers_nothing(tmp_path) -> None:
    """**Guard 2.** The dialog is dismissible, and a dismissal runs nothing:
    ``on_dismiss`` is the literal ``"ignore"``, never a callback that could answer. Run
    after run with no answer given, the queue file is byte-identical and the dialog is
    back."""
    (dialog_call,) = _calls(_function(_tree(), "approval_modal"), "dialog")
    options = {k.arg: k.value for k in dialog_call.keywords}
    assert isinstance(options.get("on_dismiss"), ast.Constant), "on_dismiss not literal"
    assert options["on_dismiss"].value == "ignore", options["on_dismiss"].value
    assert isinstance(options.get("dismissible"), ast.Constant)
    assert options["dismissible"].value is True

    _queue(tmp_path, ENTRY)
    queued = (tmp_path / records.PENDING_FILE).read_bytes()
    at = _run(_modal_script, str(tmp_path), True)
    for _ in range(2):
        assert not at.exception, at.exception
        assert len(_dialogs(at)) == 1, "the dialog did not come back"
        assert (tmp_path / records.PENDING_FILE).read_bytes() == queued
        at = at.run()


# ── guard 3: neither answer is a default ─────────────────────────────────────────


def test_neither_answer_is_a_default() -> None:
    """**Guard 3, in the source.** Every answer button is a caption and a key and nothing
    else: no ``type`` to style one as primary, no ``shortcut`` (a keyboard binding
    Streamlit 1.61 offers on ``st.button``), no ``on_click``; and no form, whose submit
    button Enter would press. Where focus lands on the rendered page is asserted in
    ``test_approval_fills_on_the_page.py``."""
    tree = _tree()
    for name in ("approval_modal", "_approval_dialog", "_approval_controls"):
        function = _function(tree, name)
        forms = _calls(function, "form") + _calls(function, "form_submit_button")
        assert not forms, f"{name} opens a form, whose submit Enter would press"
        for button in _calls(function, "button"):
            extras = sorted({k.arg for k in button.keywords} - {"key"})
            assert (
                len(button.args) == 1 and not extras
            ), f"{name} line {button.lineno}: {ast.unparse(button)} carries {extras}"


# ── guard 4: everything on the card is the record's own ──────────────────────────


def test_every_value_on_the_card_is_the_records_own() -> None:
    """**Guard 4.** Each figure is the record's value through ``str()``, the narrative
    is the record's narrative rendered as the panel renders it, and **every number a reader
    can see occurs in the record** - so a card that multiplied, rounded or re-worded
    anything fails, whatever it printed."""
    card = app.approval_record_html(ENTRY, loop_present=True)
    for key in ("decision_id", "as_of", "symbol", *(key for _, key in app.CARD_FACTS)):
        assert app.escape(str(ENTRY[key])) in card, f"{key} is not shown verbatim"
    assert app.narrative_html(ENTRY["narrative"]) in card, "the narrative was changed"
    assert ENTRY["record"]["signal"]["action"] in card

    visible = html.unescape(re.sub(r"<[^>]+>", " ", card))
    source = " ".join(str(value) for value in ENTRY.values())
    invented = [n for n in re.findall(r"\d+(?:[.:,]\d+)*", visible) if n not in source]
    assert (
        not invented
    ), f"numbers on the card that the record does not hold: {invented}"

    assert app.escape(app.LOOP_DOWN_WARNING) not in card
    assert app.escape(app.LOOP_DOWN_WARNING) in app.approval_record_html(
        ENTRY, loop_present=False
    ), "the loop-down warning is missing"


# ── guard 6: opened by state, on every full run ──────────────────────────────────


def test_the_modal_is_opened_by_state_on_every_run_never_by_a_button(tmp_path) -> None:
    """**Guard 6.** The open state is written to ``st.session_state`` from the queue, the
    dialog is called under exactly that state, no condition above it is a call - a button
    press is a call - and ``approval_modal`` renders no button of its own. Run after run
    with nothing clicked, the dialog is there."""
    modal = _function(_tree(), "approval_modal")
    assert not _calls(modal, "button"), "approval_modal renders a button of its own"
    writes = [
        node
        for node in ast.walk(modal)
        if isinstance(node, ast.Assign)
        and any(_session_state_subscript(target) for target in node.targets)
    ]
    assert len(writes) == 1, "the open state is not written to session_state once"
    assert ast.unparse(writes[0].value) == "bool(queue)", ast.unparse(writes[0].value)
    assert any(
        isinstance(node, ast.Assign)
        and ast.unparse(node.value) == "records.load_pending(root)"
        for node in ast.walk(modal)
    ), "the queue is not read as _copilot_panel reads it"

    (dialog_call,) = _calls(modal, "dialog")
    guards = [
        branch
        for branch in ast.walk(modal)
        if isinstance(branch, ast.If | ast.While)
        and any(node is dialog_call for part in branch.body for node in ast.walk(part))
    ]
    assert len(guards) == 1 and _session_state_subscript(guards[0].test), (
        "the dialog is not opened by the session state alone: "
        f"{[ast.unparse(guard.test) for guard in guards]}"
    )
    assert not [
        node
        for guard in guards
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
    ], "a call decides whether the dialog opens"

    _queue(tmp_path, ENTRY)
    at = _run(_modal_script, str(tmp_path), True)
    for _ in range(3):
        assert not at.exception, at.exception
        assert len(_dialogs(at)) == 1, "a full run with something pending had no dialog"
        assert at.session_state[app.APPROVAL_OPEN] is True
        at = at.run()


# ── step 2b: the sizing price says what it is ────────────────────────────────────


def test_the_sizing_price_is_never_labelled_as_a_price_to_pay(tmp_path) -> None:
    """**GB-67 step 2b.** The record's ``price`` is the last price the loop fetched when
    it sized the order - in session, the in-progress bar's latest price. Labelled
    ``PRICE`` on the card and written after an ``@`` in the panel, it read as the price
    the order would fill at, and on 2026-09-30 it misled a reader who had the context.

    Neither surface may call it the bare word PRICE, the card must still show the
    record's value verbatim under its label, and both surfaces must say the order fills
    at market.
    """
    assert app.PRICE_LABEL != "PRICE"
    card = app.approval_record_html(ENTRY, loop_present=True)
    shown = dict(
        re.findall(
            r'gb-stat-key">([^<]*)</span><span class="gb-stat-value"[^>]*>([^<]*)<',
            card,
        )
    )
    assert "PRICE" not in shown, f"the card labels a figure PRICE: {shown}"
    assert shown.get(app.PRICE_LABEL) == app.escape(str(ENTRY["price"])), shown

    summary = app.pending_summary(ENTRY)
    assert "@" not in summary, f"the panel still prices with an @: {summary}"
    assert f"{app.PRICE_LABEL} {ENTRY['price']:,.2f}" in summary, summary

    _queue(tmp_path, ENTRY)
    (dialog,) = _dialogs(_run(_modal_script, str(tmp_path), True))
    assert app.FILLS_AT_MARKET in "".join(m.value for m in dialog.markdown)
    panel = _run(_panel_script, str(tmp_path))
    assert app.FILLS_AT_MARKET in "".join(m.value for m in panel.markdown)


# ── step 4: what each answer does, and what the action is called ─────────────────


def test_the_dialog_says_what_each_answer_does_with_equal_weight(tmp_path) -> None:
    """**GB-67 step 4.** Both consequences are stated in the dialog itself, in the same
    class: an explained Approve beside a silent Reject is not two equal answers, and the
    panel's line about rejections sits behind the backdrop while the dialog is open."""
    _queue(tmp_path, ENTRY)
    (dialog,) = _dialogs(_run(_modal_script, str(tmp_path), True))
    shown = "".join(m.value for m in dialog.markdown)

    for line in (app.APPROVE_DOES, app.REJECT_DOES):
        rendered = f'<div class="gb-meta">{app.escape(line)}</div>'
        assert shown.count(rendered) == 1, f"the dialog does not say, once: {line}"


def test_every_action_has_a_label_and_an_unknown_one_is_shown_as_stored() -> None:
    """**GB-67 step 4.** The reader's word for an action comes from one fixed mapping,
    total over the actions the engine can record, and a value outside it is printed as
    stored rather than guessed. The card shows both, so the word stays traceable to the
    record."""
    from glassbox.engine.signal import ACTIONS

    assert set(app.ACTION_LABELS) == set(ACTIONS), sorted(app.ACTION_LABELS)
    assert all(label.strip() for label in app.ACTION_LABELS.values())
    assert app.action_label("not-an-action") == "not-an-action"

    card = app.approval_record_html(ENTRY, loop_present=True)
    stored = ENTRY["record"]["signal"]["action"]
    assert app.escape(app.action_label(stored)) in card, "the reader's word is missing"
    assert (
        f'<span class="gb-meta">{app.escape(stored)}</span>' in card
    ), "the stored action is no longer visible on the card"


# ── step 4: the dialog does not depend on the broker ─────────────────────────────


def _main_without_broker(root: str) -> None:
    from glassbox.dashboard import app

    def refuse(cfg) -> None:
        raise ConnectionError("the broker read is refused for this test")

    original = app._broker_view
    app._broker_view = refuse
    try:
        app.main(root)
    finally:
        app._broker_view = original


def test_the_dialog_is_reached_when_the_broker_read_fails(tmp_path) -> None:
    """**GB-67 step 4.** The real ``main``, with the broker read raising and no earlier
    read to fall back on: the page stops at NO BROKER READ HAS SUCCEEDED YET, and the
    pending decision is still asked and still counted. The dialog needs nothing the read
    provides, so the guarantee holds whatever makes the read raise - not because
    ``_broker_view`` happens to swallow its own failures today.

    The NO BROKER READ line is asserted first, because without it this test could pass
    on a run where the read never failed at all.
    """
    _queue(tmp_path, ENTRY)
    at = _run(_main_without_broker, str(tmp_path))
    assert not at.exception, at.exception
    page = "".join(m.value for m in at.markdown)

    assert "NO BROKER READ HAS SUCCEEDED YET" in page, "the broker read did not fail"
    assert len(_dialogs(at)) == 1, "a decision was pending and no dialog was reached"
    assert (
        "1 AWAITING AN ANSWER" in page
    ), "the strip does not count the pending decision"


# ── step 5a: every run of the page finishes ──────────────────────────────────────


def test_nothing_in_the_dashboard_sleeps_inside_a_run() -> None:
    """**GB-67 step 5a.** A sleep inside a run is a run that has not finished, and
    Streamlit clears an element a newer run stopped sending only when a run finishes. On
    2026-10-01 ``_refresh`` slept for a cycle and then reran, so no run of the page ever
    finished and an answered approval dialog stayed on screen inviting a second answer.
    The refresh is a fragment Streamlit re-executes; nothing in the module sleeps, by any
    binding of ``sleep``."""
    tree = _tree()
    sleeps = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "time"
        for alias in node.names
        if alias.name == "sleep"
    }
    found = [
        f"line {node.lineno}: {ast.unparse(node)}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "sleep")
            or (isinstance(node.func, ast.Name) and node.func.id in sleeps)
        )
    ]
    assert not found, f"the dashboard sleeps inside a run: {found}"

    # `_refresh` reads the run's start, and only `main` writes it: two places, pinned.
    main = _function(tree, "main")
    writes = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and ast.unparse(target) == "st.session_state[RUN_STARTED]"
            for target in node.targets
        )
    ]
    refreshes = [call.lineno for call in _calls_by_name(main, "_refresh")]
    assert (
        writes and refreshes and min(writes) < min(refreshes)
    ), f"main does not record when its run began before refreshing: {writes}, {refreshes}"
