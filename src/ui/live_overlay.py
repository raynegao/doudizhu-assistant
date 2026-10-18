from __future__ import annotations

import queue
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.capture.screen_geometry import WindowCaptureStatus
from src.logic.monte_carlo import ActionEvaluation
from src.pipeline.live_runtime import LiveRuntimeSnapshot
from src.state.events import PlayerSeat, RoundPhase
from src.vision.scene_recognizer import RoundResultOutcome


@dataclass(frozen=True)
class LiveOverlayViewModel:
    status: str
    roles: str
    remaining: str
    trick: str
    best: str
    top_k: tuple[str, ...]
    confidence: str
    warnings: str
    warning_color: str = "#f97066"

    @classmethod
    def from_runtime_error(cls, message: str) -> LiveOverlayViewModel:
        return cls(
            status="识别线程异常停止",
            roles="识别状态：不可用",
            remaining="余牌：--",
            trick="当前牌：--",
            best="推荐：已暂停",
            top_k=(),
            confidence="场面置信度：0.0%",
            warnings=message,
        )

    @classmethod
    def from_snapshot(cls, snapshot: LiveRuntimeSnapshot) -> LiveOverlayViewModel:
        if not snapshot.window_available:
            return cls(
                status=snapshot.window_message,
                roles="识别状态：不可用",
                remaining="余牌：--",
                trick="当前牌：--",
                best="推荐：已暂停",
                top_k=(),
                confidence="场面置信度：0.0%",
                warnings=(
                    "请打开斗地主窗口"
                    if snapshot.window_status is WindowCaptureStatus.NOT_OPEN
                    else "请切换到或还原斗地主窗口后继续"
                ),
            )
        visual_result = snapshot.tracker_update.round_result
        if visual_result is not None:
            result_text = {
                RoundResultOutcome.VICTORY: "胜利",
                RoundResultOutcome.DEFEAT: "失败",
                RoundResultOutcome.UNKNOWN: "本局已结束",
            }[visual_result.outcome]
            return cls(
                status=(f"F{snapshot.frame_id} · {snapshot.tracker_update.message}"),
                roles="已识别结算画面",
                remaining="余牌：本局已结束",
                trick="当前牌：--",
                best=f"结果：{result_text}",
                top_k=(),
                confidence=f"结算置信度：{visual_result.confidence:.1%}",
                warnings="已停止本局推荐并清除旧断点",
                warning_color="#75e0a7",
            )
        state = snapshot.state
        if snapshot.current_scan_pending:
            return cls(
                status=(
                    f"正在扫描当前牌局，等待牌面与余牌数稳定…\n{snapshot.tracker_update.message}"
                ),
                roles="正在重新识别地主与当前行动者",
                remaining="余牌：扫描中…",
                trick="当前牌：扫描中…",
                best="推荐：正在重新建模…",
                top_k=(),
                confidence=f"场面置信度：{snapshot.scene.confidence:.1%}",
                warnings="\n".join(snapshot.scene.warnings[:4]),
            )
        if state is None:
            return cls(
                status=snapshot.tracker_update.message,
                roles="等待地主/加倍完成",
                remaining="余牌：--",
                trick="当前牌：--",
                best="推荐：--",
                top_k=(),
                confidence=f"场面置信度：{snapshot.scene.confidence:.1%}",
                warnings="\n".join(snapshot.scene.warnings[:4]),
            )
        remaining = state.remaining_by_player
        if state.phase is RoundPhase.FINISHED:
            winner = state.winner
            victory = bool(
                winner is not None
                and (
                    (state.landlord is PlayerSeat.SELF and winner is PlayerSeat.SELF)
                    or (state.landlord is not PlayerSeat.SELF and winner is not state.landlord)
                )
            )
            return cls(
                status=(
                    f"R{state.revision} · F{snapshot.frame_id} · {snapshot.tracker_update.message}"
                ),
                roles=(
                    f"地主：{state.landlord.value}  "
                    f"胜者：{winner.value if winner is not None else '--'}"
                ),
                remaining=(
                    f"余牌 我{remaining[PlayerSeat.SELF]} "
                    f"右{remaining[PlayerSeat.RIGHT]} "
                    f"左{remaining[PlayerSeat.LEFT]}"
                ),
                trick="本局已结束",
                best=("结果：胜利" if victory else "结果：失败"),
                top_k=(),
                confidence=f"状态置信度：{state.state_confidence:.1%}",
                warnings="牌局断点已自动清除，等待下一局",
            )
        scope = "个人" if state.landlord is PlayerSeat.SELF else "农民团队"
        top_k: tuple[str, ...] = ()
        decision_warnings: tuple[str, ...] = ()
        if snapshot.decision_block_reason:
            best = f"推荐：已暂停\n{snapshot.decision_block_reason}"
            top_k = ()
        elif not state.decision_ready:
            best = "推荐：状态未确认，已暂停"
            top_k = ()
        elif snapshot.decision_pending:
            best = "推荐：胜率计算中…"
            top_k = ()
        elif snapshot.decision is None:
            best = "推荐：当前不输出"
            top_k = ()
        else:
            result = snapshot.decision.result
            decision_warnings = result.warnings
            first = result.rankings[0]
            best = (
                f"最佳：{result.action}\n{_format_evaluation(first, scope=scope)}"
            )
            top_k = tuple(
                (
                    f"{index}. {evaluation.action}  "
                    f"{_format_evaluation(evaluation, compact=True)}  n={evaluation.simulations}"
                )
                for index, evaluation in enumerate(result.rankings, start=1)
            )
        warnings, warning_color = _format_state_warnings(
            (*state.warnings, *snapshot.tracker_update.warnings),
            decision_warnings=decision_warnings,
            block_reason=snapshot.decision_block_reason,
        )
        return cls(
            status=(
                f"R{state.revision} · F{snapshot.frame_id} · "
                f"{snapshot.tracker_update.mode.value} · "
                f"{snapshot.tracker_update.message}"
            ),
            roles=(f"地主：{state.landlord.value}  当前：{state.current_actor.value}"),
            remaining=(
                f"余牌 我{remaining[PlayerSeat.SELF]} "
                f"右{remaining[PlayerSeat.RIGHT]} "
                f"左{remaining[PlayerSeat.LEFT]}"
            ),
            trick=(
                "待压："
                + (" ".join(state.trick_target.cards) if state.trick_target else "自由出牌")
            ),
            best=best,
            top_k=top_k,
            confidence=(
                f"状态置信度：{state.state_confidence:.1%}  "
                f"帧耗时：{snapshot.total_latency_ms:.0f}ms"
            ),
            warnings=warnings,
            warning_color=warning_color,
        )


def _format_evaluation(
    evaluation: ActionEvaluation, *, scope: str = "", compact: bool = False,
) -> str:
    if evaluation.estimated_win_rate is None:
        return "胜率不可用（规则回退）"
    if "exact_finish" in evaluation.reason_factors:
        return "可直接出完（规则已确认）"
    if evaluation.win_rate_estimate_kind == "douzero_value":
        return f"DouZero 估值（未校准）：{evaluation.estimated_win_rate:.1%}"
    is_score = (
        evaluation.win_rate_estimate_kind == "bounded_rollout_score"
        or "depth_limit_reached" in evaluation.risk_flags
    )
    label = f"{scope}模拟评分" if is_score else f"模型内{scope}胜率"
    if compact:
        label = "模拟评分" if is_score else "模型胜率"
    interval = (
        f"（95%{'评分' if is_score else ''}区间 "
        f"{evaluation.win_rate_ci95_low:.1%}–{evaluation.win_rate_ci95_high:.1%}）"
        if evaluation.win_rate_ci95_low is not None and evaluation.win_rate_ci95_high is not None
        else ""
    )
    return f"{label}：{evaluation.estimated_win_rate:.1%}{interval}"


def _format_state_warnings(
    warnings: tuple[str, ...],
    *,
    decision_warnings: tuple[str, ...] = (),
    block_reason: str = "",
) -> tuple[str, str]:
    """Keep blockers first; compact and deduplicate model/sampling caveats."""

    approximation_keys = {
        "automatic_current_game_scan",
        "estimated_win_rate_uses_uniform_unknown_history",
    }
    history_prefix = "historical_played_cards_unknown="
    combined = tuple(dict.fromkeys((*warnings, *decision_warnings)))
    approximation = [
        warning
        for warning in combined
        if warning in approximation_keys or warning.startswith(history_prefix)
    ]
    sampling_labels = {
        "low_sample_count": "样本不足",
        "top_action_confidence_intervals_overlap": "候选区间重叠，排序不稳定",
        "time_budget_exhausted": "计算预算耗尽",
        "time_budget_overrun": "计算预算超时",
    }
    model_labels = {
        "uniform_opponent_model": "均匀对手模型近似，区间不含模型偏差",
        "heuristic_score_not_win_probability": "含截断评分，不等同真实胜率",
    }
    other_labels = {
        "candidate_pruned": "仅评估部分合法动作",
        "rule_subset_only": "仅覆盖当前规则子集",
    }
    informational = {*sampling_labels, *model_labels, *other_labels}
    keys = {warning.partition(":")[0] for warning in combined}
    blocking = list(dict.fromkeys(
        ([block_reason] if block_reason else [])
        + [
            warning for warning in combined
            if warning not in approximation and warning.partition(":")[0] not in informational
        ]
    ))
    lines = blocking[:3]
    if approximation:
        hidden_count = next(
            (
                warning.removeprefix(history_prefix)
                for warning in approximation
                if warning.startswith(history_prefix)
            ),
            "若干",
        )
        lines.append(f"中途扫描：{hidden_count} 张历史牌未知，胜率为近似估计")
    for labels in (sampling_labels, model_labels, other_labels):
        line = "；".join(label for key, label in labels.items() if key in keys)
        if line:
            lines.append(line)
    return "\n".join(lines), ("#f97066" if blocking else "#fdb022")


def _advance_frame_cursor(previous: int, incoming: int) -> tuple[int, bool]:
    """Track worker-local frame ids and expose recognition process restarts."""

    restarted = incoming < previous
    return (incoming if restarted else max(previous, incoming), restarted)


class LiveAssistantOverlay:
    """Read-only Tk overlay; game capture and decisions stay outside the UI layer."""

    def __init__(
        self,
        snapshots: queue.Queue[LiveRuntimeSnapshot],
        *,
        runtime_errors: queue.Queue[str] | None = None,
        on_close: Callable[[], None] | None = None,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_scan: Callable[[], None] | None = None,
        health_check: Callable[[], str | None] | None = None,
        geometry: str = "260x480+0+70",
    ) -> None:
        import tkinter as tk

        self._tk = tk
        self.snapshots = snapshots
        self.runtime_errors = runtime_errors
        self._runtime_error: str | None = None
        self.on_close = on_close
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_scan = on_scan
        self.health_check = health_check
        self.root = tk.Tk()
        self.root.title("斗地主助手")
        self.root.geometry(geometry)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#101828")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False
        self._scanning = False
        self._last_frame_id = 0
        self._scan_after_frame_id: int | None = None

        self.status = tk.StringVar(value="扫描已停止；未获取屏幕")
        self.roles = tk.StringVar(value="等待地主/加倍完成")
        self.remaining = tk.StringVar(value="余牌：--")
        self.trick = tk.StringVar(value="当前牌：--")
        self.best = tk.StringVar(value="推荐：--")
        self.top_k = tk.StringVar(value="")
        self.confidence = tk.StringVar(value="")
        self.warnings = tk.StringVar(value="点击“开始扫描”后才会读取并限量记录斗地主窗口")

        self._label(
            "Phase 6 · 只读助手",
            font=("PingFang SC", 15, "bold"),
            foreground="#fdb022",
        )
        self.scan_toggle_button = tk.Button(
            self.root,
            text="开始扫描",
            command=self._toggle_scanning,
            background="#067647",
            foreground="#101828",
            activebackground="#079455",
            activeforeground="#101828",
            disabledforeground="#667085",
            relief="flat",
            borderwidth=0,
            font=("PingFang SC", 12, "bold"),
            padx=10,
            pady=7,
            cursor="pointinghand",
            state=("normal" if on_start is not None and on_stop is not None else "disabled"),
        )
        self.scan_toggle_button.pack(fill="x", padx=10, pady=(2, 5))
        self.scan_button = tk.Button(
            self.root,
            text="重新扫描当前牌局",
            command=self._request_scan,
            background="#175cd3",
            foreground="#101828",
            activebackground="#1570ef",
            activeforeground="#101828",
            disabledforeground="#667085",
            relief="flat",
            borderwidth=0,
            font=("PingFang SC", 11, "bold"),
            padx=10,
            pady=5,
            cursor="pointinghand",
            state="disabled",
        )
        self.scan_button.pack(fill="x", padx=10, pady=(2, 5))
        # Keep capture controls reachable while longer caveats/results scroll.
        body = tk.Frame(self.root, background="#101828")
        body.pack(fill="both", expand=True)
        self.content_canvas = tk.Canvas(body, background="#101828", highlightthickness=0)
        scrollbar = tk.Scrollbar(body, orient="vertical", command=self.content_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.content_canvas.pack(side="left", fill="both", expand=True)
        self.content_canvas.configure(yscrollcommand=scrollbar.set)
        self._content = tk.Frame(self.content_canvas, background="#101828")
        content_window = self.content_canvas.create_window((0, 0), window=self._content, anchor="nw")
        self._content.bind(
            "<Configure>",
            lambda _event: self.content_canvas.configure(
                scrollregion=self.content_canvas.bbox("all"),
            ),
        )
        self.content_canvas.bind(
            "<Configure>",
            lambda event: self.content_canvas.itemconfigure(content_window, width=event.width),
        )
        self.root.bind(
            "<MouseWheel>",
            lambda event: self.content_canvas.yview_scroll(-int(event.delta), "units"),
        )
        self._variable_label(self.status, foreground="#98a2b3", wraplength=230)
        self._variable_label(self.roles)
        self._variable_label(self.remaining)
        self._variable_label(self.trick)
        self._variable_label(
            self.best,
            font=("PingFang SC", 15, "bold"),
            foreground="#75e0a7",
            wraplength=230,
        )
        self.warning_label = self._variable_label(
            self.warnings,
            foreground="#f97066",
            wraplength=230,
        )
        self._variable_label(self.top_k, foreground="#d0d5dd", wraplength=230)
        self._variable_label(self.confidence, foreground="#98a2b3")
        self._label(
            "估计胜率 · 不自动点击",
            foreground="#667085",
        )
        self.root.after(80, self._poll)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.on_close is not None:
                self.on_close()
        except Exception as exc:  # noqa: BLE001
            # Do not close the only control surface or imply capture stopped
            # when the controller still owns a live worker.
            self._closed = False
            self.status.set("无法确认扫描已经停止")
            self.warnings.set(str(exc))
            self.warning_label.configure(foreground="#f97066")
            return
        # The closed overlay can remain reachable through Tk/signal callbacks.
        # Do not let it retain the retired IPC queue's named semaphores.
        self.snapshots = queue.Queue()
        self.root.destroy()

    def _poll(self) -> None:
        if self._closed:
            return
        if self._scanning and self.health_check is not None:
            health_error = self.health_check()
            if health_error is not None:
                self._runtime_error = health_error
        latest_error: str | None = None
        if self.runtime_errors is not None:
            while True:
                try:
                    latest_error = self.runtime_errors.get_nowait()
                except queue.Empty:
                    break
        if latest_error is not None:
            self._runtime_error = latest_error
        if self._runtime_error is not None:
            if self._scanning and self._scan_after_frame_id is not None:
                # A worker can exhaust its restart budget before producing its
                # first frame. Re-enable the explicit recovery button instead
                # of asking the user to click a permanently disabled control.
                self._finish_scan_request()
            self.present(LiveOverlayViewModel.from_runtime_error(self._runtime_error))
            self.root.after(80, self._poll)
            return
        latest: LiveRuntimeSnapshot | None = None
        while True:
            try:
                latest = self.snapshots.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._last_frame_id, worker_restarted = _advance_frame_cursor(
                self._last_frame_id,
                latest.frame_id,
            )
            if worker_restarted and self._scan_after_frame_id is not None:
                # Recognition workers restart their local frame counter at 1.
                # A scan request tied to the previous counter can otherwise
                # leave the button disabled for hundreds of frames.
                self._finish_scan_request()
            self.present(LiveOverlayViewModel.from_snapshot(latest))
            if (
                self._scan_after_frame_id is not None
                and latest.frame_id > self._scan_after_frame_id
                and not latest.current_scan_pending
            ):
                self._finish_scan_request()
        self.root.after(80, self._poll)

    def _toggle_scanning(self) -> None:
        if self._scanning:
            self._stop_scanning()
        else:
            self._start_scanning()

    def _start_scanning(self) -> None:
        if self.on_start is None or self._scanning:
            return
        self._runtime_error = None
        self._last_frame_id = 0
        self._scan_after_frame_id = 0
        self.scan_toggle_button.configure(text="正在启动…", state="disabled")
        self.status.set("正在启动扫描，请保持斗地主窗口可见…")
        self.best.set("推荐：正在重新建模…")
        try:
            self.on_start()
        except Exception as exc:  # noqa: BLE001
            if bool(getattr(exc, "scan_state_unconfirmed", False)):
                # The controller retains the worker handle and will retry the
                # stop operation. Keep the UI conservative instead of
                # claiming that the screen was never acquired.
                self._scanning = True
                self._scan_after_frame_id = None
                self.status.set("启动失败，尚未确认屏幕扫描已经停止")
                self.warnings.set(str(exc))
                self.scan_toggle_button.configure(
                    text="结束扫描",
                    state="normal",
                    background="#b42318",
                    activebackground="#d92d20",
                )
                self.scan_button.configure(
                    text="重新扫描当前牌局",
                    state="disabled",
                )
                return
            self._scan_after_frame_id = None
            self.status.set("扫描未启动；未获取屏幕")
            self.warnings.set(f"启动扫描失败：{exc}")
            self.scan_toggle_button.configure(text="开始扫描", state="normal")
            return
        self._scanning = True
        self.scan_toggle_button.configure(
            text="结束扫描",
            state="normal",
            background="#b42318",
            activebackground="#d92d20",
        )
        self.scan_button.configure(text="扫描中…", state="disabled")

    def _stop_scanning(self) -> None:
        if self.on_stop is None or not self._scanning:
            return
        self.scan_toggle_button.configure(text="正在停止…", state="disabled")
        try:
            self.on_stop()
        except Exception as exc:  # noqa: BLE001
            self.status.set("尚未确认扫描已经停止，请再次点击“结束扫描”")
            self.warnings.set(f"停止扫描失败：{exc}")
            self.scan_toggle_button.configure(text="结束扫描", state="normal")
            self.scan_button.configure(text="重新扫描当前牌局", state="disabled")
            return
        self._scanning = False
        self._runtime_error = None
        self._last_frame_id = 0
        self._scan_after_frame_id = None
        self.scan_toggle_button.configure(
            text="开始扫描",
            state="normal",
            background="#067647",
            activebackground="#079455",
        )
        self.scan_button.configure(text="重新扫描当前牌局", state="disabled")
        self.present(
            LiveOverlayViewModel(
                status="扫描已停止；未获取屏幕",
                roles="识别状态：已停止",
                remaining="余牌：--",
                trick="当前牌：--",
                best="推荐：--",
                top_k=(),
                confidence="",
                warnings="点击“开始扫描”后才会读取斗地主窗口",
                warning_color="#98a2b3",
            )
        )

    def _request_scan(self) -> None:
        if not self._scanning or self.on_scan is None or self._scan_after_frame_id is not None:
            return
        # The same button is the explicit recovery path after the background
        # worker exceeded its automatic restart budget.
        self._runtime_error = None
        self._scan_after_frame_id = self._last_frame_id
        self.scan_button.configure(text="扫描中…", state="disabled")
        self.status.set("正在扫描当前牌局，请保持斗地主窗口可见…")
        self.best.set("推荐：正在重新建模…")
        try:
            self.on_scan()
        except Exception as exc:  # noqa: BLE001
            self.warnings.set(f"扫描请求失败：{exc}")
            self._finish_scan_request()

    def _finish_scan_request(self) -> None:
        self._scan_after_frame_id = None
        self.scan_button.configure(
            text="重新扫描当前牌局",
            state=("normal" if self._scanning and self.on_scan is not None else "disabled"),
        )

    def present(self, view: LiveOverlayViewModel) -> None:
        self.status.set(view.status)
        self.roles.set(view.roles)
        self.remaining.set(view.remaining)
        self.trick.set(view.trick)
        self.best.set(view.best)
        self.top_k.set("\n".join(view.top_k))
        self.confidence.set(view.confidence)
        self.warnings.set(view.warnings)
        self.warning_label.configure(foreground=view.warning_color)

    def _label(
        self,
        text: str,
        *,
        foreground: str = "#f2f4f7",
        font: tuple[str, int, str] | str = "TkDefaultFont",
        wraplength: int = 0,
    ) -> Any:
        label = self._tk.Label(
            getattr(self, "_content", self.root),
            text=text,
            background="#101828",
            foreground=foreground,
            font=font,
            wraplength=wraplength,
            anchor="w",
            justify="left",
            padx=10,
            pady=4,
        )
        label.pack(fill="x")
        return label

    def _variable_label(
        self,
        variable: Any,
        *,
        foreground: str = "#f2f4f7",
        font: tuple[str, int, str] | str = "TkDefaultFont",
        wraplength: int = 0,
    ) -> Any:
        label = self._tk.Label(
            self._content,
            textvariable=variable,
            background="#101828",
            foreground=foreground,
            font=font,
            wraplength=wraplength,
            anchor="w",
            justify="left",
            padx=10,
            pady=4,
        )
        label.pack(fill="x")
        return label


__all__ = ["LiveAssistantOverlay", "LiveOverlayViewModel"]
