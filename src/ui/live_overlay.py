from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any, Callable

from src.capture.screen_geometry import WindowCaptureStatus
from src.pipeline.live_runtime import LiveRuntimeSnapshot
from src.state.events import PlayerSeat, RoundPhase


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

    @classmethod
    def from_runtime_error(cls, message: str) -> "LiveOverlayViewModel":
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
    def from_snapshot(cls, snapshot: LiveRuntimeSnapshot) -> "LiveOverlayViewModel":
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
        state = snapshot.state
        if snapshot.current_scan_pending:
            return cls(
                status=(
                    "正在扫描当前牌局，等待牌面与余牌数稳定…\n"
                    f"{snapshot.tracker_update.message}"
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
                    (
                        state.landlord is PlayerSeat.SELF
                        and winner is PlayerSeat.SELF
                    )
                    or (
                        state.landlord is not PlayerSeat.SELF
                        and winner is not state.landlord
                    )
                )
            )
            return cls(
                status=(
                    f"R{state.revision} · F{snapshot.frame_id} · "
                    f"{snapshot.tracker_update.message}"
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
            first = result.rankings[0]
            best = (
                f"最佳：{result.action}\n"
                f"估计{scope}胜率：{first.estimated_win_rate:.1%}"
            )
            top_k = tuple(
                (
                    f"{index}. {evaluation.action}  "
                    f"{evaluation.estimated_win_rate:.1%}  "
                    f"n={evaluation.simulations}"
                )
                for index, evaluation in enumerate(result.rankings, start=1)
            )
        return cls(
            status=(
                f"R{state.revision} · F{snapshot.frame_id} · "
                f"{snapshot.tracker_update.mode.value} · "
                f"{snapshot.tracker_update.message}"
            ),
            roles=(
                f"地主：{state.landlord.value}  当前：{state.current_actor.value}"
            ),
            remaining=(
                f"余牌 我{remaining[PlayerSeat.SELF]} "
                f"右{remaining[PlayerSeat.RIGHT]} "
                f"左{remaining[PlayerSeat.LEFT]}"
            ),
            trick=(
                "待压："
                + (
                    " ".join(state.trick_target.cards)
                    if state.trick_target
                    else "自由出牌"
                )
            ),
            best=best,
            top_k=top_k,
            confidence=(
                f"状态置信度：{state.state_confidence:.1%}  "
                f"帧耗时：{snapshot.total_latency_ms:.0f}ms"
            ),
            warnings="\n".join(state.warnings[:4]),
        )


def _advance_frame_cursor(previous: int, incoming: int) -> tuple[int, bool]:
    """Track worker-local frame ids and expose recognition process restarts."""

    restarted = incoming < previous
    return (incoming if restarted else max(previous, incoming), restarted)


class LiveAssistantOverlay:
    """Read-only Tk overlay; game capture and decisions stay outside the UI layer."""

    def __init__(
        self,
        snapshots: "queue.Queue[LiveRuntimeSnapshot]",
        *,
        runtime_errors: "queue.Queue[str] | None" = None,
        on_close: Callable[[], None] | None = None,
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
        self.on_scan = on_scan
        self.health_check = health_check
        self.root = tk.Tk()
        self.root.title("斗地主助手")
        self.root.geometry(geometry)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#101828")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False
        self._last_frame_id = 0
        self._scan_after_frame_id: int | None = None

        self.status = tk.StringVar(value="正在启动…")
        self.roles = tk.StringVar(value="等待地主/加倍完成")
        self.remaining = tk.StringVar(value="余牌：--")
        self.trick = tk.StringVar(value="当前牌：--")
        self.best = tk.StringVar(value="推荐：--")
        self.top_k = tk.StringVar(value="")
        self.confidence = tk.StringVar(value="")
        self.warnings = tk.StringVar(value="")

        self._label(
            "Phase 6 · 只读助手",
            font=("PingFang SC", 15, "bold"),
            foreground="#fdb022",
        )
        self.scan_button = tk.Button(
            self.root,
            text="扫描当前牌局",
            command=self._request_scan,
            background="#175cd3",
            foreground="#101828",
            activebackground="#1570ef",
            activeforeground="#101828",
            disabledforeground="#667085",
            relief="flat",
            borderwidth=0,
            font=("PingFang SC", 12, "bold"),
            padx=10,
            pady=7,
            cursor="pointinghand",
            state="normal" if on_scan is not None else "disabled",
        )
        self.scan_button.pack(fill="x", padx=10, pady=(2, 5))
        self._variable_label(self.status, foreground="#98a2b3", wraplength=230)
        self._variable_label(self.roles)
        self._variable_label(self.remaining)
        self._variable_label(self.trick)
        self._variable_label(
            self.best,
            font=("PingFang SC", 15, "bold"),
            foreground="#75e0a7",
        )
        self._variable_label(self.top_k, foreground="#d0d5dd")
        self._variable_label(self.confidence, foreground="#98a2b3")
        self._variable_label(
            self.warnings,
            foreground="#f97066",
            wraplength=230,
        )
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
        if self.on_close is not None:
            self.on_close()
        self.root.destroy()

    def _poll(self) -> None:
        if self._closed:
            return
        if self.health_check is not None:
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
            self.present(
                LiveOverlayViewModel.from_runtime_error(self._runtime_error)
            )
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

    def _request_scan(self) -> None:
        if self.on_scan is None or self._scan_after_frame_id is not None:
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
            text="扫描当前牌局",
            state="normal" if self.on_scan is not None else "disabled",
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

    def _label(self, text: str, **kwargs: object) -> Any:
        label = self._tk.Label(
            self.root,
            text=text,
            background="#101828",
            foreground=kwargs.pop("foreground", "#f2f4f7"),
            anchor="w",
            justify="left",
            padx=10,
            pady=4,
            **kwargs,
        )
        label.pack(fill="x")
        return label

    def _variable_label(
        self,
        variable: Any,
        **kwargs: object,
    ) -> Any:
        label = self._tk.Label(
            self.root,
            textvariable=variable,
            background="#101828",
            foreground=kwargs.pop("foreground", "#f2f4f7"),
            anchor="w",
            justify="left",
            padx=10,
            pady=4,
            **kwargs,
        )
        label.pack(fill="x")
        return label


__all__ = ["LiveAssistantOverlay", "LiveOverlayViewModel"]
