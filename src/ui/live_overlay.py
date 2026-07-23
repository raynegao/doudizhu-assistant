from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any, Callable

from src.pipeline.live_runtime import LiveRuntimeSnapshot
from src.state.events import PlayerSeat


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
    def from_snapshot(cls, snapshot: LiveRuntimeSnapshot) -> "LiveOverlayViewModel":
        state = snapshot.state
        if state is None:
            return cls(
                status=snapshot.tracker_update.message,
                roles="等待完整新局",
                remaining="余牌：--",
                trick="当前牌：--",
                best="推荐：--",
                top_k=(),
                confidence=f"场面置信度：{snapshot.scene.confidence:.1%}",
                warnings="\n".join(snapshot.scene.warnings[:4]),
            )
        remaining = state.remaining_by_player
        scope = "个人" if state.landlord is PlayerSeat.SELF else "农民团队"
        if not state.decision_ready:
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


class LiveAssistantOverlay:
    """Read-only Tk overlay; game capture and decisions stay outside the UI layer."""

    def __init__(
        self,
        snapshots: "queue.Queue[LiveRuntimeSnapshot]",
        *,
        on_close: Callable[[], None] | None = None,
        geometry: str = "250x430+0+70",
    ) -> None:
        import tkinter as tk

        self._tk = tk
        self.snapshots = snapshots
        self.on_close = on_close
        self.root = tk.Tk()
        self.root.title("斗地主助手")
        self.root.geometry(geometry)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#101828")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False

        self.status = tk.StringVar(value="正在启动…")
        self.roles = tk.StringVar(value="等待完整新局")
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
        latest: LiveRuntimeSnapshot | None = None
        while True:
            try:
                latest = self.snapshots.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.present(LiveOverlayViewModel.from_snapshot(latest))
        self.root.after(80, self._poll)

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
