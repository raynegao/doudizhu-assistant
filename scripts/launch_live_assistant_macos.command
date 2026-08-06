#!/bin/zsh

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
PYTHON_PATH="$PROJECT_DIR/.venv/bin/python"
CONFIG_PATH="$PROJECT_DIR/configs/live_game.local.json"
LOG_PATH="$PROJECT_DIR/logs/live_assistant.stdout.log"
PID_PATH="$PROJECT_DIR/logs/live_assistant.pid"
LOCK_PATH="$PROJECT_DIR/logs/live_assistant.launch.lock"
LAUNCH_TTY="$(tty 2>/dev/null)"
LAUNCH_LOCK_ACQUIRED=false

schedule_close_launcher_window() {
  if [[ -z "$LAUNCH_TTY" || "$LAUNCH_TTY" == "not a tty" ]]; then
    return
  fi

  # The AppleScript waits until this .command process has returned. Closing a
  # still-busy Terminal tab can show a confirmation dialog and leave yet
  # another unused window behind.
  /usr/bin/nohup /usr/bin/osascript - "$LAUNCH_TTY" \
    >/dev/null 2>&1 <<'APPLESCRIPT' &
on run argv
  set targetTTY to item 1 of argv
  delay 0.8
  tell application "Terminal"
    -- Clean only completed tabs created by older versions of this launcher.
    -- Matching the launcher-specific output avoids touching unrelated idle
    -- shells, even when they are in the same Terminal application.
    set staleLauncherTTYs to {}
    repeat with terminalWindow in windows
      repeat with terminalTab in tabs of terminalWindow
        if (tty of terminalTab) is not targetTTY and not (busy of terminalTab) then
          set terminalHistory to history of terminalTab
          if terminalHistory contains "助手可以先启动并等待斗地主窗口。" then
            set end of staleLauncherTTYs to tty of terminalTab
          end if
        end if
      end repeat
    end repeat
    repeat with staleTTY in staleLauncherTTYs
      repeat with terminalWindow in windows
        repeat with terminalTab in tabs of terminalWindow
          if (tty of terminalTab) is staleTTY and not (busy of terminalTab) then
            if (count of tabs of terminalWindow) is 1 then
              close terminalWindow
            else
              close terminalTab
            end if
            exit repeat
          end if
        end repeat
      end repeat
    end repeat

    repeat with terminalWindow in windows
      repeat with terminalTab in tabs of terminalWindow
        if (tty of terminalTab) is targetTTY then
          repeat 20 times
            if not (busy of terminalTab) then exit repeat
            delay 0.2
          end repeat
          if not (busy of terminalTab) then
            if (count of tabs of terminalWindow) is 1 then
              close terminalWindow
            else
              close terminalTab
            end if
          else
            set miniaturized of terminalWindow to true
          end if
          return
        end if
      end repeat
    end repeat
  end tell
end run
APPLESCRIPT
  disown $! 2>/dev/null || true
}

cleanup_launcher() {
  if [[ "$LAUNCH_LOCK_ACQUIRED" == true ]]; then
    /bin/rm -f "$LOCK_PATH/pid" 2>/dev/null
    /bin/rmdir "$LOCK_PATH" 2>/dev/null
  fi
  schedule_close_launcher_window
}

trap cleanup_launcher EXIT

show_error_and_wait() {
  print ""
  print "启动失败：$1"
  print ""
  read "?按回车键关闭此窗口..."
}

assistant_is_running() {
  if [[ ! -f "$PID_PATH" ]]; then
    return 1
  fi
  local existing_pid
  local existing_command
  existing_pid="$(<"$PID_PATH")"
  if [[ "$existing_pid" != <-> ]] || ! kill -0 "$existing_pid" 2>/dev/null; then
    /bin/rm -f "$PID_PATH" 2>/dev/null
    return 1
  fi
  existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null)"
  if [[ "$existing_command" == *"scripts.run_live_assistant --config configs/live_game.local.json"* ]]; then
    return 0
  fi
  /bin/rm -f "$PID_PATH" 2>/dev/null
  return 1
}

acquire_launch_lock() {
  /bin/mkdir -p "$PROJECT_DIR/logs"
  if /bin/mkdir "$LOCK_PATH" 2>/dev/null; then
    print "$$" > "$LOCK_PATH/pid"
    LAUNCH_LOCK_ACQUIRED=true
    return 0
  fi

  local lock_pid=""
  if [[ -f "$LOCK_PATH/pid" ]]; then
    lock_pid="$(<"$LOCK_PATH/pid")"
  fi
  if [[ "$lock_pid" != <-> ]] || ! kill -0 "$lock_pid" 2>/dev/null; then
    /bin/rm -f "$LOCK_PATH/pid" 2>/dev/null
    /bin/rmdir "$LOCK_PATH" 2>/dev/null
    if /bin/mkdir "$LOCK_PATH" 2>/dev/null; then
      print "$$" > "$LOCK_PATH/pid"
      LAUNCH_LOCK_ACQUIRED=true
      return 0
    fi
  fi
  return 1
}

if [[ ! -d "$PROJECT_DIR" ]]; then
  show_error_and_wait "找不到项目目录：$PROJECT_DIR"
  exit 1
fi

if [[ ! -x "$PYTHON_PATH" ]]; then
  show_error_and_wait "找不到项目 Python：$PYTHON_PATH"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  show_error_and_wait "找不到实时配置：$CONFIG_PATH"
  exit 1
fi

if assistant_is_running; then
  /usr/bin/osascript -e \
    'display notification "助手已经在运行，无需重复启动" with title "斗地主助手"'
  exit 0
fi

if ! acquire_launch_lock; then
  /usr/bin/osascript -e \
    'display notification "助手正在启动，请勿重复双击" with title "斗地主助手"'
  exit 0
fi

# Close the double-click race: a second launcher can pass the first running
# check before the first process has written its PID, but not the launch lock.
if assistant_is_running; then
  /usr/bin/osascript -e \
    'display notification "助手已经在运行，无需重复启动" with title "斗地主助手"'
  exit 0
fi

cd "$PROJECT_DIR" || {
  show_error_and_wait "无法进入项目目录"
  exit 1
}

clear
print -n $'\e]0;斗地主助手启动器\a'
print "正在启动斗地主助手..."
print "助手可以先启动并等待斗地主窗口。"
print "启动成功后，本终端窗口会自动关闭。"
print "关闭左侧助手小窗即可停止运行。"
print ""

/usr/bin/nohup "$PYTHON_PATH" -m scripts.run_live_assistant \
  --config configs/live_game.local.json \
  --pid-file "$PID_PATH" \
  >> "$LOG_PATH" 2>&1 </dev/null &
assistant_pid=$!
disown "$assistant_pid" 2>/dev/null || true
sleep 2

if ! kill -0 "$assistant_pid" >/dev/null 2>&1; then
  print "最近错误："
  tail -n 12 "$LOG_PATH" 2>/dev/null
  show_error_and_wait "助手进程未能保持运行"
  exit 1
fi
/usr/bin/osascript -e \
  'display notification "实时识别小窗已经启动" with title "斗地主助手"'
exit 0
