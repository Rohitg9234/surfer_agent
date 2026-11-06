import platform
import subprocess


def refresh_last_chrome_tab():
  system = platform.system()

  if system == "Darwin":  # macOS
    applescript = """
        tell application "Google Chrome"
            set frontWindow to front window
            set numTabs to count of tabs of frontWindow
            set active tab index of frontWindow to numTabs
            reload active tab of frontWindow
        end tell
        """
    subprocess.run(["osascript", "-e", applescript])
    print("macOS: Switched to the last tab and refreshed.")

  elif system == "Windows":  # Windows
    # PowerShell uses WScript.Shell to target Chrome and send keyboard shortcuts
    # Ctrl+9 jumps to the last tab in Chrome, Ctrl+R refreshes it.
    ps_script = """
        $wshell = New-Object -ComObject WScript.Shell
        if ($wshell.AppActivate("Google Chrome")) {
            Start-Sleep -Milliseconds 50
            $wshell.SendKeys("^{9}")
            Start-Sleep -Milliseconds 50
            $wshell.SendKeys("^r")
        }
        """
    subprocess.run(["powershell", "-Command", ps_script], shell=True)
    print("Windows: Switched to the last tab and refreshed.")

  elif system == "Linux":  # Linux
    # Requires xdotool (sudo apt install xdotool)
    subprocess.run([
        "xdotool",
        "search",
        "--class",
        "google-chrome",
        "windowactivate",
        "--sync",
        "key",
        "ctrl+9",
        "key",
        "ctrl+r",
    ])
    print("Linux: Switched to the last tab and refreshed.")

  else:
    print(f"Unsupported operating system: {system}")


if __name__ == "__main__":
  refresh_last_chrome_tab()