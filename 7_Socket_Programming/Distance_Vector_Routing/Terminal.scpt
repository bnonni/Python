#!/usr/bin/osascript
on run argv
    tell application "Terminal"
        activate
        do script "sudo python3 " & argv as string
    end tell
end run